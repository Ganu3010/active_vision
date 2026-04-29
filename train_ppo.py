"""
PPO training for the ShapeNet active-vision environment.

Reward:  classifier_entropy_reward using YOLOv8 classification logits —
         higher reward when the agent moves to a view that makes YOLO more
         certain about the object.

Usage:
    python train_ppo.py --max_steps 50
    python train_ppo.py --max_steps 100  --total_timesteps 200000
    python train_ppo.py --max_steps 150  --use_wandb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from shapenet_gym.wrappers import make_training_env


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------
def make_yolo_entropy_reward(
    scale: float = 10.0,
    motion_cost: float = 0.05,
    correctness_bonus: float = 0.0,
    incorrect_penalty: float = 0.0,
):
    """Reward = scale * (prev_entropy - current_entropy)
              - motion_cost (on non-STAY actions)
              + correctness_bonus  (top-1 in accepted set)
              - incorrect_penalty  (top-1 outside accepted set)

    Bonus / penalty are skipped for synsets without an entry in
    `SYNSET_TO_IMAGENET_INDICES` so the agent isn't punished for things
    outside the labeled set.
    """
    from shapenet_gym.env import STAY
    from shapenet_gym.labels import SYNSET_TO_IMAGENET_INDICES

    prev_entropy = [None]

    def reward_fn(env, action, terminated):
        probs = env._last_yolo_probs
        if probs is None:
            return 0.0
        clipped = np.maximum(probs, 1e-9)
        H = float(-(probs * np.log(clipped)).sum())

        if prev_entropy[0] is None:
            prev_entropy[0] = H
            delta = 0.0
        else:
            delta = prev_entropy[0] - H
            prev_entropy[0] = H

        if terminated:
            prev_entropy[0] = None

        cost = 0.0 if action == STAY else motion_cost

        correctness_term = 0.0
        if correctness_bonus > 0.0 or incorrect_penalty > 0.0:
            accepted = SYNSET_TO_IMAGENET_INDICES.get(env.current_synset_id, set())
            if accepted:
                if int(np.argmax(probs)) in accepted:
                    correctness_term = correctness_bonus
                else:
                    correctness_term = -incorrect_penalty

        return scale * delta - cost + correctness_term

    return reward_fn


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------
def make_env(
    dataset_root: str,
    max_steps: int,
    yolo,
    categories: list[str] | None = None,
    seed: int | None = None,
    upper_hemisphere_only: bool = False,
):
    """Build one PPO-ready env with multi-input observation (image + YOLO
    probs + pose + summary stats)."""
    reward_fn = make_yolo_entropy_reward(
        scale=10.0, correctness_bonus=4.0, incorrect_penalty=1.0,
    )

    env = make_training_env(
        dataset_root=dataset_root,
        image_size=(84, 84),       # policy CNN input
        max_steps=max_steps,
        categories=categories,
        reward_fn=reward_fn,
        channel_first=True,        # (C, H, W) for PyTorch
        normalize=False,           # keep uint8 — CombinedExtractor /255 internally
        grayscale=False,
        seed=seed,
        yolo_model=yolo,           # enables YoloPoseObservationWrapper
        upper_hemisphere_only=upper_hemisphere_only,
    )
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[train_ppo] device={device} max_steps={args.max_steps} "
          f"total_timesteps={args.total_timesteps}")

    # wandb (optional)
    wandb_run = None
    callback = None
    if args.use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb_run = wandb.init(
            project="active-view-rl",
            name=f"ppo_maxsteps{args.max_steps}",
            config=vars(args),
            sync_tensorboard=True,
            save_code=True,
        )
        callback = WandbCallback(verbose=2)

    # One shared YOLO instance — used by both the env (for the cached probs in
    # the multi-input observation) and the entropy reward.
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)

    # Build env — wrap in DummyVecEnv (SB3 requires a VecEnv)
    env = DummyVecEnv([
        lambda: make_env(
            dataset_root=args.dataset_root,
            max_steps=args.max_steps,
            yolo=yolo,
            categories=args.categories,
            seed=args.seed,
            upper_hemisphere_only=args.upper_hemisphere_only,
        )
    ])

    # MultiInputPolicy — Dict observation (image + yolo_probs + pose + summary).
    # SB3's CombinedExtractor uses NatureCNN for "image" and small MLPs for the
    # vector keys, then concatenates.
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        device=device,
        n_steps=min(2048, args.total_timesteps),
        batch_size=64,
        learning_rate=2e-4,
        gamma=0.99,
        tensorboard_log=args.logdir if args.use_wandb else None,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    # Save
    save_path = Path(args.save_dir) / f"ppo_maxsteps{args.max_steps}.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"[train_ppo] saved model → {save_path}")

    if wandb_run is not None:
        wandb_run.finish()

    env.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="data/ShapeNetCore")
    p.add_argument("--yolo_weights", type=str, default="yolov8n-cls.pt")
    p.add_argument("--max_steps", type=int, default=50,
                   help="Episode length (max_iter in the task description).")
    p.add_argument("--total_timesteps", type=int, default=100_000)
    p.add_argument("--save_dir", type=str, default="outputs/ppo")
    p.add_argument("--logdir", type=str, default="outputs/tb")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--categories",
        nargs="+",
        default=["02691156", "02958343"],  # airplane, car
        help="Synset IDs to train on. Pass --categories '' to use all extracted folders.",
    )
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--upper_hemisphere_only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
