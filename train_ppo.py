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
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from shapenet_gym.wrappers import make_training_env


# ---------------------------------------------------------------------------
# YOLOv8 classifier loader
# ---------------------------------------------------------------------------
def make_yolo_entropy_reward(weights_path: str, scale: float = 100.0):
    """Reward = `scale * (prev_entropy - current_entropy)` using YOLOv8-cls.

    Uses ultralytics' high-level predict API which handles preprocessing,
    device placement, and the cls head correctly. Calling `yolo.model(x)`
    directly returns wrapped output that doesn't behave as logits.
    """
    from ultralytics import YOLO

    yolo = YOLO(weights_path)
    prev_entropy = [None]

    def reward_fn(env, action, terminated):
        obs = env._render_observation()  # (H, W, 3) uint8
        results = yolo.predict(obs, verbose=False)
        probs = results[0].probs.data  # tensor shape (1000,)
        H = -(probs.clamp(min=1e-9) * probs.clamp(min=1e-9).log()).sum().item()

        if prev_entropy[0] is None:
            prev_entropy[0] = H
            return 0.0

        delta = prev_entropy[0] - H
        prev_entropy[0] = H
        if terminated:
            prev_entropy[0] = None  # reset for next episode

        return scale * delta

    return reward_fn


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------
def make_env(
    dataset_root: str,
    max_steps: int,
    yolo_weights: str,
    device: str,
    categories: list[str] | None = None,
    seed: int | None = None,
):
    """Build one PPO-ready env. Note: we keep observations as uint8 (C, H, W)
    and let SB3's CnnPolicy handle normalisation — that matches the
    NatureCNN default and avoids double-scaling."""
    reward_fn = make_yolo_entropy_reward(yolo_weights, scale=10.0)

    env = make_training_env(
        dataset_root=dataset_root,
        image_size=(84, 84),       # PPO policy input
        max_steps=max_steps,
        categories=categories,     # pin to specific synsets if provided
        reward_fn=reward_fn,
        channel_first=True,        # (C, H, W) for PyTorch
        normalize=False,           # keep uint8 — SB3 CnnPolicy /255 internally
        grayscale=False,
        seed=seed,
    )
    env = Monitor(env)             # logs episode reward/length
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

    # Build env — wrap in DummyVecEnv (SB3 requires a VecEnv)
    env = DummyVecEnv([
        lambda: make_env(
            dataset_root=args.dataset_root,
            max_steps=args.max_steps,
            yolo_weights=args.yolo_weights,
            device=device,
            categories=args.categories,
            seed=args.seed,
        )
    ])

    # PPO with CnnPolicy — input is (3, 84, 84) uint8, policy uses NatureCNN.
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        device=device,
        n_steps=min(2048, args.total_timesteps),
        batch_size=64,
        learning_rate=1e-4,
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
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
