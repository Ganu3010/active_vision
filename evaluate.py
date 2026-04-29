"""
Evaluate a trained PPO policy on the active-vision env.

Reports top-1 accuracy, AUCC, and steps-to-confidence over N episodes.

Usage:
    python evaluate.py --model_path outputs/ppo/ppo_maxsteps50.zip --episodes 50

Static / greedy baseline modes are not yet implemented — they will plug into
`run_episode` via the `policy_fn` argument.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from stable_baselines3 import PPO
from ultralytics import YOLO

from shapenet_gym.labels import SYNSET_TO_IMAGENET_INDICES, SYNSET_TO_NAME, has_mapping
from shapenet_gym.wrappers import make_training_env


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_episode(env, policy_fn: Callable, max_steps: int):
    """Run one episode under `policy_fn` and return per-step diagnostics.

    `policy_fn(obs) -> int` selects the next action.
    """
    obs, info = env.reset()
    synset_id = env.unwrapped.current_synset_id
    accepted = SYNSET_TO_IMAGENET_INDICES.get(synset_id, set())

    top1_probs: list[float] = []
    top1_correct: list[int] = []  # 1 if top-1 in accepted classes else 0

    for _ in range(max_steps):
        action = policy_fn(obs)
        obs, _reward, terminated, truncated, _info = env.step(int(action))

        probs = env.unwrapped._last_yolo_probs
        top1_idx = int(np.argmax(probs))
        top1_probs.append(float(probs[top1_idx]))
        top1_correct.append(1 if top1_idx in accepted else 0)

        if terminated or truncated:
            break

    return {
        "synset_id": synset_id,
        "category": SYNSET_TO_NAME.get(synset_id, synset_id),
        "has_mapping": has_mapping(synset_id),
        "top1_probs": top1_probs,
        "top1_correct": top1_correct,
    }


def aggregate(records: list[dict], threshold: float) -> dict:
    eligible = [r for r in records if r["has_mapping"]]
    n_total = len(records)
    n_eligible = len(eligible)

    final_correct = [r["top1_correct"][-1] for r in eligible if r["top1_correct"]]
    top1_accuracy = float(np.mean(final_correct)) if final_correct else 0.0

    aucc_per_ep = [float(np.mean(r["top1_probs"])) for r in records if r["top1_probs"]]
    aucc = float(np.mean(aucc_per_ep)) if aucc_per_ep else 0.0

    def steps_to_threshold(probs: list[float]) -> int:
        for i, p in enumerate(probs, start=1):
            if p > threshold:
                return i
        return len(probs) + 1  # never

    s2t = [steps_to_threshold(r["top1_probs"]) for r in records if r["top1_probs"]]
    steps_to_conf = float(np.mean(s2t)) if s2t else 0.0

    return {
        "n_episodes": n_total,
        "n_with_mapping": n_eligible,
        "top1_accuracy": top1_accuracy,
        "aucc": aucc,
        "steps_to_confidence": steps_to_conf,
        "confidence_threshold": threshold,
    }


def write_markdown(summary: dict, per_category: dict, out_path: Path):
    lines = [
        "# Evaluation Summary",
        "",
        f"- Episodes: **{summary['n_episodes']}** ({summary['n_with_mapping']} with ImageNet mapping)",
        f"- Top-1 accuracy: **{summary['top1_accuracy']:.3f}**",
        f"- AUCC (mean top-1 prob over episode): **{summary['aucc']:.3f}**",
        f"- Steps to top-1 > {summary['confidence_threshold']}: **{summary['steps_to_confidence']:.2f}**",
        "",
        "## Per-category breakdown",
        "",
        "| Category | Synset | Episodes | Top-1 acc | AUCC |",
        "|---|---|---:|---:|---:|",
    ]
    for synset, stats in sorted(per_category.items()):
        lines.append(
            f"| {stats['category']} | {synset} | {stats['n_episodes']} | "
            f"{stats['top1_accuracy']:.3f} | {stats['aucc']:.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, type=str)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--dataset_root", type=str, default="data/ShapeNetCore")
    p.add_argument("--yolo_weights", type=str, default="yolov8n-cls.pt")
    p.add_argument("--categories", nargs="+", default=["02691156", "02958343"])
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out_dir", type=str, default="outputs/eval")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = Path(args.model_path).stem

    device = pick_device()
    print(f"[evaluate] device={device} model={args.model_path} episodes={args.episodes}")

    yolo = YOLO(args.yolo_weights)
    env = make_training_env(
        dataset_root=args.dataset_root,
        image_size=(84, 84),
        max_steps=args.max_steps,
        categories=args.categories,
        reward_fn=None,
        channel_first=True,
        normalize=False,
        grayscale=False,
        seed=args.seed,
        yolo_model=yolo,
    )

    model = PPO.load(args.model_path, device=device)

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    records = [run_episode(env, ppo_policy, args.max_steps) for _ in range(args.episodes)]
    env.close()

    summary = aggregate(records, args.threshold)

    per_category = {}
    for synset in args.categories:
        sub = [r for r in records if r["synset_id"] == synset]
        if not sub:
            continue
        per_category[synset] = aggregate(sub, args.threshold)
        per_category[synset]["category"] = SYNSET_TO_NAME.get(synset, synset)

    json_path = out_dir / f"{run_name}.json"
    json_path.write_text(json.dumps({
        "config": vars(args),
        "summary": summary,
        "per_category": per_category,
        "records": records,
    }, indent=2))

    md_path = out_dir / f"{run_name}.md"
    write_markdown(summary, per_category, md_path)

    print(f"\n=== {run_name} ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nsaved: {json_path}")
    print(f"saved: {md_path}")


if __name__ == "__main__":
    main()
