"""
Reward Functions
================
Drop-in reward functions for ShapeNetViewEnv.

Pass any of these as the ``reward_fn`` constructor argument:

    env = ShapeNetViewEnv(
        dataset_root="...",
        reward_fn=classifier_entropy_reward(model, preprocess),
    )

All reward functions follow the signature:
    fn(env: ShapeNetViewEnv, action: int, terminated: bool) -> float
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .env import ShapeNetViewEnv


# ---------------------------------------------------------------------------
# 1. Sparse oracle reward
# ---------------------------------------------------------------------------
def sparse_reward(
    env: "ShapeNetViewEnv", action: int, terminated: bool
) -> float:
    """
    +1 at the end of an episode; 0 otherwise.
    Useful as a baseline — the agent must learn to get to a good view
    *before* the episode ends, with no intermediate guidance.
    """
    return 1.0 if terminated else 0.0


# ---------------------------------------------------------------------------
# 2. Classifier entropy reward (information gain)
# ---------------------------------------------------------------------------
def classifier_entropy_reward(
    model,
    preprocess,
    device: str = "cpu",
    scale: float = 1.0,
) -> Callable:
    """
    Reward = reduction in classifier entropy compared to the previous step.

    A positive reward means the agent moved to a view that made the
    classifier *more* certain; a negative reward means it became less certain.

    Parameters
    ----------
    model:
        A PyTorch model that accepts a batched image tensor and returns logits.
        Should already be in eval mode.
    preprocess:
        torchvision transform that converts a (H, W, 3) uint8 numpy array
        to a (1, C, H, W) float tensor suitable for `model`.
    device:
        Torch device string.
    scale:
        Multiply the entropy delta by this factor.

    Usage
    -----
    >>> import torchvision.models as models
    >>> import torchvision.transforms as T
    >>> resnet = models.resnet50(pretrained=True).eval()
    >>> preprocess = T.Compose([T.ToTensor(),
    ...                         T.Normalize([0.485,0.456,0.406],
    ...                                     [0.229,0.224,0.225])])
    >>> env = ShapeNetViewEnv("data/", reward_fn=classifier_entropy_reward(resnet, preprocess))
    """
    import torch
    import torch.nn.functional as F

    _prev_entropy = [None]

    def _entropy(obs: np.ndarray) -> float:
        img = preprocess(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img)
            probs = F.softmax(logits, dim=-1)
            H = -(probs * probs.clamp(min=1e-9).log()).sum().item()
        return H

    def reward_fn(env: "ShapeNetViewEnv", action: int, terminated: bool) -> float:
        obs = env._render_observation()
        current_entropy = _entropy(obs)

        if _prev_entropy[0] is None:
            _prev_entropy[0] = current_entropy
            return 0.0

        delta = _prev_entropy[0] - current_entropy  # positive = more certain
        _prev_entropy[0] = current_entropy

        if terminated:
            _prev_entropy[0] = None  # reset for next episode

        return scale * delta

    return reward_fn


# ---------------------------------------------------------------------------
# 3. Step penalty (encourage efficiency)
# ---------------------------------------------------------------------------
def step_penalty_reward(
    penalty: float = -0.01,
    terminal_bonus: float = 1.0,
) -> Callable:
    """
    Small negative reward each step to encourage the agent to reach a
    good viewpoint quickly, with a bonus at episode end.

    Parameters
    ----------
    penalty:
        Reward applied at every non-terminal step (should be negative).
    terminal_bonus:
        Reward added at the terminal step.
    """
    def reward_fn(env: "ShapeNetViewEnv", action: int, terminated: bool) -> float:
        return terminal_bonus if terminated else penalty

    return reward_fn


# ---------------------------------------------------------------------------
# 4. Oracle best-view reward (requires ground-truth labels + classifier)
# ---------------------------------------------------------------------------
def oracle_best_view_reward(
    model,
    preprocess,
    label_map: dict,
    device: str = "cpu",
) -> Callable:
    """
    +1 if at episode end the classifier correctly identifies the object,
    -1 if not, 0 otherwise.

    Parameters
    ----------
    model:
        Classification model.
    preprocess:
        Image preprocessing transform.
    label_map:
        Dict mapping synset_id → class index in the model's output.
    device:
        Torch device.
    """
    import torch
    import torch.nn.functional as F

    def reward_fn(env: "ShapeNetViewEnv", action: int, terminated: bool) -> float:
        if not terminated:
            return 0.0

        obs = env._render_observation()
        synset_id = env.current_synset_id
        true_class = label_map.get(synset_id)

        if true_class is None:
            return 0.0  # unknown class, skip

        import torchvision
        img = preprocess(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img)
            pred = logits.argmax(dim=-1).item()

        return 1.0 if pred == true_class else -1.0

    return reward_fn
