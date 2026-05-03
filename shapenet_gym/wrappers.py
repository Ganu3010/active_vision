"""
Gymnasium Wrappers
==================
Convenience wrappers that extend ShapeNetViewEnv for common training setups.
"""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


class GrayscaleWrapper(gym.ObservationWrapper):
    """Convert RGB observations to single-channel grayscale.

    Reduces observation size by 3× and is often sufficient for shape
    classification tasks.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        gray = (
            0.2989 * obs[:, :, 0]
            + 0.5870 * obs[:, :, 1]
            + 0.1140 * obs[:, :, 2]
        ).astype(np.uint8)
        return gray[:, :, np.newaxis]


class ResizeWrapper(gym.ObservationWrapper):
    """Resize observations to a smaller resolution.

    Useful for speeding up training when high resolution isn't needed.

    Parameters
    ----------
    env:
        The environment to wrap.
    size:
        Target (height, width).
    """

    def __init__(self, env: gym.Env, size: Tuple[int, int] = (84, 84)):
        super().__init__(env)
        import cv2  # opencv-python
        self._cv2 = cv2
        self._size = size  # (H, W)
        h, w = size
        c = env.observation_space.shape[2] if len(env.observation_space.shape) == 3 else 1
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        h, w = self._size
        resized = self._cv2.resize(obs, (w, h), interpolation=self._cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        return resized


class ChannelFirstWrapper(gym.ObservationWrapper):
    """Transpose (H, W, C) observations to (C, H, W) for PyTorch CNNs."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.transpose(obs, (2, 0, 1))


class NormalizeWrapper(gym.ObservationWrapper):
    """Normalise pixel values to [0, 1] float32."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=shape, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0


class ViewpointAugmentationWrapper(gym.Wrapper):
    """Randomly perturb the initial viewpoint on each reset.

    Adds diversity to the starting states seen during training, which
    can help prevent the agent from overfitting to a fixed starting position.

    Parameters
    ----------
    theta_noise_deg:
        Standard deviation of Gaussian noise added to the initial elevation.
    phi_noise_deg:
        Standard deviation of Gaussian noise added to the initial azimuth.
    """

    def __init__(
        self,
        env: gym.Env,
        theta_noise_deg: float = 10.0,
        phi_noise_deg: float = 30.0,
    ):
        super().__init__(env)
        import math
        self._theta_noise = math.radians(theta_noise_deg)
        self._phi_noise = math.radians(phi_noise_deg)

    def reset(self, **kwargs):
        import math
        obs, info = self.env.reset(**kwargs)
        # Add small random perturbation to the initial viewpoint
        self.env.unwrapped._theta += float(
            np.random.normal(0, self._theta_noise)
        )
        self.env.unwrapped._phi = (
            self.env.unwrapped._phi + float(np.random.normal(0, self._phi_noise))
        ) % (2 * math.pi)
        lower = 0.0 if getattr(self.env.unwrapped, "_upper_hemisphere_only", False) else -math.pi / 2 + 1e-4
        self.env.unwrapped._theta = np.clip(
            self.env.unwrapped._theta, lower, math.pi / 2 - 1e-4
        )
        # Re-render after perturbation
        obs = self.env.unwrapped._render_observation()
        return obs, info


class YoloPoseObservationWrapper(gym.Wrapper):
    """Augment the image observation with YOLO probs, camera pose, and summary
    stats — the multi-input observation specified in the project proposal.

    Requires the underlying env to have a `yolo_model` set (it caches probs on
    `env._last_yolo_probs` after each render). Apply this wrapper *last*, so it
    sees the already-resized / channel-first image as the `image` key.

    Output observation:
        {
          "image":      same Box as the inner env (e.g. (3, 84, 84) uint8)
          "yolo_probs": Box(0, 1, (1000,), float32)
          "pose":       Box(-1, 1, (4,), float32)  -> (sin θ, cos θ, sin φ, cos φ)
          "summary":    Box(0, 7, (3,), float32)   -> (entropy, top1_prob, top1_top2_margin)
        }
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if getattr(env.unwrapped, "_yolo", None) is None:
            raise ValueError(
                "YoloPoseObservationWrapper requires the underlying env to "
                "have yolo_model set. Pass yolo_model=... to ShapeNetViewEnv."
            )
        self._n_classes = 1000
        self.observation_space = gym.spaces.Dict({
            "image": env.observation_space,
            "yolo_probs": gym.spaces.Box(
                low=0.0, high=1.0, shape=(self._n_classes,), dtype=np.float32,
            ),
            "pose": gym.spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
            ),
            "summary": gym.spaces.Box(
                low=0.0, high=float(np.log(self._n_classes)) + 1e-3,
                shape=(3,), dtype=np.float32,
            ),
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._build_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._build_obs(obs), reward, terminated, truncated, info

    def _build_obs(self, image_obs: np.ndarray) -> dict:
        import math
        env = self.env.unwrapped
        probs = env._last_yolo_probs
        if probs is None:
            probs = np.full(self._n_classes, 1.0 / self._n_classes, dtype=np.float32)

        clipped = np.maximum(probs, 1e-9)
        H = float(-(probs * np.log(clipped)).sum())
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])

        theta, phi = env._theta, env._phi
        pose = np.array(
            [math.sin(theta), math.cos(theta), math.sin(phi), math.cos(phi)],
            dtype=np.float32,
        )
        summary = np.array([H, float(sorted_probs[0]), margin], dtype=np.float32)

        return {
            "image": image_obs,
            "yolo_probs": probs.astype(np.float32),
            "pose": pose,
            "summary": summary,
        }


def make_training_env(
    dataset_root: str,
    image_size: Tuple[int, int] = (84, 84),
    max_steps: int = 20,
    categories: Optional[list] = None,
    reward_fn=None,
    channel_first: bool = True,
    normalize: bool = True,
    grayscale: bool = False,
    seed: Optional[int] = None,
    yolo_model=None,
    upper_hemisphere_only: bool = False,
    split: str = "all",
) -> gym.Env:
    """
    Factory that builds a fully-wrapped ShapeNetViewEnv ready for training.

    Parameters
    ----------
    dataset_root:
        Path to local ShapeNetCore directory.
    image_size:
        Output image size (H, W) after resizing.
    max_steps:
        Episode length.
    categories:
        Optional list of synset IDs or category names.
    reward_fn:
        Reward function. If None, the default (0 everywhere) is used.
    channel_first:
        If True, transpose observations to (C, H, W) for PyTorch.
    normalize:
        If True, normalise pixel values to [0, 1] float32.
    grayscale:
        If True, convert to single-channel grayscale.
    seed:
        Random seed.

    Returns
    -------
    gym.Env
        Fully-wrapped environment.
    """
    from .env import ShapeNetViewEnv

    env = ShapeNetViewEnv(
        dataset_root=dataset_root,
        max_steps=max_steps,
        image_size=(224, 224),  # render at high res, then resize
        categories=categories,
        reward_fn=reward_fn,
        seed=seed,
        offscreen=True,
        yolo_model=yolo_model,
        upper_hemisphere_only=upper_hemisphere_only,
        split=split,
    )
    env = ViewpointAugmentationWrapper(env)
    if grayscale:
        env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, size=image_size)
    if channel_first:
        env = ChannelFirstWrapper(env)
    if normalize:
        env = NormalizeWrapper(env)
    if yolo_model is not None:
        env = YoloPoseObservationWrapper(env)
    return env
