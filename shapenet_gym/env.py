"""
ShapeNetCore Gymnasium Environment
===================================
An RL environment where an agent navigates a spherical viewpoint around
3D objects from the ShapeNetCore dataset, observing 2D rendered images.

Action Space:
    Discrete(4) → [UP, DOWN, LEFT, RIGHT]

Observation Space:
    Box(H, W, 3, dtype=uint8) — rendered RGB image from current viewpoint

Reward:
    Configurable (see reward_fn). Default: 0 every step, +1 on correct
    classification at episode end (suitable for object-ID training).
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .renderer import MeshRenderer
from .dataset import ShapeNetDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class ShapeNetViewEnv(gym.Env):
    """Gymnasium environment for viewpoint optimisation on ShapeNetCore.

    Parameters
    ----------
    dataset_root:
        Path to the local ShapeNetCore directory (downloaded from HuggingFace).
    max_steps:
        Number of steps before the episode ends and the next object is loaded.
    image_size:
        (height, width) of the rendered observation image.
    theta_step_deg:
        Angular step (degrees) for elevation moves (UP/DOWN).
    phi_step_deg:
        Angular step (degrees) for azimuth moves (LEFT/RIGHT).
    radius:
        Distance from the camera to the object centre (world units).
    reward_fn:
        Optional callable ``(env, action, terminated) → float``.
        If None the default sparse reward is used.
    categories:
        List of ShapeNet synset IDs to include. ``None`` = all categories.
    seed:
        Random seed for reproducibility.
    render_mode:
        ``"rgb_array"`` (default) or ``"human"`` (opens a window).
    offscreen:
        Force off-screen rendering even in human mode (useful on servers).
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    # ------------------------------------------------------------------
    def __init__(
        self,
        dataset_root: str | Path,
        max_steps: int = 20,
        image_size: Tuple[int, int] = (224, 224),
        theta_step_deg: float = 15.0,
        phi_step_deg: float = 15.0,
        radius: float = 3.0,
        reward_fn: Optional[Callable] = None,
        categories: Optional[list[str]] = None,
        seed: Optional[int] = None,
        render_mode: str = "rgb_array",
        offscreen: bool = True,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.max_steps = max_steps
        self.image_size = image_size  # (H, W)
        self.theta_step = math.radians(theta_step_deg)
        self.phi_step = math.radians(phi_step_deg)
        self.radius = radius
        self.reward_fn = reward_fn or self._default_reward
        self.render_mode = render_mode
        self.offscreen = offscreen

        # Action & observation spaces
        self.action_space = spaces.Discrete(4)
        H, W = image_size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(H, W, 3), dtype=np.uint8
        )

        # Dataset & renderer (lazy-initialised on first reset)
        self._dataset: Optional[ShapeNetDataset] = None
        self._renderer: Optional[MeshRenderer] = None
        self._categories = categories

        # Episode state
        self._theta: float = 0.0        # elevation angle (radians, 0 = equator)
        self._phi: float = 0.0          # azimuth angle (radians)
        self._steps: int = 0
        self._current_obj: Optional[dict] = None  # metadata dict from dataset
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Lazy-init dataset & renderer
        if self._dataset is None:
            self._dataset = ShapeNetDataset(
                self.dataset_root, categories=self._categories
            )
        if self._renderer is None:
            self._renderer = MeshRenderer(
                image_size=self.image_size,
                offscreen=self.offscreen,
            )

        # Pick a new object
        self._current_obj = self._dataset.sample(rng=self._rng)
        self._renderer.load_mesh(self._current_obj["mesh_path"])

        # Random initial viewpoint
        self._theta = float(self._rng.uniform(-math.pi / 3, math.pi / 3))
        self._phi = float(self._rng.uniform(0, 2 * math.pi))
        self._steps = 0

        obs = self._render_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Move observer on the sphere
        self._apply_action(action)
        self._steps += 1

        terminated = self._steps >= self.max_steps
        truncated = False

        obs = self._render_observation()
        reward = float(self.reward_fn(self, action, terminated))
        info = self._build_info()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._render_observation()
        elif self.render_mode == "human":
            self._renderer.show_window(self._camera_position())

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_action(self, action: int):
        """Update spherical coordinates based on the chosen action."""
        if action == UP:
            self._theta = min(self._theta + self.theta_step, math.pi / 2 - 1e-4)
        elif action == DOWN:
            self._theta = max(self._theta - self.theta_step, -math.pi / 2 + 1e-4)
        elif action == LEFT:
            self._phi = (self._phi - self.phi_step) % (2 * math.pi)
        elif action == RIGHT:
            self._phi = (self._phi + self.phi_step) % (2 * math.pi)

    def _camera_position(self) -> np.ndarray:
        """Convert (radius, theta, phi) → Cartesian camera position."""
        r = self.radius
        t, p = self._theta, self._phi
        x = r * math.cos(t) * math.cos(p)
        y = r * math.cos(t) * math.sin(p)
        z = r * math.sin(t)
        return np.array([x, y, z], dtype=np.float32)

    def _render_observation(self) -> np.ndarray:
        cam_pos = self._camera_position()
        return self._renderer.render(
            camera_pos=cam_pos,
            look_at=np.zeros(3, dtype=np.float32),
            up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

    def _build_info(self) -> dict:
        return {
            "synset_id": self._current_obj.get("synset_id", ""),
            "model_id": self._current_obj.get("model_id", ""),
            "category": self._current_obj.get("category", ""),
            "theta_deg": math.degrees(self._theta),
            "phi_deg": math.degrees(self._phi),
            "step": self._steps,
        }

    # ------------------------------------------------------------------
    # Default reward
    # ------------------------------------------------------------------
    @staticmethod
    def _default_reward(env: "ShapeNetViewEnv", action: int, terminated: bool) -> float:
        """
        Sparse baseline reward — returns 0 every step.

        Override via ``reward_fn`` for task-specific signals, e.g.:
          - Entropy reduction of a classification posterior
          - Information gain over a belief distribution
          - Oracle reward (+1 if agent picks the best view)
        """
        return 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def current_category(self) -> str:
        return self._current_obj.get("category", "") if self._current_obj else ""

    @property
    def current_synset_id(self) -> str:
        return self._current_obj.get("synset_id", "") if self._current_obj else ""

    @property
    def camera_position(self) -> np.ndarray:
        return self._camera_position()

    @property
    def viewpoint_angles(self) -> Tuple[float, float]:
        """Return (theta_deg, phi_deg) of the current viewpoint."""
        return math.degrees(self._theta), math.degrees(self._phi)
