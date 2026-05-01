"""
Mesh Renderer
=============
Renders 3D meshes to 2D RGB images using trimesh + pyrender.

Requirements:
    pip install trimesh pyrender

On headless servers, set the environment variable before importing:
    export PYOPENGL_PLATFORM=osmesa   # or "egl"

The renderer is created once and reused across episodes to avoid the
overhead of re-initialising the OpenGL context for every object.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# pyrender (≤0.1.45) still references np.infty, removed in NumPy 2.0.
if not hasattr(np, "infty"):
    np.infty = np.inf


GROUND_Z = -1.0
GROUND_EXTENT = 10.0


class MeshRenderer:
    """Off-screen pyrender wrapper with realistic-ish lighting and background.

    Scene layout: object centred at origin, normalised to unit sphere; ground
    plane at z=GROUND_Z; key+fill directional lights, key casts shadows;
    background composited as a vertical sky-ground gradient (lighter at +z
    image side).
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        offscreen: bool = True,
        ambient_light: float = 0.3,
        sky_color: Tuple[float, float, float] = (0.78, 0.82, 0.88),
        ground_color: Tuple[float, float, float] = (0.28, 0.26, 0.24),
        key_intensity: float = 4.0,
        fill_ratio: float = 0.3,
    ):
        self.image_size = image_size  # (H, W)
        self.offscreen = offscreen
        self.ambient_light = ambient_light
        self.sky_color = np.array(sky_color, dtype=np.float32)
        self.ground_color = np.array(ground_color, dtype=np.float32)
        self.key_intensity = key_intensity
        self.fill_ratio = fill_ratio

        self._scene = None
        self._renderer = None
        self._mesh_node = None
        self._camera_node = None
        self._gradient_bg = None  # precomputed (H, W, 3) uint8
        self._mesh_center = np.zeros(3, dtype=np.float32)

        import sys
        if (
            offscreen
            and "PYOPENGL_PLATFORM" not in os.environ
            and sys.platform != "darwin"
        ):
            os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

        self._init_scene()
        self._init_gradient()

    # ------------------------------------------------------------------
    def _init_scene(self):
        import pyrender
        import trimesh

        H, W = self.image_size

        self._scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ambient_light=np.array([self.ambient_light] * 3),
        )

        key_dir = np.array([-0.3, -0.5, -1.0])
        key_dir /= np.linalg.norm(key_dir)
        key_light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0], intensity=self.key_intensity
        )
        key_pose = self._look_at(
            eye=-key_dir * 3.0, center=np.zeros(3), up=np.array([0.0, 0.0, 1.0])
        )
        self._scene.add(key_light, pose=key_pose)

        fill_dir = np.array([0.6, 0.4, -0.4])
        fill_dir /= np.linalg.norm(fill_dir)
        fill_light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0],
            intensity=self.key_intensity * self.fill_ratio,
        )
        fill_pose = self._look_at(
            eye=-fill_dir * 3.0, center=np.zeros(3), up=np.array([0.0, 0.0, 1.0])
        )
        self._scene.add(fill_light, pose=fill_pose)

        ground = trimesh.creation.box(
            extents=[GROUND_EXTENT, GROUND_EXTENT, 0.02]
        )
        ground.apply_translation([0.0, 0.0, GROUND_Z - 0.01])
        ground.visual = trimesh.visual.ColorVisuals(
            mesh=ground,
            vertex_colors=np.tile(
                [int(self.ground_color[0] * 255),
                 int(self.ground_color[1] * 255),
                 int(self.ground_color[2] * 255), 255],
                (len(ground.vertices), 1),
            ),
        )
        pr_ground = pyrender.Mesh.from_trimesh(ground, smooth=False)
        self._scene.add(pr_ground, pose=np.eye(4))

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=W / H)
        self._camera_node = self._scene.add(camera, pose=np.eye(4))

        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=W,
            viewport_height=H,
        )

    # ------------------------------------------------------------------
    def _init_gradient(self):
        H, W = self.image_size
        t = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]  # 0=top
        col = (1.0 - t) * self.sky_color + t * self.ground_color
        bg = np.tile(col[:, None, :], (1, W, 1))
        self._gradient_bg = (bg * 255.0).clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def load_mesh(self, mesh_path: Path | str):
        """Load a new OBJ mesh, replacing the previous one in the scene."""
        import pyrender
        import trimesh

        mesh_path = Path(mesh_path)

        if self._mesh_node is not None:
            self._scene.remove_node(self._mesh_node)
            self._mesh_node = None

        loaded = trimesh.load(str(mesh_path), force="mesh", process=True)

        if isinstance(loaded, trimesh.Scene):
            geometries = list(loaded.geometry.values())
            if not geometries:
                raise ValueError(f"No geometry found in {mesh_path}")
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = loaded

        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(
                np.pi / 2, [1.0, 0.0, 0.0]
            )
        )

        mesh.vertices -= mesh.bounding_box.centroid
        scale = mesh.bounding_sphere.primitive.radius
        if scale > 0:
            mesh.vertices /= scale

        # Drop the object so its lowest point sits on the ground plane.
        mesh.vertices[:, 2] -= mesh.vertices[:, 2].min() - GROUND_Z
        self._mesh_center = mesh.bounding_box.centroid.astype(np.float32)

        if not hasattr(mesh.visual, "material") or mesh.visual.material is None:
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.tile([180, 180, 200, 255], (len(mesh.vertices), 1)),
            )

        _MAX_TEX = 2048
        material = getattr(mesh.visual, "material", None)
        if material is not None:
            for attr in ("image", "baseColorTexture"):
                img = getattr(material, attr, None)
                if img is not None and hasattr(img, "size"):
                    w, h = img.size
                    if max(w, h) > _MAX_TEX:
                        s = _MAX_TEX / max(w, h)
                        new_size = (max(1, int(w * s)), max(1, int(h * s)))
                        setattr(material, attr, img.resize(new_size))

        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        self._mesh_node = self._scene.add(pr_mesh, pose=np.eye(4))

    # ------------------------------------------------------------------
    def render(
        self,
        camera_pos: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray,
    ) -> np.ndarray:
        """Render the current scene from the given camera position."""
        import pyrender

        # Offset both camera and look-at by the mesh's post-drop centre so
        # the env's origin-relative spherical pose still frames the object.
        offset = self._mesh_center.astype(np.float64)
        cam_pose = self._look_at(camera_pos.astype(np.float64) + offset,
                                 look_at.astype(np.float64) + offset,
                                 up.astype(np.float64))
        self._scene.set_pose(self._camera_node, cam_pose)

        flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        color, _ = self._renderer.render(self._scene, flags=flags)

        rgb = color[:, :, :3].astype(np.float32)
        alpha = (color[:, :, 3:4].astype(np.float32)) / 255.0
        bg = self._gradient_bg.astype(np.float32)
        out = rgb * alpha + bg * (1.0 - alpha)
        return out.clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def close(self):
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None

    # ------------------------------------------------------------------
    @staticmethod
    def _look_at(
        eye: np.ndarray,
        center: np.ndarray,
        up: np.ndarray,
    ) -> np.ndarray:
        """Compute a 4×4 camera pose matrix from eye/center/up vectors."""
        f = center - eye
        norm = np.linalg.norm(f)
        if norm < 1e-8:
            f = np.array([0.0, 0.0, -1.0])
        else:
            f /= norm

        right = np.cross(f, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= right_norm

        true_up = np.cross(right, f)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = true_up
        pose[:3, 2] = -f
        pose[:3, 3] = eye
        return pose
