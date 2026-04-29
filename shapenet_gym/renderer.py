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


class MeshRenderer:
    """Thin wrapper around pyrender for off-screen rendering.

    Parameters
    ----------
    image_size:
        (height, width) of the output image in pixels.
    offscreen:
        If True (default), render without a display (requires osmesa or egl).
        Set False only when a real display is available.
    background_color:
        RGBA background colour (floats 0–1).
    ambient_light:
        Ambient light intensity.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        offscreen: bool = True,
        background_color: Tuple[float, ...] = (0.1, 0.1, 0.1, 1.0),
        ambient_light: float = 0.4,
    ):
        self.image_size = image_size  # (H, W)
        self.offscreen = offscreen
        self.background_color = np.array(background_color, dtype=np.float32)
        self.ambient_light = ambient_light

        self._scene = None
        self._renderer = None
        self._mesh_node = None
        self._camera_node = None

        # Choose platform before importing pyrender.
        # On Linux headless we need osmesa/egl; on macOS PyOpenGL auto-selects
        # the native CGL backend which works offscreen via pyglet.
        import sys
        if (
            offscreen
            and "PYOPENGL_PLATFORM" not in os.environ
            and sys.platform != "darwin"
        ):
            os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

        self._init_scene()

    # ------------------------------------------------------------------
    def _init_scene(self):
        import pyrender
        import trimesh

        H, W = self.image_size

        self._scene = pyrender.Scene(
            bg_color=self.background_color,
            ambient_light=np.array([self.ambient_light] * 3),
        )

        # Directional lights for good 3D shape perception
        for direction in [
            [1.0, 1.0, 1.0],
            [-1.0, -0.5, 0.5],
            [0.0, -1.0, 0.3],
        ]:
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            pose = self._look_at(
                eye=np.array(direction, dtype=np.float64),
                center=np.zeros(3),
                up=np.array([0.0, 0.0, 1.0]),
            )
            self._scene.add(light, pose=pose)

        # Perspective camera (will be repositioned each render call)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=W / H)
        self._camera_node = self._scene.add(camera, pose=np.eye(4))

        # Off-screen renderer
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=W,
            viewport_height=H,
        )

    # ------------------------------------------------------------------
    def load_mesh(self, mesh_path: Path | str):
        """Load a new OBJ mesh, replacing the previous one in the scene."""
        import pyrender
        import trimesh

        mesh_path = Path(mesh_path)

        # Remove previous mesh node
        if self._mesh_node is not None:
            self._scene.remove_node(self._mesh_node)
            self._mesh_node = None

        # Load & normalise the mesh
        loaded = trimesh.load(str(mesh_path), force="mesh", process=True)

        # Handle scenes with multiple geometries
        if isinstance(loaded, trimesh.Scene):
            geometries = list(loaded.geometry.values())
            if not geometries:
                raise ValueError(f"No geometry found in {mesh_path}")
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = loaded

        # ShapeNet meshes use Y-up; our world is Z-up. Rotate so the
        # object's natural up direction aligns with the camera's up vector.
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(
                np.pi / 2, [1.0, 0.0, 0.0]
            )
        )

        # Centre and scale to unit sphere
        mesh.vertices -= mesh.bounding_box.centroid
        scale = mesh.bounding_sphere.primitive.radius
        if scale > 0:
            mesh.vertices /= scale

        # Apply a default material if none exists
        if not hasattr(mesh.visual, "material") or mesh.visual.material is None:
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.tile([180, 180, 200, 255], (len(mesh.vertices), 1)),
            )

        # Some ShapeNet meshes ship absurdly large textures (e.g. 32768×4096)
        # that exceed GL_MAX_TEXTURE_SIZE. Downscale on the fly.
        _MAX_TEX = 2048
        material = getattr(mesh.visual, "material", None)
        if material is not None:
            for attr in ("image", "baseColorTexture"):
                img = getattr(material, attr, None)
                if img is not None and hasattr(img, "size"):
                    w, h = img.size
                    if max(w, h) > _MAX_TEX:
                        scale = _MAX_TEX / max(w, h)
                        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
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
        """Render the current scene from the given camera position.

        Parameters
        ----------
        camera_pos:
            3D position of the camera.
        look_at:
            3D point the camera is looking at.
        up:
            Up vector for the camera.

        Returns
        -------
        np.ndarray
            Shape (H, W, 3), dtype uint8.
        """
        import pyrender

        cam_pose = self._look_at(camera_pos.astype(np.float64),
                                  look_at.astype(np.float64),
                                  up.astype(np.float64))
        self._scene.set_pose(self._camera_node, cam_pose)

        color, _ = self._renderer.render(
            self._scene,
            flags=pyrender.RenderFlags.RGBA,
        )
        return color[:, :, :3]  # drop alpha

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
        pose[:3, 2] = -f   # OpenGL convention: camera looks along -Z
        pose[:3, 3] = eye
        return pose
