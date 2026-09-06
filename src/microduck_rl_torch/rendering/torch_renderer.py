"""Torch-backed RGB renderer."""

from __future__ import annotations

from typing import Any

import mujoco
import mujoco_torch
import numpy as np
import torch

from microduck_rl_torch.envs.model import ModelBundle

from .config import CameraConfig


class TorchRenderer:
    """Render RGB pixels through the pure ``mujoco-torch`` ray renderer."""

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        width: int,
        height: int,
        camera: CameraConfig,
        ray_chunk_size: int,
    ):
        if camera.name is None:
            raise ValueError("The Torch renderer requires a named MuJoCo camera")
        camera_id = mujoco.mj_name2id(bundle.native_model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name)
        if camera_id < 0:
            raise ValueError(f"Camera {camera.name!r} was not found in the model")
        self._bundle = bundle
        self._width = width
        self._height = height
        self._camera_id = int(camera_id)
        self._ray_chunk_size = ray_chunk_size
        self._precomp = mujoco_torch.precompute_render_data(bundle.torch_model)

    def render(self, env: Any) -> np.ndarray:
        state = env.data
        if state is None:
            raise RuntimeError("Environment must be reset before rendering")
        rgb, _depth, _segmentation = mujoco_torch.render(
            self._bundle.torch_model,
            state,
            camera_id=self._camera_id,
            width=self._width,
            height=self._height,
            precomp=self._precomp,
            ray_chunk_size=self._ray_chunk_size,
        )
        return rgb.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()

    def close(self) -> None:
        """Release renderer-owned resources.

        The Torch renderer currently keeps only model-level precomputation,
        which is owned by Python and needs no explicit teardown.
        """
