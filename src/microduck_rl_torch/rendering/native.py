"""Native MuJoCo RGB renderer for Torch-backed environments."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from microduck_rl_torch.envs.model import ModelBundle

from .config import CameraConfig


def _required_body_id(model: mujoco.MjModel, name: str, *, entity_name: str | None = None) -> int:
    candidates = (name, f"{entity_name}/{name}") if entity_name else (name,)
    for candidate in candidates:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if body_id >= 0:
            return int(body_id)
    raise ValueError(f"Camera tracking body {name!r} was not found in the model")


class NativeRenderer:
    """Render one Torch environment state with native MuJoCo OpenGL."""

    def __init__(self, bundle: ModelBundle, *, width: int, height: int, camera: CameraConfig):
        self.bundle = bundle
        self.camera_config = camera
        self._model = bundle.native_model
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=height, width=width)
        self._camera: str | mujoco.MjvCamera
        self._original_fovy: float | None = None

        if camera.name is not None:
            camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name)
            if camera_id < 0:
                raise ValueError(f"Camera {camera.name!r} was not found in the model")
            self._camera = camera.name
        else:
            view = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self._model, view)
            view.lookat[:] = camera.lookat
            view.distance = camera.distance
            view.azimuth = camera.azimuth
            view.elevation = camera.elevation
            if camera.track_body is not None:
                view.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
                view.trackbodyid = _required_body_id(
                    self._model,
                    camera.track_body,
                    entity_name=self.bundle.entity_cfg.name,
                )
                view.fixedcamid = -1
            else:
                view.type = mujoco.mjtCamera.mjCAMERA_FREE.value
                view.trackbodyid = -1
                view.fixedcamid = -1
            self._camera = view

            if camera.fovy is not None:
                self._original_fovy = float(self._model.vis.global_.fovy)
                self._model.vis.global_.fovy = camera.fovy

    def render(self, env: Any) -> np.ndarray:
        """Render the environment's current state as RGB uint8 pixels."""

        if getattr(env, "num_envs", 1) != 1:
            raise ValueError("NativeRenderer renders one selected environment at a time")
        state = env.data
        if state is None:
            raise RuntimeError("Environment must be reset before rendering")

        self._data.qpos[:] = state.qpos.detach().cpu().numpy()
        self._data.qvel[:] = state.qvel.detach().cpu().numpy()
        self._data.ctrl[:] = state.ctrl.detach().cpu().numpy()
        self._data.time = float(state.time)

        # These fields are optional in the current mujoco-torch data contract,
        # but copying them when present keeps the bridge useful for future
        # props, mocap bodies, and interactive perturbations.
        if self._model.nmocap > 0:
            mocap_pos = getattr(state, "mocap_pos", None)
            mocap_quat = getattr(state, "mocap_quat", None)
            if mocap_pos is not None:
                self._data.mocap_pos[:] = mocap_pos.detach().cpu().numpy()
            if mocap_quat is not None:
                self._data.mocap_quat[:] = mocap_quat.detach().cpu().numpy()
        xfrc_applied = getattr(state, "xfrc_applied", None)
        if xfrc_applied is not None and hasattr(self._data, "xfrc_applied"):
            self._data.xfrc_applied[:] = xfrc_applied.detach().cpu().numpy()

        mujoco.mj_forward(self._model, self._data)
        if (
            isinstance(self._camera, mujoco.MjvCamera)
            and self.camera_config.follow_root
            and self.camera_config.track_body is None
        ):
            root_name = self.bundle.entity_cfg.name
            root_view = self.bundle.entities.get(root_name)
            if root_view is not None and root_view.free_qpos_indices.numel() >= 2:
                indices = root_view.free_qpos_indices[:2].detach().cpu().numpy()
                self._camera.lookat[:2] = self._data.qpos[indices]
        self._renderer.update_scene(self._data, camera=self._camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8)

    def close(self) -> None:
        self._renderer.close()
        if self._original_fovy is not None:
            self._model.vis.global_.fovy = self._original_fovy
            self._original_fovy = None

    def __enter__(self) -> NativeRenderer:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self.close()
