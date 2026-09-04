"""Reference native MuJoCo rollout used by the environment validator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from microduck_rl_torch.envs.model import SERVO_JOINT_NAMES, default_scene_path

mujoco_api: Any = mujoco


def _quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[1:]
    t = 2.0 * np.cross(xyz, vector)
    return vector - quaternion[0] * t + np.cross(xyz, t)


class NativeMicroDuckEnv:
    """Native MuJoCo counterpart with identical action and observation semantics."""

    def __init__(
        self,
        xml_path: Path | None = None,
        *,
        timestep: float = 0.005,
        decimation: int = 4,
        solver_iterations: int | None = None,
        line_search_iterations: int | None = None,
        disable_contacts: bool = False,
    ):
        self.xml_path = (xml_path or default_scene_path()).resolve()
        self.model = mujoco_api.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = timestep
        if solver_iterations is not None:
            self.model.opt.iterations = solver_iterations
        if line_search_iterations is not None:
            self.model.opt.ls_iterations = line_search_iterations
        if disable_contacts:
            self.model.opt.disableflags |= int(mujoco_api.mjtDisableBit.mjDSBL_CONTACT)
        self.data = mujoco_api.MjData(self.model)
        self.decimation = decimation
        qpos_indices = []
        qvel_indices = []
        names = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            qvel_indices.append(int(self.model.jnt_dofadr[joint_id]))
            names.append(mujoco_api.mj_id2name(self.model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_id))
        if tuple(names) != SERVO_JOINT_NAMES:
            raise ValueError(f"Unexpected native actuator order: {tuple(names)!r}")
        self.qpos_indices = np.asarray(qpos_indices, dtype=np.int64)
        self.qvel_indices = np.asarray(qvel_indices, dtype=np.int64)
        stand_id = mujoco_api.mj_name2id(self.model, mujoco_api.mjtObj.mjOBJ_KEY, "STAND")
        self.default_qpos = np.asarray(self.model.key_qpos[stand_id], dtype=np.float64).copy()
        self.default_pose = self.default_qpos[self.qpos_indices].copy()
        self.imu_ang_vel = self._sensor_slice("imu_ang_vel")
        self.trunk_body_id = mujoco_api.mj_name2id(
            self.model, mujoco_api.mjtObj.mjOBJ_BODY, "trunk_base"
        )
        self.command = np.zeros(13, dtype=np.float32)
        self.last_action = np.zeros(14, dtype=np.float64)
        self.reset()

    def _sensor_slice(self, name: str) -> slice:
        sensor_id = mujoco_api.mj_name2id(self.model, mujoco_api.mjtObj.mjOBJ_SENSOR, name)
        start = int(self.model.sensor_adr[sensor_id])
        return slice(start, start + int(self.model.sensor_dim[sensor_id]))

    def reset(self) -> np.ndarray:
        self.data.qpos[:] = self.default_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_pose
        mujoco_api.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0
        return self.observation()

    def observation(self) -> np.ndarray:
        gravity = _quat_rotate_inverse(
            self.data.xquat[self.trunk_body_id], np.array([0.0, 0.0, -1.0])
        )
        observation = np.concatenate(
            [
                self.data.sensordata[self.imu_ang_vel],
                gravity,
                self.data.qpos[self.qpos_indices] - self.default_pose,
                self.data.qvel[self.qvel_indices],
                self.last_action,
                self.command,
            ]
        )
        if observation.shape != (61,):
            raise RuntimeError(f"Built native observation with shape {observation.shape}")
        return observation.astype(np.float32)

    def step(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (14,):
            raise ValueError(f"Expected action shape (14,), got {action.shape}")
        self.data.ctrl[:] = self.default_pose + action
        for _ in range(self.decimation):
            mujoco_api.mj_step(self.model, self.data)
        self.last_action[:] = action
        return self.observation()

    def snapshot(self) -> dict[str, np.ndarray | float]:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "sensordata": self.data.sensordata.copy(),
            "time": float(self.data.time),
        }
