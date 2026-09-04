"""Independent native MuJoCo reference for lockstep trajectory tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from microduck_rl_torch.envs.actuation import BamM6Parameters, friction_budget, motor_torque
from microduck_rl_torch.envs.model import (
    SERVO_JOINT_NAMES,
    MicroDuckModelBundle,
    default_scene_path,
)

mujoco_api: Any = mujoco


def _quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[1:]
    t = 2.0 * np.cross(xyz, vector)
    return vector - quaternion[0] * t + np.cross(xyz, t)


class NativeMicroDuckEnv:
    """Native counterpart with the same policy and BAM control semantics.

    Physics is intentionally evaluated through MuJoCo's C API here rather
    than sharing the Torch stepping function. This makes trajectory fixtures
    useful for detecting target-engine drift instead of merely testing two
    wrappers around the same implementation.
    """

    def __init__(
        self,
        xml_path: Path | None = None,
        *,
        bundle: MicroDuckModelBundle | None = None,
        timestep: float = 0.005,
        decimation: int = 4,
        solver_iterations: int | None = None,
        line_search_iterations: int | None = None,
        disable_contacts: bool = False,
        actuator_mode: str = "bam",
        parameters: BamM6Parameters | None = None,
        action_delay_lag: int = 0,
    ):
        self.bundle = bundle
        if bundle is not None:
            self.xml_path = bundle.xml_path
            self.model = bundle.native_model
            self.actuator_mode = bundle.actuator_mode
            self.parameters = bundle.bam_parameters
            self.qpos_indices = bundle.qpos_indices.detach().cpu().numpy()
            self.qvel_indices = bundle.qvel_indices.detach().cpu().numpy()
            self.backlash_qpos_indices = bundle.backlash_qpos_indices.detach().cpu().numpy()
            self.backlash_qvel_indices = bundle.backlash_qvel_indices.detach().cpu().numpy()
            self.backlash_mask = bundle.backlash_mask.detach().cpu().numpy()
            self.friction_dof_count = bundle.friction_dof_count
            self.foot_geom_ids = bundle.foot_geom_ids
            self.foot_site_ids = bundle.foot_site_ids
            self.collision_geom_ids = bundle.collision_geom_ids
            self.default_qpos = np.asarray(self.model.key("STAND").qpos, dtype=np.float64).copy()
            self.default_pose = self.default_qpos[self.qpos_indices].copy()
        else:
            self.xml_path = (xml_path or default_scene_path()).resolve()
            if actuator_mode != "xml":
                raise ValueError("Pass a model bundle for native BAM reference construction")
            self.model = mujoco_api.MjModel.from_xml_path(str(self.xml_path))
            self.actuator_mode = actuator_mode
            self.parameters = parameters
            qpos_indices = []
            qvel_indices = []
            names = []
            for actuator_id in range(self.model.nu):
                joint_id = int(self.model.actuator_trnid[actuator_id, 0])
                qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
                qvel_indices.append(int(self.model.jnt_dofadr[joint_id]))
                names.append(
                    mujoco_api.mj_id2name(self.model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_id)
                )
            if tuple(names) != SERVO_JOINT_NAMES:
                raise ValueError(f"Unexpected native actuator order: {tuple(names)!r}")
            self.qpos_indices = np.asarray(qpos_indices, dtype=np.int64)
            self.qvel_indices = np.asarray(qvel_indices, dtype=np.int64)
            self.backlash_qpos_indices = self.qpos_indices.copy()
            self.backlash_qvel_indices = self.qvel_indices.copy()
            self.backlash_mask = np.zeros(14, dtype=np.float64)
            self.friction_dof_count = 0
            self.foot_geom_ids = tuple(
                mujoco_api.mj_name2id(self.model, mujoco_api.mjtObj.mjOBJ_GEOM, name)
                for name in ("left_foot_collision", "right_foot_collision")
            )
            self.foot_site_ids = tuple(
                mujoco_api.mj_name2id(self.model, mujoco_api.mjtObj.mjOBJ_SITE, name)
                for name in ("left_foot", "right_foot")
            )
            self.collision_geom_ids = tuple(
                geom_id
                for geom_id in range(self.model.ngeom)
                if (
                    name := mujoco_api.mj_id2name(self.model, mujoco_api.mjtObj.mjOBJ_GEOM, geom_id)
                )
                and name.endswith("_collision")
            )
            stand_id = mujoco_api.mj_name2id(self.model, mujoco_api.mjtObj.mjOBJ_KEY, "STAND")
            self.default_qpos = np.asarray(self.model.key_qpos[stand_id], dtype=np.float64).copy()
            self.default_pose = self.default_qpos[self.qpos_indices].copy()

        self.model.opt.timestep = timestep
        if solver_iterations is not None:
            self.model.opt.iterations = solver_iterations
        if line_search_iterations is not None:
            self.model.opt.ls_iterations = line_search_iterations
        if disable_contacts:
            self.model.opt.disableflags |= int(mujoco_api.mjtDisableBit.mjDSBL_CONTACT)
        self.data = mujoco_api.MjData(self.model)
        self._base_dof_frictionloss = self.model.dof_frictionloss.copy()
        self._base_dof_damping = self.model.dof_damping.copy()
        self.timestep = timestep
        self.decimation = decimation
        self.action_delay_lag = action_delay_lag
        self.action_buffer = [np.zeros(14, dtype=np.float64) for _ in range(action_delay_lag + 1)]
        self.bam_previous_torque = np.zeros(14, dtype=np.float64)
        self.imu_ang_vel = self._sensor_slice("imu_ang_vel")
        self.imu_accel = self._sensor_slice("imu_accel")
        self.trunk_body_id = mujoco_api.mj_name2id(
            self.model, mujoco_api.mjtObj.mjOBJ_BODY, "trunk_base"
        )
        self.command = np.zeros(13, dtype=np.float64)
        self.last_action = np.zeros(14, dtype=np.float64)
        self.previous_joint_velocity = np.zeros(14, dtype=np.float64)
        self.previous_foot_positions = np.zeros((2, 3), dtype=np.float64)
        self.foot_air_time = np.zeros(2, dtype=np.float64)
        self.foot_contact = np.zeros(2, dtype=bool)
        self.last_reward = 0.0
        self.last_reward_terms: dict[str, float] = {}
        self._step_count = 0
        self.reset()

    def _sensor_slice(self, name: str) -> slice:
        sensor_id = mujoco_api.mj_name2id(self.model, mujoco_api.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"Required sensor {name!r} was not found")
        start = int(self.model.sensor_adr[sensor_id])
        return slice(start, start + int(self.model.sensor_dim[sensor_id]))

    def _friction_force(self) -> np.ndarray:
        if self.friction_dof_count == 0 or self.data.nefc == 0:
            return np.zeros(14, dtype=np.float64)
        force = np.zeros(14, dtype=np.float64)
        for row in range(int(self.data.nefc)):
            if int(self.data.efc_type[row]) != int(mujoco_api.mjtConstraint.mjCNSTR_FRICTION_DOF):
                continue
            # For mjCNSTR_FRICTION_DOF, MuJoCo stores the constrained DOF
            # address in efc_id (not the corresponding joint id).  The
            # actuator order is a subset of qvel, so map through qvel_indices.
            dof = int(self.data.efc_id[row])
            matches = np.flatnonzero(self.qvel_indices == dof)
            if matches.size:
                force[matches[0]] += self.data.efc_force[row]
        return force

    def _bam_control(self, target: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            raise RuntimeError("BAM parameters are not initialized")
        q = self.data.qpos[self.qpos_indices]
        if np.any(self.backlash_mask):
            q = q + self.data.qpos[self.backlash_qpos_indices] * self.backlash_mask
        dq = self.data.qvel[self.qvel_indices]
        vin = self.parameters.vin
        if self.parameters.vin_drop_gain is not None:
            vin -= self.parameters.vin_drop_gain * np.abs(self.bam_previous_torque).sum()
            if self.parameters.vin_min is not None:
                vin = max(vin, self.parameters.vin_min)
        torque = motor_torque(target, q, dq, params=self.parameters, vin=vin)
        external = (
            -self.data.qfrc_bias[self.qvel_indices]
            + self.data.qfrc_constraint[self.qvel_indices]
            - self._friction_force()
        )
        friction, damping = friction_budget(
            self.bam_previous_torque,
            external,
            dq,
            params=self.parameters,
        )
        self.model.dof_frictionloss[self.qvel_indices] = friction
        self.model.dof_damping[self.qvel_indices] = damping
        return np.asarray(torque, dtype=np.float64)

    def _contact_mask(self) -> np.ndarray:
        result = np.zeros(2, dtype=bool)
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            for foot_index, geom_id in enumerate(self.foot_geom_ids):
                result[foot_index] |= contact.geom1 == geom_id or contact.geom2 == geom_id
        return result

    def _self_collision(self) -> bool:
        collision_geoms = set(self.collision_geom_ids)
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            if contact.geom1 in collision_geoms and contact.geom2 in collision_geoms:
                return True
        return False

    def _compute_reward(
        self,
        action: np.ndarray,
        previous_action: np.ndarray,
        previous_contact: np.ndarray,
        previous_air_time: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        q = self.data.qpos[self.qpos_indices]
        q_error = q - self.default_pose
        leg_indices = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
        speed = np.linalg.norm(self.command[:2]) + abs(self.command[2])
        leg_std = np.array(
            ([0.1, 0.05, 0.15, 0.15, 0.1] * 2)
            if speed < 0.01
            else ([0.3, 0.05, 0.4, 0.4, 0.25] * 2)
        )
        pose = float(np.exp(-((q_error[leg_indices] / leg_std) ** 2)).mean())
        trunk_rotation = self.data.xmat[self.trunk_body_id].reshape(3, 3)
        gravity = trunk_rotation.T @ np.array([0.0, 0.0, -1.0])
        upright = float(np.exp(-np.square(gravity[:2]).sum() / 0.05))
        body_velocity = self.data.cvel[self.trunk_body_id, 3:6]
        track_linear_velocity = float(
            np.exp(-np.square(body_velocity[:2] - self.command[:2]).sum() / 0.1)
        )
        angular_velocity = self.data.sensordata[self.imu_ang_vel]
        track_angular_velocity = float(
            np.exp(-np.square(angular_velocity[2] - self.command[2]) / 0.5)
        )
        touchdown = self.foot_contact & ~previous_contact
        active = speed >= 0.01
        air_time = float(
            np.sum(touchdown & (previous_air_time >= 0.125) & (previous_air_time <= 0.300))
            if active
            else 0.0
        )
        foot_positions = self.data.site_xpos[list(self.foot_site_ids)]
        foot_velocity = (foot_positions - self.previous_foot_positions) / (
            self.timestep * self.decimation
        )
        foot_slip = float(np.sum(np.square(foot_velocity[:, :2]) * self.foot_contact[:, None]))
        foot_clearance = float(
            np.sum(np.square(np.maximum(0.02 - foot_positions[:, 2], 0.0)) * self.foot_contact)
        )
        foot_swing_height = float(
            np.sum(np.square(np.maximum(0.02 - foot_positions[:, 2], 0.0)) * ~self.foot_contact)
        )
        if not active:
            foot_slip = 0.0
            foot_clearance = 0.0
            foot_swing_height = 0.0
        head_position = q[5:9] + (
            self.data.qpos[self.backlash_qpos_indices[5:9]] * self.backlash_mask[5:9]
        )
        head_error = (head_position - self.default_pose[5:9]) - self.command[3:7]
        head_pose_tracking = float(np.exp(-np.square(head_error) / 0.5**2).mean())
        body_ang_vel = float(np.square(angular_velocity).sum())
        angular_momentum = float(np.square(self.data.subtree_angmom[self.trunk_body_id]).sum())
        action_rate_l2 = float(np.square(action - previous_action).sum())
        self_collisions = float(self._self_collision())
        terms = {
            "pose": pose,
            "upright": upright,
            "track_linear_velocity": track_linear_velocity,
            "track_angular_velocity": track_angular_velocity,
            "air_time": air_time,
            "head_pose_tracking": head_pose_tracking,
            "foot_slip": foot_slip,
            "body_ang_vel": body_ang_vel,
            "angular_momentum": angular_momentum,
            "action_rate_l2": action_rate_l2,
            "foot_clearance": foot_clearance,
            "foot_swing_height": foot_swing_height,
            "self_collisions": self_collisions,
        }
        reward = (
            pose
            + 2.0 * upright
            + 2.0 * track_linear_velocity
            + 2.0 * track_angular_velocity
            + 3.0 * air_time
            + 2.0 * head_pose_tracking
            - 0.1 * foot_slip
            - 0.05 * body_ang_vel
            - 0.02 * angular_momentum
            - 0.1 * action_rate_l2
            - 2.0 * foot_clearance
            - 0.25 * foot_swing_height
            - self_collisions
        )
        return reward, terms

    def reset(self) -> np.ndarray:
        self.model.dof_frictionloss[:] = self._base_dof_frictionloss
        self.model.dof_damping[:] = self._base_dof_damping
        self.data.qpos[:] = self.default_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0 if self.actuator_mode == "bam" else self.default_pose
        mujoco_api.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0
        self.previous_joint_velocity[:] = 0.0
        self.bam_previous_torque[:] = 0.0
        self.previous_foot_positions[:] = self.data.site_xpos[list(self.foot_site_ids)]
        self.foot_air_time[:] = 0.0
        self.foot_contact[:] = self._contact_mask()
        self.last_reward = 0.0
        self.last_reward_terms = {}
        self._step_count = 0
        for entry in self.action_buffer:
            entry[:] = 0.0
        return self.observation()

    def observation(self) -> np.ndarray:
        gravity = _quat_rotate_inverse(
            self.data.xquat[self.trunk_body_id], np.array([0.0, 0.0, -1.0])
        )
        joint_position = self.data.qpos[self.qpos_indices]
        if np.any(self.backlash_mask):
            joint_position = joint_position + (
                self.data.qpos[self.backlash_qpos_indices] * self.backlash_mask
            )
        observation = np.concatenate(
            [
                self.data.sensordata[self.imu_ang_vel],
                gravity,
                joint_position - self.default_pose,
                self.previous_joint_velocity,
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
        self.previous_joint_velocity[:] = self.data.qvel[self.qvel_indices]
        if np.any(self.backlash_mask):
            self.previous_joint_velocity[:] += (
                self.data.qvel[self.backlash_qvel_indices] * self.backlash_mask
            )
        previous_contact = self.foot_contact.copy()
        previous_air_time = self.foot_air_time.copy()
        previous_action = self.last_action.copy()
        index = self._step_count % len(self.action_buffer) if hasattr(self, "_step_count") else 0
        self.action_buffer[index] = action.copy()
        delayed = self.action_buffer[(index - self.action_delay_lag) % len(self.action_buffer)]
        target = self.default_pose + delayed
        for _ in range(self.decimation):
            if self.actuator_mode == "bam":
                self.data.ctrl[:] = self._bam_control(target)
            else:
                self.data.ctrl[:] = target
            mujoco_api.mj_step(self.model, self.data)
            if self.actuator_mode == "bam":
                self.bam_previous_torque[:] = self.data.qfrc_actuator[self.qvel_indices]
        self.last_action[:] = action
        self._step_count = getattr(self, "_step_count", 0) + 1
        self.foot_contact = self._contact_mask()
        self.foot_air_time = np.where(
            self.foot_contact,
            0.0,
            previous_air_time + self.timestep * self.decimation,
        )
        self.last_reward, self.last_reward_terms = self._compute_reward(
            action, previous_action, previous_contact, previous_air_time
        )
        self.previous_foot_positions[:] = self.data.site_xpos[list(self.foot_site_ids)]
        return self.observation()

    def snapshot(self) -> dict[str, np.ndarray | float]:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "qacc": self.data.qacc.copy(),
            "ctrl": self.data.ctrl.copy(),
            "sensordata": self.data.sensordata.copy(),
            "time": float(self.data.time),
            "foot_contact": self.foot_contact.copy(),
        }
