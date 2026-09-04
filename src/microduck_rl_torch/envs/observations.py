"""The 61-element `new_cmd_obs` actor observation contract."""

from __future__ import annotations

from typing import Any

import torch

from .model import MicroDuckModelBundle


def command_vector(
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    vtheta: float = 0.0,
    neck_pitch: float = 0.0,
    head_pitch: float = 0.0,
    head_yaw: float = 0.0,
    head_roll: float = 0.0,
    body_x: float = 0.0,
    body_y: float = 0.0,
    body_z: float = 0.0,
    body_roll: float = 0.0,
    body_pitch: float = 0.0,
    body_yaw: float = 0.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(
        [
            vx,
            vy,
            vtheta,
            neck_pitch,
            head_pitch,
            head_yaw,
            head_roll,
            body_x,
            body_y,
            body_z,
            body_roll,
            body_pitch,
            body_yaw,
        ],
        dtype=dtype,
        device=device,
    )


def _quat_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[..., 1:]
    t = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector - quaternion[..., :1] * t + torch.cross(xyz, t, dim=-1)


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in ``[w, x, y, z]`` order."""

    xyz = quaternion[..., 1:]
    t = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * t + torch.cross(xyz, t, dim=-1)


def _expand_last(value: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
    if value.ndim == 1:
        return value.expand(*batch_shape, value.shape[-1])
    if value.shape[:-1] != batch_shape:
        raise ValueError(
            f"Expected batch shape {tuple(batch_shape)}, got {tuple(value.shape[:-1])}"
        )
    return value


def build_actor_observation(
    bundle: MicroDuckModelBundle,
    data: Any,
    last_action: torch.Tensor,
    command: torch.Tensor,
    *,
    joint_position: torch.Tensor | None = None,
    joint_position_bias: torch.Tensor | None = None,
    joint_velocity: torch.Tensor | None = None,
    base_ang_vel: torch.Tensor | None = None,
    projected_gravity: torch.Tensor | None = None,
    imu_quaternion: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the upstream 61D actor observation.

    The optional values are the runtime sensor path: delayed joint velocity,
    encoder bias, IMU misalignment, and observation noise.  Leaving them unset
    preserves the deterministic nominal observation used by structural tests.
    """

    qpos_indices = bundle.qpos_indices.to(data.qpos.device)
    qvel_indices = bundle.qvel_indices.to(data.qvel.device)
    if base_ang_vel is None:
        base_ang_vel = data.sensordata[..., bundle.sensor_slices["imu_ang_vel"]]
    quaternion = data.xquat[..., bundle.trunk_body_id, :]
    gravity_world = torch.zeros(
        (*quaternion.shape[:-1], 3), dtype=data.qpos.dtype, device=data.qpos.device
    )
    gravity_world[..., 2] = -1.0
    if projected_gravity is None:
        projected_gravity = _quat_rotate_inverse(quaternion, gravity_world)
    if joint_position_bias is None:
        joint_position_bias = torch.zeros_like(bundle.default_pose)
    if joint_position is None:
        joint_position = data.qpos.index_select(-1, qpos_indices)
    joint_position = joint_position + joint_position_bias - bundle.default_pose
    if joint_velocity is None:
        joint_velocity = data.qvel.index_select(-1, qvel_indices)
    if imu_quaternion is not None:
        base_ang_vel = _quat_apply(imu_quaternion, base_ang_vel)
        projected_gravity = _quat_apply(imu_quaternion, projected_gravity)
    batch_shape = data.qpos.shape[:-1]
    action = _expand_last(torch.as_tensor(last_action, device=data.qpos.device), batch_shape)
    command = _expand_last(torch.as_tensor(command, device=data.qpos.device), batch_shape)
    observation = torch.cat(
        [base_ang_vel, projected_gravity, joint_position, joint_velocity, action, command], dim=-1
    )
    if observation.shape[-1] != bundle.observation_size:
        raise RuntimeError(f"Built observation of size {observation.shape[-1]}, expected 61")
    if noise is not None:
        if noise.shape != observation.shape:
            raise ValueError(
                "Expected observation noise shape "
                f"{tuple(observation.shape)}, got {tuple(noise.shape)}"
            )
        observation = observation + noise
    return observation.to(dtype=torch.float32)
