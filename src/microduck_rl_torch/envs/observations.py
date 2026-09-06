"""Reusable observation terms and the current policy command helper."""

from __future__ import annotations

from typing import Any

import torch


def _sensor_state(env: Any) -> Any:
    if env.state is None:
        raise RuntimeError("Call reset() before reading observations")
    return env.state.sensors


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in ``[w, x, y, z]`` order."""

    xyz = quaternion[..., 1:]
    t = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * t + torch.cross(xyz, t, dim=-1)


def base_ang_vel(env: Any, *, misaligned: bool = True) -> torch.Tensor:
    """Return the delayed IMU angular-velocity observation term."""

    sensor = _sensor_state(env)
    lag = sensor.imu_lag
    if isinstance(lag, torch.Tensor) and lag.ndim > 0:
        values = torch.stack(sensor.imu_ang_vel_history, dim=1)
        indices = (values.shape[1] - 1 - lag).clamp_min(0)
        value = values[torch.arange(env.num_envs, device=values.device), indices]
    else:
        index = max(0, len(sensor.imu_ang_vel_history) - 1 - int(lag))
        value = sensor.imu_ang_vel_history[index]
    return _quat_apply(sensor.imu_quaternion, value) if misaligned else value


def projected_gravity(env: Any, *, misaligned: bool = True) -> torch.Tensor:
    """Return the delayed projected-gravity observation term."""

    sensor = _sensor_state(env)
    lag = sensor.imu_lag
    if isinstance(lag, torch.Tensor) and lag.ndim > 0:
        values = torch.stack(sensor.projected_gravity_history, dim=1)
        indices = (values.shape[1] - 1 - lag).clamp_min(0)
        value = values[torch.arange(env.num_envs, device=values.device), indices]
    else:
        index = max(0, len(sensor.projected_gravity_history) - 1 - int(lag))
        value = sensor.projected_gravity_history[index]
    return _quat_apply(sensor.imu_quaternion, value) if misaligned else value


def joint_position(env: Any, *, biased: bool = True) -> torch.Tensor:
    """Return output-side joint position relative to the model home pose."""

    sensor = _sensor_state(env)
    position = env._joint_measurements()[0]
    if biased:
        position = position + sensor.encoder_bias
    return position - env.bundle.default_pose


def joint_velocity(env: Any, *, delayed: bool = True) -> torch.Tensor:
    """Return the delayed output-side joint velocity term."""

    sensor = _sensor_state(env)
    return sensor.previous_joint_velocity if delayed else env._encoder_velocity()


def joint_position_rel_backlash(env: Any, *, biased: bool = True) -> torch.Tensor:
    """Read the output-side encoder position for backlash entities.

    The model bundle's actuator map already pairs each servo with its
    passive backlash hinge.  Keeping this as a distinct term function mirrors
    upstream task mutation and makes the semantic choice visible in a cloned
    task configuration.
    """

    return joint_position(env, biased=biased)


def joint_velocity_rel_backlash(env: Any, *, delayed: bool = True) -> torch.Tensor:
    """Read the output-side encoder velocity for backlash entities."""

    return joint_velocity(env, delayed=delayed)


def base_lin_vel(env: Any) -> torch.Tensor:
    """Return privileged trunk linear velocity in the trunk frame."""

    if env.data is None:
        raise RuntimeError("Call reset() before reading observations")
    return env.data.cvel[..., env.bundle.root_body_id, 3:6]


def last_action(env: Any) -> torch.Tensor:
    """Return the previous policy action term."""

    return _sensor_state(env).last_action


def command(env: Any) -> torch.Tensor:
    """Return the concatenated command-manager output."""

    return env.command


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
