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
    """Return the current IMU angular-velocity observation term.

    Temporal delay is configured on this observation term, where it belongs in
    the manager pipeline, rather than in task-specific sensor state.
    """

    sensor = _sensor_state(env)
    value = sensor.imu_ang_vel_history[-1]
    return _quat_apply(sensor.imu_quaternion, value) if misaligned else value


def projected_gravity(env: Any, *, misaligned: bool = True) -> torch.Tensor:
    """Return the current projected-gravity observation term."""

    sensor = _sensor_state(env)
    value = sensor.projected_gravity_history[-1]
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
    task mutation and makes the semantic choice visible in a cloned task
    configuration.
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
    """Return the action manager's current action-history value."""

    return env.action_manager.last_action


def foot_height(env: Any, *, sensor_name: str = "foot_height_scan") -> torch.Tensor:
    """Return terrain-relative height for each foot from the named ray sensor."""

    return env.sensors.read(sensor_name)


def foot_contact(env: Any, *, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    """Return per-foot contact flags."""

    contact = env.sensors.data(sensor_name)
    found = getattr(contact, "found", None)
    if found is None:
        raise RuntimeError(f"Contact sensor {sensor_name!r} does not expose found")
    value = (found > 0).to(dtype=env.bundle.dtype)
    return value.reshape(-1) if getattr(env, "num_envs", 1) == 1 else value


def foot_air_time(env: Any, *, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    """Return the contact sensor's per-foot accumulated air time."""

    value = env.sensors.air_time(sensor_name)
    return value.reshape(-1) if getattr(env, "num_envs", 1) == 1 else value


def foot_contact_forces(env: Any, *, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    """Return signed log-scaled per-foot contact forces for the critic."""

    contact = env.sensors.data(sensor_name)
    force = getattr(contact, "force", None)
    if force is None:
        raise RuntimeError(f"Contact sensor {sensor_name!r} does not expose force")
    force = torch.as_tensor(force, dtype=env.bundle.dtype, device=env.bundle.device)
    value = torch.sign(force) * torch.log1p(torch.abs(force))
    if getattr(env, "num_envs", 1) == 1:
        return value.reshape(-1)
    return value.flatten(start_dim=1)


def command(env: Any) -> torch.Tensor:
    """Return the concatenated command-manager output."""

    return env.command


def command_term(env: Any, *, name: str) -> torch.Tensor:
    """Return one named command term without depending on layout offsets."""

    return env.command_manager.get_command(name)


def command_component(env: Any, *, start: int, size: int) -> torch.Tensor:
    """Return a raw command slice for low-level compatibility use."""

    return env.command[..., start : start + size]


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
