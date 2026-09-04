"""Generic temporal sensor state owned outside the environment lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..config import sample_uniform
from ..sensors import contact_mask


@dataclass(frozen=True)
class SensorStateCfg:
    """Opt-in temporal state declaration for a task's sensor contract.

    The manager owns generic action/joint/IMU temporal state.  Contact timing
    is an optional named channel layered on top, so tasks without feet do not
    inherit Microduck velocity assumptions merely by using sensor history.
    Calibration settings live here rather than in ``env.config`` so this
    manager is independent of any task-family dataclass.
    """

    enabled: bool = False
    imu_sensor_name: str = "imu_ang_vel"
    foot_site_handle: str = "foot_site_ids"
    foot_geom_handle: str = "foot_geom_groups"
    foot_contact_sensor_names: tuple[str, ...] = (
        "left_foot_contact",
        "right_foot_contact",
    )
    air_time_sensor_name: str | None = "feet_ground_contact"
    history_length: int = 4
    randomize_encoder_bias: bool = False
    encoder_bias_range: tuple[float, float] = (0.0, 0.0)
    randomize_imu_orientation: bool = False
    imu_angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        if self.history_length < 0:
            raise ValueError("Sensor history_length must be non-negative")
        if self.encoder_bias_range[0] > self.encoder_bias_range[1]:
            raise ValueError("encoder_bias_range must be ascending")
        if self.imu_angle_degrees < 0.0:
            raise ValueError("imu_angle_degrees must be non-negative")


@dataclass
class SensorState:
    """Temporal state for configured sensors, not for the environment core."""

    last_action: torch.Tensor
    previous_action: torch.Tensor
    previous_joint_velocity: torch.Tensor
    previous_foot_positions: torch.Tensor | None
    foot_air_time: torch.Tensor | None
    foot_contact: torch.Tensor | None
    foot_peak_heights: torch.Tensor | None
    imu_ang_vel_history: list[torch.Tensor]
    projected_gravity_history: list[torch.Tensor]
    encoder_bias: torch.Tensor
    imu_quaternion: torch.Tensor


@dataclass(frozen=True)
class TransitionData:
    """Generic transition context with optional contact timing channels."""

    action: torch.Tensor
    previous_action: torch.Tensor
    previous_foot_positions: torch.Tensor | None = None
    foot_air_time: torch.Tensor | None = None
    foot_contact: torch.Tensor | None = None
    foot_touchdown: torch.Tensor | None = None
    foot_peak_heights: torch.Tensor | None = None


@dataclass(frozen=True)
class SensorStepContext:
    previous_foot_contact: torch.Tensor | None
    previous_foot_air_time: torch.Tensor | None
    previous_action: torch.Tensor
    previous_foot_positions: torch.Tensor | None


class SensorStateManager:
    """Own temporal sensor/calibration state and transition baselines.

    The manager is deliberately opt-in for task-specific IMU/foot state.  A
    task with no such sensors still receives generic action/joint baselines,
    while the lifecycle owner never branches on velocity concepts.
    """

    def __init__(self, config: SensorStateCfg | None = None) -> None:
        self.config = config or SensorStateCfg()
        self.state: SensorState | None = None
        self.transition: TransitionData | None = None

    @property
    def active(self) -> bool:
        return self.config.enabled

    @staticmethod
    def _quat_from_euler(
        roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
    ) -> torch.Tensor:
        cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
        cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
        cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
        return torch.stack(
            (
                cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                cy * sp * cr + sy * cp * sr,
                sy * cp * cr - cy * sp * sr,
            ),
            dim=-1,
        )

    @staticmethod
    def _ids(env: Any, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(env.num_envs, device=env.bundle.device)
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.bundle.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=env.bundle.device).reshape(-1)

    def _imu_and_gravity(self, env: Any) -> tuple[torch.Tensor | None, torch.Tensor]:
        if env.data is None:
            raise RuntimeError("Physics must be reset before sensor state initialization")
        sensor_name = self.config.imu_sensor_name
        sensor_slice = env.bundle.sensor_slices.get(sensor_name)
        if sensor_slice is None and "/" not in sensor_name:
            sensor_name = f"{env.bundle.primary_entity_name}/{sensor_name}"
            sensor_slice = env.bundle.sensor_slices.get(sensor_name)
        angular_velocity = (
            env.data.sensordata[..., sensor_slice].clone() if sensor_slice is not None else None
        )
        gravity_world = torch.zeros(3, dtype=env.bundle.dtype, device=env.bundle.device)
        gravity_world[2] = -1.0
        root = env.bundle.entity(env.bundle.primary_entity_name).root_body_id
        gravity = env.data.xmat[..., root, :, :].transpose(-1, -2) @ gravity_world
        return angular_velocity, gravity

    def _foot_positions(self, env: Any) -> torch.Tensor | None:
        if not self.config.enabled:
            return None
        ids = env.bundle.handle(self.config.foot_site_handle)
        if not ids or env.data is None:
            return None
        return env.data.site_xpos[..., list(ids), :].clone()

    def _foot_contact(self, env: Any) -> torch.Tensor | None:
        if not self.config.enabled:
            return None
        names = self.config.foot_contact_sensor_names
        if all(name in env.sensor_manager.active_sensors for name in names):
            values = [env.sensor_manager.read(name).reshape(-1) > 0 for name in names]
            if env.num_envs > 1:
                return torch.stack(values, dim=-1)
            return torch.stack([value.squeeze() for value in values])
        if env.bundle.handle(self.config.foot_geom_handle):
            return contact_mask(env.data, env.bundle.handle(self.config.foot_geom_handle))
        return None

    def _foot_air_time_from_sensor(self, env: Any) -> torch.Tensor | None:
        """Read the authoritative contact-sensor air-time channel."""

        sensor_name = self.config.air_time_sensor_name
        if sensor_name is None:
            return None
        if sensor_name not in env.sensor_manager.active_sensors:
            return None
        contact = env.sensor_manager.data(sensor_name)
        value = getattr(contact, "current_air_time", None)
        if value is None:
            return None
        return value.clone()

    def _sample_imu_calibration(self, env: Any, state: SensorState, env_ids: Any) -> None:
        ids = self._ids(env, env_ids)
        full_batch = env.num_envs == 1 or env_ids is None
        target_ids = None if full_batch else ids
        batch_size = env.num_envs if full_batch else int(ids.numel())
        generators = getattr(env.physics, "generators", None)
        if generators is not None and not full_batch:
            generators = tuple(generators[int(index)] for index in ids.tolist())
        if generators is None:
            generators = env._generator
        if env.domain_randomization and self.config.randomize_encoder_bias:
            bias = sample_uniform(
                (self.config.encoder_bias_range,) * env.bundle.action_size,
                generator=generators,
                device=env.bundle.device,
                dtype=env.bundle.dtype,
                batch_size=batch_size,
            )
            if target_ids is None:
                state.encoder_bias = bias
            else:
                state.encoder_bias[target_ids] = bias
        elif target_ids is None:
            state.encoder_bias.zero_()
        else:
            state.encoder_bias[target_ids] = 0

        identity = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=env.bundle.dtype, device=env.bundle.device
        )
        if target_ids is None:
            state.imu_quaternion = identity.clone()
        else:
            state.imu_quaternion[target_ids] = identity
        angle_limit = float(self.config.imu_angle_degrees)
        if not (
            env.domain_randomization and self.config.randomize_imu_orientation and angle_limit > 0
        ):
            return
        scalar_batch = env.num_envs == 1 and full_batch
        axis_shape = (3,) if scalar_batch else (batch_size, 3)
        if batch_size == env.num_envs and full_batch:
            axis = env._random_tensor(axis_shape, dtype=env.bundle.dtype, normal=True)
        else:
            axis = torch.stack(
                [
                    torch.randn(
                        (3,),
                        generator=stream,
                        dtype=env.bundle.dtype,
                        device=env.bundle.device,
                    )
                    for stream in generators
                ]
            )
        axis = axis / (torch.linalg.vector_norm(axis, dim=-1, keepdim=True) + 1.0e-8)
        if scalar_batch:
            angle = torch.rand(
                (),
                generator=generators[0] if isinstance(generators, tuple) else generators,
                dtype=env.bundle.dtype,
                device=env.bundle.device,
            )
        elif full_batch:
            angle = env._random_tensor((batch_size,), dtype=env.bundle.dtype)
        else:
            angle = torch.stack(
                [
                    torch.rand(
                        (), generator=stream, dtype=env.bundle.dtype, device=env.bundle.device
                    )
                    for stream in generators
                ]
            )
        angle = torch.deg2rad(angle * angle_limit)
        half = angle / 2.0
        quaternion = torch.cat(
            (torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)), dim=-1
        )
        if target_ids is None:
            state.imu_quaternion = quaternion
        else:
            state.imu_quaternion[target_ids] = quaternion

    def initialize(self, env: Any) -> SensorState:
        """Allocate generic state after the backend has received reset data."""

        angular_velocity, gravity = self._imu_and_gravity(env)
        zero_action = torch.zeros(
            env.task_cfg.action_size, dtype=env.bundle.dtype, device=env.bundle.device
        )
        if env.num_envs > 1:
            zero_action = zero_action.unsqueeze(0).expand(env.num_envs, -1).clone()
        shape = (
            (env.num_envs, env.bundle.action_size)
            if env.num_envs > 1
            else (env.bundle.action_size,)
        )
        encoder_bias = torch.zeros(shape, dtype=env.bundle.dtype, device=env.bundle.device)
        foot_positions = self._foot_positions(env)
        foot_groups = env.bundle.handle(self.config.foot_geom_handle) if self.config.enabled else ()
        foot_air_time = None
        foot_peak_heights = None
        if foot_groups:
            foot_air_time = torch.zeros(
                (env.num_envs, len(foot_groups)) if env.num_envs > 1 else (len(foot_groups),),
                dtype=env.bundle.dtype,
                device=env.bundle.device,
            )
            foot_peak_heights = torch.zeros_like(foot_air_time)
        self.state = SensorState(
            last_action=zero_action.clone(),
            previous_action=zero_action.clone(),
            previous_joint_velocity=env.physics.encoder_velocity().clone(),
            previous_foot_positions=foot_positions,
            foot_air_time=foot_air_time,
            foot_contact=self._foot_contact(env),
            foot_peak_heights=foot_peak_heights,
            imu_ang_vel_history=[]
            if angular_velocity is None or not self.config.enabled
            else [angular_velocity],
            projected_gravity_history=[]
            if angular_velocity is None or not self.config.enabled
            else [gravity],
            encoder_bias=encoder_bias,
            imu_quaternion=(
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=env.bundle.dtype, device=env.bundle.device)
                if env.num_envs == 1
                else torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], dtype=env.bundle.dtype, device=env.bundle.device
                )
                .unsqueeze(0)
                .expand(env.num_envs, -1)
                .clone()
            ),
        )
        self._sample_imu_calibration(env, self.state, None)
        self.transition = None
        return self.state

    def refresh_baseline(
        self,
        env: Any,
        env_ids: torch.Tensor | slice | None = None,
        *,
        sample_calibration: bool = True,
    ) -> None:
        """Refresh histories after all reset-time qpos/qvel mutations."""

        if self.state is None:
            raise RuntimeError("Sensor state has not been initialized")
        state = self.state
        ids = self._ids(env, env_ids)
        angular_velocity, gravity = self._imu_and_gravity(env)
        if env.num_envs == 1 or env_ids is None:
            state.previous_joint_velocity = env.physics.encoder_velocity().clone()
            state.previous_foot_positions = self._foot_positions(env)
            state.foot_contact = self._foot_contact(env)
            if state.foot_air_time is not None:
                state.foot_air_time.zero_()
            if state.foot_peak_heights is not None:
                state.foot_peak_heights.zero_()
            state.imu_ang_vel_history = (
                [] if angular_velocity is None or not self.config.enabled else [angular_velocity]
            )
            state.projected_gravity_history = (
                [] if angular_velocity is None or not self.config.enabled else [gravity]
            )
        else:
            state.previous_joint_velocity[ids] = env.physics.encoder_velocity()[ids]
            foot_positions = self._foot_positions(env)
            if state.previous_foot_positions is not None and foot_positions is not None:
                state.previous_foot_positions[ids] = foot_positions[ids]
            current_contact = self._foot_contact(env)
            if current_contact is not None and state.foot_contact is not None:
                state.foot_contact[ids] = current_contact[ids]
            if state.foot_air_time is not None:
                state.foot_air_time[ids] = 0.0
            if state.foot_peak_heights is not None:
                state.foot_peak_heights[ids] = 0.0
            if angular_velocity is not None and state.imu_ang_vel_history:
                for value in state.imu_ang_vel_history:
                    value[ids] = angular_velocity[ids]
                for value in state.projected_gravity_history:
                    value[ids] = gravity[ids]
        if env.num_envs == 1 or env_ids is None:
            if sample_calibration:
                self._sample_imu_calibration(env, state, None)
        else:
            if sample_calibration:
                self._sample_imu_calibration(env, state, ids)
        self.transition = None

    def begin_step(self, env: Any) -> SensorStepContext:
        if self.state is None:
            raise RuntimeError("Call reset() before stepping")
        state = self.state
        previous = SensorStepContext(
            previous_foot_contact=None
            if state.foot_contact is None
            else state.foot_contact.clone(),
            previous_foot_air_time=None
            if state.foot_air_time is None
            else state.foot_air_time.clone(),
            previous_action=env.action_manager.last_action.clone(),
            previous_foot_positions=None
            if state.previous_foot_positions is None
            else state.previous_foot_positions.clone(),
        )
        state.previous_joint_velocity = env.physics.encoder_velocity().clone()
        state.previous_action = previous.previous_action
        return previous

    def end_step(
        self, env: Any, action: torch.Tensor, context: SensorStepContext
    ) -> TransitionData:
        if self.state is None:
            raise RuntimeError("Call reset() before stepping")
        state = self.state
        current_contact = self._foot_contact(env)
        sensor_air_time = self._foot_air_time_from_sensor(env)
        air_time_sensor_name = self.config.air_time_sensor_name
        touchdown = (
            env.sensor_manager.compute_first_contact(
                air_time_sensor_name, env.bundle.timestep * env.decimation
            )
            if sensor_air_time is not None and air_time_sensor_name is not None
            else (
                current_contact & ~context.previous_foot_contact
                if current_contact is not None and context.previous_foot_contact is not None
                else None
            )
        )
        if state.foot_air_time is not None and sensor_air_time is not None:
            state.foot_air_time = sensor_air_time
        peak_heights = None
        if state.foot_peak_heights is not None and current_contact is not None:
            foot_positions = self._foot_positions(env)
            if foot_positions is not None:
                # Match mjlab's swing-height state machine: accumulate the
                # highest terrain clearance while airborne and snapshot that
                # value for the touchdown transition before clearing it.
                if "foot_height_scan" in env.sensors.active_sensors:
                    measured_height = env.sensors.read("foot_height_scan")
                else:
                    measured_height = foot_positions[..., 2]
                if touchdown is None:
                    touchdown = torch.zeros_like(current_contact)
                state.foot_peak_heights = torch.where(
                    ~current_contact,
                    torch.maximum(state.foot_peak_heights, measured_height),
                    state.foot_peak_heights,
                )
                peak_heights = state.foot_peak_heights.clone()
                state.foot_peak_heights = torch.where(
                    touchdown,
                    torch.zeros_like(state.foot_peak_heights),
                    state.foot_peak_heights,
                )
        state.foot_contact = current_contact
        angular_velocity, gravity = self._imu_and_gravity(env)
        if angular_velocity is not None and self.config.enabled:
            state.imu_ang_vel_history.append(angular_velocity)
            state.projected_gravity_history.append(gravity)
            limit = max(1, self.config.history_length)
            del state.imu_ang_vel_history[:-limit]
            del state.projected_gravity_history[:-limit]
        if state.previous_foot_positions is not None:
            state.previous_foot_positions = self._foot_positions(env)
        state.last_action = action.clone()
        self.transition = TransitionData(
            action=action,
            previous_action=context.previous_action,
            previous_foot_positions=context.previous_foot_positions,
            # Contact duration is accumulated through this transition so the
            # reward and observation see the current value, not the preceding
            # transition.
            foot_air_time=None if state.foot_air_time is None else state.foot_air_time.clone(),
            foot_contact=current_contact,
            foot_touchdown=touchdown,
            foot_peak_heights=peak_heights,
        )
        return self.transition


__all__ = [
    "SensorState",
    "SensorStateCfg",
    "SensorStateManager",
    "SensorStepContext",
    "TransitionData",
]
