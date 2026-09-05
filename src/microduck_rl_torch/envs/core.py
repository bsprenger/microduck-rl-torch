"""Manager-based task runtime and task-facing environment facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .config import MicroDuckVelocityConfig, sample_command, sample_twist, sample_uniform
from .managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    EventManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from .model import ModelBundle, load_microduck_model
from .observations import build_actor_observation
from .physics import PhysicsBackend
from .rewards import compute_reward, foot_contact_mask
from .scene import SceneBuild, SceneBuilder
from .task_config import TaskEnvCfg


@dataclass(frozen=True)
class EnvStep:
    observation: torch.Tensor
    reward: torch.Tensor
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class MicroDuckRuntimeState:
    """Explicit state for delayed sensors, action history, and episode terms."""

    last_action: torch.Tensor
    previous_action: torch.Tensor
    previous_joint_velocity: torch.Tensor
    previous_foot_positions: torch.Tensor
    foot_air_time: torch.Tensor
    foot_contact: torch.Tensor
    imu_ang_vel_history: list[torch.Tensor]
    projected_gravity_history: list[torch.Tensor]
    delay_buffer: list[torch.Tensor]
    delay_lag: int
    imu_lag: int
    encoder_bias: torch.Tensor
    imu_quaternion: torch.Tensor
    next_push_step: int
    next_twist_step: int
    next_head_step: int
    next_body_step: int
    reward_terms: dict[str, torch.Tensor]
    # Optional task-owned state for future phases, props, and state machines.
    # The velocity task leaves this empty; future task factories can allocate
    # ball, sit/stand, roller, or roulade state without changing the generic
    # physics state schema.
    task_data: dict[str, Any] = field(default_factory=dict)


def _quat_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Create a normalized ZYX quaternion."""

    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack(
        (
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        )
    )


class VelocityTaskRuntime:
    """Velocity task lifecycle composed around a generic physics backend.

    This component owns command, observation, reward, and velocity-task state.
    It deliberately does not own model compilation or physics stepping; those
    responsibilities belong to :class:`PhysicsBackend`.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        physics: PhysicsBackend | None = None,
        command: torch.Tensor | None = None,
        action_scale: float = 1.0,
        decimation: int | None = None,
        config: MicroDuckVelocityConfig | None = None,
        actuator_mode: str | None = None,
        action_delay_lag: int | tuple[int, int] = 0,
        domain_randomization: bool = False,
    ) -> None:
        self.physics = physics or PhysicsBackend(
            bundle,
            actuator_mode=actuator_mode,
            decimation=decimation,
        )
        self.bundle = self.physics.bundle
        self.action_scale = action_scale
        self.decimation = self.physics.decimation
        self.config = config or MicroDuckVelocityConfig()
        self.actuator_mode = self.physics.actuator_mode
        self._fixed_command = command is not None
        self.command = (
            torch.zeros(13, dtype=bundle.dtype, device=bundle.device)
            if command is None
            else torch.as_tensor(command, dtype=bundle.dtype, device=bundle.device)
        )
        if self.command.shape != (13,):
            raise ValueError(f"Expected a 13-element command, got {tuple(self.command.shape)}")
        if isinstance(action_delay_lag, tuple):
            low, high = action_delay_lag
            if low < 0 or low > high:
                raise ValueError("action delay range must satisfy 0 <= low <= high")
            self.action_delay_range = (low, high)
        else:
            if action_delay_lag < 0:
                raise ValueError("action_delay_lag must be non-negative")
            self.action_delay_range = (action_delay_lag, action_delay_lag)
        self.domain_randomization = domain_randomization
        self.state: MicroDuckRuntimeState | None = None
        self.last_action = torch.zeros(
            self.bundle.action_size, dtype=self.bundle.dtype, device=self.bundle.device
        )
        self._generator = self.physics._generator
        # These are installed by ManagerBasedTaskEnv.  Keeping the manager
        # references on the task runtime makes the task lifecycle explicit,
        # while the parent environment remains a generic composition shell.
        self.action_manager: ActionManager | None = None
        self.command_manager: CommandManager | None = None
        self.observation_manager: ObservationManager | None = None
        self.reward_manager: RewardManager | None = None
        self.termination_manager: TerminationManager | None = None
        self.event_manager: EventManager | None = None
        self.curriculum_manager: CurriculumManager | None = None

    @property
    def data(self) -> Any | None:
        return self.physics.data

    @data.setter
    def data(self, value: Any | None) -> None:
        self.physics.data = value

    @property
    def step_count(self) -> int:
        return self.physics.step_count

    @step_count.setter
    def step_count(self, value: int) -> None:
        self.physics.state.step_count = value

    @property
    def _bam_vin(self) -> torch.Tensor | None:
        return self.physics._bam_vin

    @_bam_vin.setter
    def _bam_vin(self, value: torch.Tensor | None) -> None:
        self.physics._bam_vin = value

    @property
    def _bam_drop_gain(self) -> torch.Tensor | float | None:
        return self.physics._bam_drop_gain

    @_bam_drop_gain.setter
    def _bam_drop_gain(self, value: torch.Tensor | float | None) -> None:
        self.physics._bam_drop_gain = value

    @property
    def _bam_friction_scale(self) -> torch.Tensor | float:
        return self.physics._bam_friction_scale

    @_bam_friction_scale.setter
    def _bam_friction_scale(self, value: torch.Tensor | float) -> None:
        self.physics._bam_friction_scale = value

    def _random(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.physics.random(dtype=dtype)

    def _sample_range(self, low: float, high: float) -> torch.Tensor:
        return self.physics.sample_range(low, high)

    def _sample_delay(self, low: int, high: int) -> int:
        return self.physics.sample_delay(low, high)

    def _initial_observation(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.data is None:
            raise RuntimeError("Call reset() before initializing observation state")
        base_ang_vel = self.data.sensordata[..., self.bundle.sensor_slices["imu_ang_vel"]].clone()
        gravity_world = torch.zeros(3, dtype=self.bundle.dtype, device=self.bundle.device)
        gravity_world[2] = -1.0
        gravity = self.data.xmat[self.bundle.trunk_body_id].transpose(-1, -2) @ gravity_world
        return base_ang_vel, gravity

    def _joint_measurements(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return resolved output position and motor velocity in actuator order."""

        return self.physics.actuator_measurements()

    def _encoder_velocity(self) -> torch.Tensor:
        """Return the output-side velocity seen by the encoder."""

        return self.physics.encoder_velocity()

    def _restore_model_defaults(self) -> None:
        """Restore scalar model fields before applying a fresh DR sample."""
        self.physics.restore_model_defaults()

    def _apply_domain_randomization(self) -> None:
        if not self.domain_randomization:
            return
        base_dof_armature = self.physics.base_field("dof_armature")
        base_body_mass = self.physics.base_field("body_mass")
        base_body_inertia = self.physics.base_field("body_inertia")
        base_geom_friction = self.physics.base_field("geom_friction")
        base_native_dof_armature = self.physics.base_field("native_dof_armature")
        base_native_body_mass = self.physics.base_field("native_body_mass")
        base_native_body_inertia = self.physics.base_field("native_body_inertia")
        base_native_geom_friction = self.physics.base_field("native_geom_friction")
        randomization = self.config.randomization
        trunk = self.bundle.trunk_body_id
        if self.config.randomize_mass_inertia:
            mass_scale = self._sample_range(*randomization.mass_inertia_range)
            self.bundle.torch_model.body_mass[trunk] = base_body_mass[trunk] * mass_scale
            self.bundle.torch_model.body_inertia[trunk] = base_body_inertia[trunk] * mass_scale
            self.bundle.native_model.body_mass[trunk] = base_native_body_mass[trunk] * float(
                mass_scale
            )
            self.bundle.native_model.body_inertia[trunk] = base_native_body_inertia[trunk] * float(
                mass_scale
            )
        if self.config.randomize_com:
            com_delta = (
                torch.rand(
                    3,
                    generator=self._generator,
                    device=self.bundle.device,
                    dtype=self.bundle.dtype,
                )
                * 2.0
                - 1.0
            ) * randomization.com_range
            self.bundle.torch_model.body_ipos[trunk] += com_delta
            self.bundle.native_model.body_ipos[trunk] += com_delta.detach().cpu().numpy()
        if self.config.randomize_head_com:
            for body_id in self.bundle.head_body_ids:
                delta = (
                    torch.rand(
                        3,
                        generator=self._generator,
                        device=self.bundle.device,
                        dtype=self.bundle.dtype,
                    )
                    * 2.0
                    - 1.0
                ) * randomization.head_com_range
                self.bundle.torch_model.body_ipos[body_id] += delta
                self.bundle.native_model.body_ipos[body_id] += delta.detach().cpu().numpy()
        if self.config.randomize_foot_friction:
            foot_scale = self._sample_range(*randomization.foot_friction_range)
            foot_ids = sorted(
                {geom_id for group in self.bundle.foot_geom_groups for geom_id in group}
            )
            self.bundle.torch_model.geom_friction[foot_ids] = (
                base_geom_friction[foot_ids] * foot_scale
            )
            self.bundle.native_model.geom_friction[foot_ids] = base_native_geom_friction[
                foot_ids
            ] * float(foot_scale)
        if self.config.randomize_armature:
            armature_scale = self._sample_range(*randomization.armature_range)
            self.bundle.torch_model.dof_armature[self.bundle.qvel_indices] = (
                base_dof_armature[self.bundle.qvel_indices] * armature_scale
            )
            indices = self.bundle.qvel_indices.cpu().numpy()
            self.bundle.native_model.dof_armature[indices] = base_native_dof_armature[
                indices
            ] * float(armature_scale)
        if self.actuator_mode == "bam":
            if self._bam_vin is None:
                self._bam_vin = self._sample_range(6.5, 8.2)
                self._bam_drop_gain = self._sample_range(0.0, 0.2)
            self._bam_friction_scale = (
                self._sample_range(*randomization.joint_friction_range)
                if self.config.randomize_joint_friction
                else torch.ones((), dtype=self.bundle.dtype, device=self.bundle.device)
            )
        else:
            self._bam_vin = None
            self._bam_drop_gain = None
            self._bam_friction_scale = torch.ones(
                (), dtype=self.bundle.dtype, device=self.bundle.device
            )
        self.physics.configure_bam(
            vin=self._bam_vin,
            drop_gain=self._bam_drop_gain,
            friction_scale=self._bam_friction_scale,
        )

    def reset(
        self,
        command: torch.Tensor | None = None,
        *,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> torch.Tensor:
        self.physics.set_seed(seed)
        self._generator = self.physics._generator
        if randomize is not None:
            self.domain_randomization = randomize
        if command is not None:
            self.command = torch.as_tensor(
                command, dtype=self.bundle.dtype, device=self.bundle.device
            )
            self._fixed_command = True
        if self.command.shape != (13,):
            raise ValueError(f"Expected a 13-element command, got {tuple(self.command.shape)}")
        self._restore_model_defaults()
        self._apply_domain_randomization()
        if self.domain_randomization and not self._fixed_command:
            self.command = sample_command(
                self.config.command,
                generator=self._generator,
                device=self.bundle.device,
                dtype=self.bundle.dtype,
            )
        qpos = self.bundle.default_qpos.clone()
        if self.domain_randomization:
            qpos[2] = self._sample_range(*self.config.initial_height_range)
        if self.domain_randomization and self.config.randomize_base_orientation:
            roll = torch.deg2rad(
                self._sample_range(
                    -self.config.randomization.base_roll_degrees,
                    self.config.randomization.base_roll_degrees,
                )
            )
            pitch = torch.deg2rad(
                self._sample_range(
                    -self.config.randomization.base_pitch_degrees,
                    self.config.randomization.base_pitch_degrees,
                )
            )
            qpos[3:7] = _quat_from_euler(roll, pitch, torch.zeros_like(roll))
        qvel = torch.zeros(
            self.bundle.native_model.nv,
            dtype=self.bundle.dtype,
            device=self.bundle.device,
        )
        reset_ctrl = (
            torch.zeros(
                self.bundle.native_model.nu,
                dtype=self.bundle.dtype,
                device=self.bundle.device,
            )
            if self.actuator_mode == "bam"
            else self.bundle.default_pose.clone()
        )
        self.physics.reset(qpos=qpos, qvel=qvel, ctrl=reset_ctrl)
        base_ang_vel, gravity = self._initial_observation()
        data = self.data
        if data is None:
            raise RuntimeError("Physics backend did not produce data during reset")
        foot_positions = data.site_xpos[list(self.bundle.foot_site_ids)].clone()
        zero_action = torch.zeros(
            self.bundle.action_size, dtype=self.bundle.dtype, device=self.bundle.device
        )
        self.last_action = zero_action.clone()
        low_delay, high_delay = self.action_delay_range
        delay_lag = self._sample_delay(low_delay, high_delay)
        if (
            self.domain_randomization
            and self.config.randomize_actuator_delay
            and self.action_delay_range == (0, 0)
        ):
            delay_lag = self._sample_delay(3, 6)
        imu_lag = self._sample_delay(*self.config.imu_delay_lag) if self.domain_randomization else 0
        self.state = MicroDuckRuntimeState(
            last_action=zero_action.clone(),
            previous_action=zero_action.clone(),
            previous_joint_velocity=self._encoder_velocity().clone(),
            previous_foot_positions=foot_positions,
            foot_air_time=torch.zeros(2, dtype=self.bundle.dtype, device=self.bundle.device),
            foot_contact=foot_contact_mask(self.data, self.bundle),
            imu_ang_vel_history=[base_ang_vel],
            projected_gravity_history=[gravity],
            delay_buffer=[zero_action.clone() for _ in range(max(delay_lag + 1, 1))],
            delay_lag=delay_lag,
            imu_lag=imu_lag,
            encoder_bias=(
                sample_uniform(
                    (self.config.randomization.encoder_bias_range,) * self.bundle.action_size,
                    generator=self._generator,
                    device=self.bundle.device,
                    dtype=self.bundle.dtype,
                )
                if self.domain_randomization and self.config.randomize_encoder_bias
                else torch.zeros_like(zero_action)
            ),
            imu_quaternion=torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=self.bundle.dtype, device=self.bundle.device
            ),
            next_push_step=(
                self._next_interval_step(self.config.randomization.velocity_push_interval)
                if self.domain_randomization
                else self.config.episode_length_steps + 1
            ),
            next_twist_step=(
                self._next_interval_step(self.config.command.twist_resample_seconds)
                if self.domain_randomization
                else self.config.episode_length_steps + 1
            ),
            next_head_step=(
                self._next_interval_step(self.config.command.head_resample_seconds)
                if self.domain_randomization
                else self.config.episode_length_steps + 1
            ),
            next_body_step=(
                self._next_interval_step(self.config.command.body_resample_seconds)
                if self.domain_randomization
                else self.config.episode_length_steps + 1
            ),
            reward_terms={},
        )
        if (
            self.domain_randomization
            and self.config.randomize_imu_orientation
            and self.config.randomization.imu_angle_degrees > 0
        ):
            axis = torch.randn(
                3, generator=self._generator, device=self.bundle.device, dtype=self.bundle.dtype
            )
            axis = axis / (torch.linalg.vector_norm(axis) + 1e-8)
            angle = torch.deg2rad(self._random() * self.config.randomization.imu_angle_degrees)
            half = angle / 2.0
            self.state.imu_quaternion = torch.cat(
                (torch.cos(half).reshape(1), axis * torch.sin(half))
            )
        if self.event_manager is not None:
            self.event_manager.apply(self, "reset")
        return self.observation()

    def _next_interval_step(self, interval: tuple[float, float]) -> int:
        seconds = float(self._sample_range(*interval))
        return max(1, int(round(seconds / (self.bundle.timestep * self.decimation))))

    def _observation_noise(self, shape: torch.Size) -> torch.Tensor | None:
        if not self.domain_randomization:
            return None
        noise = torch.zeros(shape, dtype=self.bundle.dtype, device=self.bundle.device)
        scales = self.config.actor_noise
        for (start, end), scale in zip(((0, 3), (3, 6), (6, 20), (20, 34)), scales, strict=True):
            noise[start:end] = (
                torch.rand(
                    end - start,
                    generator=self._generator,
                    device=self.bundle.device,
                    dtype=self.bundle.dtype,
                )
                * 2.0
                - 1.0
            ) * scale
        return noise

    def _build_actor_observation(self) -> torch.Tensor:
        if self.data is None or self.state is None:
            raise RuntimeError("Call reset() before observation()")
        imu_index = max(0, len(self.state.imu_ang_vel_history) - 1 - self.state.imu_lag)
        gravity_index = max(0, len(self.state.projected_gravity_history) - 1 - self.state.imu_lag)
        return build_actor_observation(
            self.bundle,
            self.data,
            self.state.last_action,
            self.command,
            joint_position=self._joint_measurements()[0],
            joint_position_bias=self.state.encoder_bias,
            joint_velocity=self.state.previous_joint_velocity,
            base_ang_vel=self.state.imu_ang_vel_history[imu_index],
            projected_gravity=self.state.projected_gravity_history[gravity_index],
            imu_quaternion=self.state.imu_quaternion,
            noise=self._observation_noise(torch.Size((61,))),
        )

    def observation(self) -> torch.Tensor:
        if self.observation_manager is not None:
            return self.observation_manager.compute(self, "actor")
        return self._build_actor_observation()

    def _apply_push(self) -> None:
        if (
            self.data is None
            or self.state is None
            or not self.domain_randomization
            or not self.config.randomize_velocity_pushes
        ):
            return
        if self.step_count < self.state.next_push_step:
            return
        push_low, push_high = self.config.randomization.velocity_push_range
        push = torch.stack(
            (self._sample_range(push_low, push_high), self._sample_range(push_low, push_high))
        )
        qvel = self.data.qvel.clone()
        qvel[:2] += push
        self.physics.forward(qvel=qvel)
        self.state.next_push_step = self.step_count + self._next_interval_step(
            self.config.randomization.velocity_push_interval
        )

    def step(self, action: torch.Tensor) -> EnvStep:
        if self.data is None or self.state is None:
            self.reset()
        if self.data is None or self.state is None:
            raise RuntimeError("Call reset() before step()")
        action = torch.as_tensor(action, dtype=self.bundle.dtype, device=self.bundle.device)
        if action.shape != (self.bundle.action_size,):
            raise ValueError(
                f"Expected action shape ({self.bundle.action_size},), got {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all():
            raise ValueError("Action contains non-finite values")
        if self.event_manager is not None:
            self.event_manager.apply(self, "pre_physics")
        else:
            self._apply_push()
        previous_foot_contact = self.state.foot_contact.clone()
        previous_foot_air_time = self.state.foot_air_time.clone()
        if self.action_manager is not None:
            applied_action, target = self.action_manager.prepare(self, action)
        else:
            self.state.previous_joint_velocity = self._encoder_velocity().clone()
            self.state.previous_action = self.state.last_action.clone()
            self.state.delay_buffer[self.step_count % len(self.state.delay_buffer)] = action.clone()
            delayed_index = (self.step_count - self.state.delay_lag) % len(self.state.delay_buffer)
            applied_action = self.state.delay_buffer[delayed_index]
            target = self.bundle.default_pose + self.action_scale * applied_action
        self.physics.step(target)
        self.state.last_action = action.clone()
        self.last_action = action.clone()
        current_contact = foot_contact_mask(self.data, self.bundle)
        touchdown = current_contact & ~previous_foot_contact
        self.state.foot_air_time = torch.where(
            current_contact,
            torch.zeros_like(self.state.foot_air_time),
            self.state.foot_air_time + self.bundle.timestep * self.decimation,
        )
        self.state.foot_contact = current_contact
        base_ang_vel, gravity = self._initial_observation()
        self.state.imu_ang_vel_history.append(base_ang_vel)
        self.state.projected_gravity_history.append(gravity)
        if len(self.state.imu_ang_vel_history) > 4:
            self.state.imu_ang_vel_history.pop(0)
            self.state.projected_gravity_history.pop(0)
        if (
            self.domain_randomization
            and self.config.delay_update_period > 0
            and self.step_count % self.config.delay_update_period == 0
        ):
            self.state.imu_lag = self._sample_delay(*self.config.imu_delay_lag)
        if self.event_manager is not None:
            # Post-physics terms observe the freshly integrated state and must
            # run before commands, observations, rewards, and terminations.
            self.event_manager.apply(self, "post_physics")
        if self.command_manager is not None:
            self.command_manager.step(self)
        elif self.domain_randomization and not self._fixed_command:
            command_config = self.config.command
            if self.step_count >= self.state.next_twist_step:
                sampled = sample_twist(
                    command_config,
                    generator=self._generator,
                    device=self.bundle.device,
                    dtype=self.bundle.dtype,
                )
                self.command[:3] = sampled[:3]
                self.state.next_twist_step = self.step_count + self._next_interval_step(
                    command_config.twist_resample_seconds
                )
            if self.step_count >= self.state.next_head_step:
                self.command[3:7] = sample_uniform(
                    command_config.head_ranges,
                    generator=self._generator,
                    device=self.bundle.device,
                    dtype=self.bundle.dtype,
                )
                self.state.next_head_step = self.step_count + self._next_interval_step(
                    command_config.head_resample_seconds
                )
            if self.step_count >= self.state.next_body_step:
                self.command[7:13] = sample_uniform(
                    command_config.body_ranges,
                    generator=self._generator,
                    device=self.bundle.device,
                    dtype=self.bundle.dtype,
                )
                self.state.next_body_step = self.step_count + self._next_interval_step(
                    command_config.body_resample_seconds
                )
        observation = self.observation()
        reward_values = {
            "action": action,
            "previous_action": self.state.previous_action,
            "previous_foot_positions": self.state.previous_foot_positions,
            "foot_air_time": previous_foot_air_time,
            "foot_contact": current_contact,
            "foot_touchdown": touchdown,
        }
        if self.reward_manager is not None:
            reward, terms = self.reward_manager.compute(self, **reward_values)
        else:
            reward, terms = compute_reward(
                self.bundle,
                self.data,
                command=self.command,
                action=action,
                previous_action=self.state.previous_action,
                previous_foot_positions=self.state.previous_foot_positions,
                foot_air_time=previous_foot_air_time,
                foot_contact=current_contact,
                config=self.config.rewards,
                foot_touchdown=touchdown,
            )
        self.state.reward_terms = terms
        self.state.previous_foot_positions = self.data.site_xpos[
            list(self.bundle.foot_site_ids)
        ].clone()
        finite = bool(
            torch.isfinite(self.data.qpos).all()
            and torch.isfinite(self.data.qvel).all()
            and torch.isfinite(observation).all()
            and torch.isfinite(reward).all()
        )
        quaternion = self.data.xquat[self.bundle.trunk_body_id]
        cos_tilt = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
        bad_limit = torch.cos(
            torch.deg2rad(
                torch.tensor(
                    self.config.bad_orientation_degrees,
                    dtype=quaternion.dtype,
                    device=quaternion.device,
                )
            )
        )
        bad_orientation_value = bool(cos_tilt < bad_limit)
        if self.termination_manager is not None:
            terminated, truncated, termination_values = self.termination_manager.evaluate(
                self, finite=finite
            )
            bad_orientation_value = termination_values.get("bad_orientation", False)
        else:
            terminated = not finite or bad_orientation_value
            truncated = self.step_count >= self.config.episode_length_steps
            termination_values = {
                "non_finite": not finite,
                "bad_orientation": bad_orientation_value,
                "timeout": truncated,
            }
        if self.curriculum_manager is not None:
            self.curriculum_manager.step(self)
        info: dict[str, Any] = {
            "step": self.step_count,
            "time": float(self.data.time),
            "finite": finite,
            "bad_orientation": bad_orientation_value,
            "terminations": termination_values,
            "reward_terms": {name: float(value) for name, value in terms.items()},
            "applied_action": applied_action.detach().clone(),
        }
        return EnvStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def snapshot(self) -> dict[str, Any]:
        if self.data is None:
            raise RuntimeError("Call reset() before snapshot()")
        return {
            "qpos": self.data.qpos.detach().clone(),
            "qvel": self.data.qvel.detach().clone(),
            "qacc": self.data.qacc.detach().clone(),
            "ctrl": self.data.ctrl.detach().clone(),
            "sensordata": self.data.sensordata.detach().clone(),
            "time": float(self.data.time),
        }


class ManagerBasedTaskEnv:
    """Generic manager-based environment composed from a task runtime/backend.

    The environment owns configuration, scene selection, and the manager graph.
    A task runtime owns task-specific state and lifecycle callbacks, while the
    physics backend owns model/data creation and simulation stepping.  This
    mirrors upstream's separation between ``ManagerBasedRlEnv`` and task MDP
    functions without making the Torch environment inherit from a
    task-specific implementation.
    """

    def __init__(
        self,
        task_cfg: TaskEnvCfg,
        *,
        bundle: ModelBundle | None = None,
        command: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        domain_randomization: bool | None = None,
        runtime_factory: Any | None = None,
        **load_options: Any,
    ) -> None:
        if "robot" not in task_cfg.scene.entities:
            raise ValueError("Task scene must contain a named 'robot' entity")
        robot_cfg = task_cfg.scene.entities["robot"]
        scene_build: SceneBuild = SceneBuilder().build(task_cfg.scene)
        if bundle is None:
            bundle = load_microduck_model(
                xml_path=scene_build.xml_path,
                entity_cfg=robot_cfg,
                device=device,
                dtype=dtype,
                actuator_mode=task_cfg.actions.actuator_mode,
                **load_options,
            )
        self.physics = PhysicsBackend(
            bundle,
            actuator_mode=task_cfg.actions.actuator_mode,
            decimation=load_options.get("decimation"),
        )
        runtime_type = runtime_factory or getattr(task_cfg, "runtime_factory", None)
        if runtime_type is None:
            runtime_type = VelocityTaskRuntime
        self.runtime = runtime_type(
            bundle,
            physics=self.physics,
            command=command,
            action_scale=task_cfg.actions.scale,
            config=task_cfg.runtime,
            actuator_mode=task_cfg.actions.actuator_mode,
            action_delay_lag=task_cfg.actions.delay_lag,
            domain_randomization=(
                task_cfg.metadata.get("domain_randomization", False)
                if domain_randomization is None
                else domain_randomization
            ),
        )
        if bundle.entity_cfg.xml_path.resolve() != robot_cfg.xml_path.resolve():
            raise ValueError(
                "Task entity and model bundle refer to different robot XMLs: "
                f"{robot_cfg.xml_path} != {bundle.entity_cfg.xml_path}"
            )
        if task_cfg.action_size != bundle.action_size:
            raise ValueError(
                f"Task declares {task_cfg.action_size} actions, model exposes {bundle.action_size}"
            )
        self.task_cfg = task_cfg
        self.scene_build = scene_build
        self.action_manager = ActionManager(task_cfg.actions)
        self.command_manager = CommandManager(task_cfg.commands)
        self.observation_manager = ObservationManager(task_cfg.observations)
        self.reward_manager = RewardManager(task_cfg.rewards)
        self.termination_manager = TerminationManager(task_cfg.terminations)
        self.event_manager = EventManager(task_cfg.events)
        self.curriculum_manager = CurriculumManager(task_cfg.curriculum)
        self.runtime.action_manager = self.action_manager
        self.runtime.command_manager = self.command_manager
        self.runtime.observation_manager = self.observation_manager
        self.runtime.reward_manager = self.reward_manager
        self.runtime.termination_manager = self.termination_manager
        self.runtime.event_manager = self.event_manager
        self.runtime.curriculum_manager = self.curriculum_manager

    @property
    def bundle(self) -> ModelBundle:
        return self.physics.bundle

    @property
    def data(self) -> Any | None:
        return self.physics.data

    @data.setter
    def data(self, value: Any | None) -> None:
        self.physics.data = value

    @property
    def state(self) -> MicroDuckRuntimeState | None:
        return self.runtime.state

    @property
    def command(self) -> torch.Tensor:
        return self.runtime.command

    @property
    def config(self) -> Any:
        return self.runtime.config

    @property
    def domain_randomization(self) -> bool:
        return self.runtime.domain_randomization

    @domain_randomization.setter
    def domain_randomization(self, value: bool) -> None:
        self.runtime.domain_randomization = value

    @property
    def step_count(self) -> int:
        return self.physics.step_count

    @property
    def decimation(self) -> int:
        return self.physics.decimation

    @property
    def actuator_mode(self) -> str:
        return self.physics.actuator_mode

    def reset(
        self,
        command: torch.Tensor | None = None,
        *,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> torch.Tensor:
        return self.runtime.reset(command, seed=seed, randomize=randomize)

    def step(self, action: torch.Tensor) -> EnvStep:
        return self.runtime.step(action)

    def observation(self) -> torch.Tensor:
        return self.runtime.observation()

    def snapshot(self) -> dict[str, Any]:
        return self.physics.snapshot()


__all__ = ["EnvStep", "ManagerBasedTaskEnv", "MicroDuckRuntimeState", "VelocityTaskRuntime"]
