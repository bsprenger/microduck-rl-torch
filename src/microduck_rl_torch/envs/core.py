"""Policy-facing MicroDuck velocity environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import mujoco_torch
import torch

from .actuation import (
    BamRuntime,
    apply_bam_fields,
    external_torque,
    friction_budget,
    motor_torque,
)
from .config import MicroDuckVelocityConfig, sample_command, sample_twist, sample_uniform
from .model import MicroDuckModelBundle
from .observations import build_actor_observation
from .rewards import compute_reward, foot_contact_mask

mujoco_api: Any = mujoco


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


class NominalMicroDuckEnv:
    """A scalar eager environment matching the upstream velocity task.

    The state is deliberately explicit so the same ordering can later be
    lifted to ``torch.vmap`` without hiding delays or randomization in global
    buffers. A single environment is currently supported; this is the
    reference path used by the native-vs-Torch trajectory tests.
    """

    def __init__(
        self,
        bundle: MicroDuckModelBundle,
        *,
        command: torch.Tensor | None = None,
        action_scale: float = 1.0,
        decimation: int | None = None,
        config: MicroDuckVelocityConfig | None = None,
        actuator_mode: str | None = None,
        action_delay_lag: int | tuple[int, int] = 0,
        domain_randomization: bool = False,
    ) -> None:
        self.bundle = bundle
        self.action_scale = action_scale
        self.decimation = decimation if decimation is not None else bundle.decimation
        if self.decimation < 1:
            raise ValueError("decimation must be positive")
        self.config = config or MicroDuckVelocityConfig()
        self.actuator_mode = actuator_mode or bundle.actuator_mode
        if self.actuator_mode not in {"bam", "xml"}:
            raise ValueError("actuator_mode must be 'bam' or 'xml'")
        if self.actuator_mode == "bam" and bundle.bam_parameters is None:
            raise ValueError("BAM environment requested with a non-BAM model bundle")
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
        self.data: Any | None = None
        self.state: MicroDuckRuntimeState | None = None
        self.last_action = torch.zeros(bundle.action_size, dtype=bundle.dtype, device=bundle.device)
        self.step_count = 0
        self._generator: torch.Generator | None = None
        self._bam_runtime: BamRuntime | None = None

    def _random(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.rand(
            (),
            generator=self._generator,
            dtype=dtype or self.bundle.dtype,
            device=self.bundle.device,
        )

    def _sample_range(self, low: float, high: float) -> torch.Tensor:
        return self._random() * (high - low) + low

    def _sample_delay(self, low: int, high: int) -> int:
        if low == high:
            return low
        return int(torch.randint(low, high + 1, (), generator=self._generator).item())

    def _initial_observation(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.data is None:
            raise RuntimeError("Call reset() before initializing observation state")
        base_ang_vel = self.data.sensordata[..., self.bundle.sensor_slices["imu_ang_vel"]].clone()
        gravity_world = torch.zeros(3, dtype=self.bundle.dtype, device=self.bundle.device)
        gravity_world[2] = -1.0
        gravity = self.data.xmat[self.bundle.trunk_body_id].transpose(-1, -2) @ gravity_world
        return base_ang_vel, gravity

    def _joint_measurements(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return encoder position and motor-side velocity in actuator order."""

        if self.data is None:
            raise RuntimeError("Call reset() before reading joint measurements")
        position = self.data.qpos.index_select(-1, self.bundle.qpos_indices)
        motor_velocity = self.data.qvel.index_select(-1, self.bundle.qvel_indices)
        if self.bundle.has_backlash:
            backlash_position = self.data.qpos.index_select(-1, self.bundle.backlash_qpos_indices)
            position = position + backlash_position * self.bundle.backlash_mask
        return position, motor_velocity

    def _encoder_velocity(self) -> torch.Tensor:
        """Return the output-side velocity seen by the encoder."""

        if self.data is None:
            raise RuntimeError("Call reset() before reading joint measurements")
        velocity = self.data.qvel.index_select(-1, self.bundle.qvel_indices)
        if self.bundle.has_backlash:
            backlash_velocity = self.data.qvel.index_select(-1, self.bundle.backlash_qvel_indices)
            velocity = velocity + backlash_velocity * self.bundle.backlash_mask
        return velocity

    def _restore_model_defaults(self) -> None:
        """Restore scalar model fields before applying a fresh DR sample."""

        if not hasattr(self, "_base_dof_armature"):
            self._base_dof_armature = self.bundle.torch_model.dof_armature.clone()
            self._base_dof_frictionloss = self.bundle.torch_model.dof_frictionloss.clone()
            self._base_dof_damping = self.bundle.torch_model.dof_damping.clone()
            self._base_body_mass = self.bundle.torch_model.body_mass.clone()
            self._base_body_inertia = self.bundle.torch_model.body_inertia.clone()
            self._base_body_ipos = self.bundle.torch_model.body_ipos.clone()
            self._base_geom_friction = self.bundle.torch_model.geom_friction.clone()
            self._base_native_dof_armature = self.bundle.native_model.dof_armature.copy()
            self._base_native_dof_frictionloss = self.bundle.native_model.dof_frictionloss.copy()
            self._base_native_dof_damping = self.bundle.native_model.dof_damping.copy()
            self._base_native_body_mass = self.bundle.native_model.body_mass.copy()
            self._base_native_body_inertia = self.bundle.native_model.body_inertia.copy()
            self._base_native_body_ipos = self.bundle.native_model.body_ipos.copy()
            self._base_native_geom_friction = self.bundle.native_model.geom_friction.copy()
        self.bundle.torch_model.dof_armature[:] = self._base_dof_armature
        self.bundle.torch_model.dof_frictionloss[:] = self._base_dof_frictionloss
        self.bundle.torch_model.dof_damping[:] = self._base_dof_damping
        self.bundle.torch_model.body_mass[:] = self._base_body_mass
        self.bundle.torch_model.body_inertia[:] = self._base_body_inertia
        self.bundle.torch_model.body_ipos[:] = self._base_body_ipos
        self.bundle.torch_model.geom_friction[:] = self._base_geom_friction
        self.bundle.native_model.dof_armature[:] = self._base_native_dof_armature
        self.bundle.native_model.dof_frictionloss[:] = self._base_native_dof_frictionloss
        self.bundle.native_model.dof_damping[:] = self._base_native_dof_damping
        self.bundle.native_model.body_mass[:] = self._base_native_body_mass
        self.bundle.native_model.body_inertia[:] = self._base_native_body_inertia
        self.bundle.native_model.body_ipos[:] = self._base_native_body_ipos
        self.bundle.native_model.geom_friction[:] = self._base_native_geom_friction
        mujoco_api.mj_setConst(
            self.bundle.native_model, mujoco_api.MjData(self.bundle.native_model)
        )

    def _apply_domain_randomization(self) -> None:
        if not self.domain_randomization:
            return
        randomization = self.config.randomization
        trunk = self.bundle.trunk_body_id
        if self.config.randomize_mass_inertia:
            mass_scale = self._sample_range(*randomization.mass_inertia_range)
            self.bundle.torch_model.body_mass[trunk] = self._base_body_mass[trunk] * mass_scale
            self.bundle.torch_model.body_inertia[trunk] = (
                self._base_body_inertia[trunk] * mass_scale
            )
            self.bundle.native_model.body_mass[trunk] = self._base_native_body_mass[trunk] * float(
                mass_scale
            )
            self.bundle.native_model.body_inertia[trunk] = self._base_native_body_inertia[
                trunk
            ] * float(mass_scale)
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
            foot_ids = list(self.bundle.foot_geom_ids)
            self.bundle.torch_model.geom_friction[foot_ids] = (
                self._base_geom_friction[foot_ids] * foot_scale
            )
            self.bundle.native_model.geom_friction[foot_ids] = self._base_native_geom_friction[
                foot_ids
            ] * float(foot_scale)
        if self.config.randomize_armature:
            armature_scale = self._sample_range(*randomization.armature_range)
            self.bundle.torch_model.dof_armature[self.bundle.qvel_indices] = (
                self._base_dof_armature[self.bundle.qvel_indices] * armature_scale
            )
            indices = self.bundle.qvel_indices.cpu().numpy()
            self.bundle.native_model.dof_armature[indices] = self._base_native_dof_armature[
                indices
            ] * float(armature_scale)
        if self.actuator_mode == "bam":
            if not hasattr(self, "_bam_vin"):
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

    def reset(
        self,
        command: torch.Tensor | None = None,
        *,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> torch.Tensor:
        if seed is not None:
            self._generator = torch.Generator(device=self.bundle.device)
            self._generator.manual_seed(seed)
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
        self.data = self.bundle.new_data()
        qpos = self.data.qpos.clone()
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
        qvel = torch.zeros_like(self.data.qvel)
        reset_ctrl = (
            torch.zeros_like(self.data.ctrl)
            if self.actuator_mode == "bam"
            else self.bundle.default_pose
        )
        self.data = mujoco_torch.forward(
            self.bundle.torch_model,
            self.data.replace(qpos=qpos, qvel=qvel, ctrl=reset_ctrl),
            fixed_iterations=self.bundle.fixed_iterations,
        )
        base_ang_vel, gravity = self._initial_observation()
        foot_positions = self.data.site_xpos[list(self.bundle.foot_site_ids)].clone()
        zero_action = torch.zeros(
            self.bundle.action_size, dtype=self.bundle.dtype, device=self.bundle.device
        )
        self.last_action = zero_action.clone()
        self.step_count = 0
        low_delay, high_delay = self.action_delay_range
        delay_lag = self._sample_delay(low_delay, high_delay)
        if (
            self.domain_randomization
            and self.config.randomize_actuator_delay
            and self.action_delay_range == (0, 0)
        ):
            delay_lag = self._sample_delay(3, 6)
        imu_lag = self._sample_delay(*self.config.imu_delay_lag) if self.domain_randomization else 0
        self._bam_runtime = BamRuntime(
            previous_torque=torch.zeros(
                self.bundle.action_size, dtype=self.bundle.dtype, device=self.bundle.device
            ),
            friction_scale=getattr(
                self,
                "_bam_friction_scale",
                torch.ones((), dtype=self.bundle.dtype, device=self.bundle.device),
            ),
        )
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

    def observation(self) -> torch.Tensor:
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

    def _bam_control(self, target: torch.Tensor) -> torch.Tensor:
        if self.data is None or self._bam_runtime is None or self.bundle.bam_parameters is None:
            raise RuntimeError("BAM state is not initialized")
        parameters = self.bundle.bam_parameters
        position, velocity = self._joint_measurements()
        vin = getattr(self, "_bam_vin", parameters.vin)
        drop_gain = getattr(self, "_bam_drop_gain", parameters.vin_drop_gain)
        if drop_gain is not None:
            vin = vin - drop_gain * self._bam_runtime.previous_torque.abs().sum()
            if parameters.vin_min is not None:
                vin = torch.clamp(vin, min=parameters.vin_min)
        torque = motor_torque(target, position, velocity, params=parameters, vin=vin)
        load = external_torque(self.data, self.bundle.qvel_indices, self.bundle.friction_dof_count)
        friction, damping = friction_budget(
            self._bam_runtime.previous_torque,
            load,
            velocity,
            params=parameters,
            friction_scale=self._bam_runtime.friction_scale,
        )
        apply_bam_fields(self.bundle.torch_model, self.bundle.qvel_indices, friction, damping)
        # MuJoCo refreshes this derived solver array from Model during
        # mj_step.  ``mujoco-torch`` stores it in Data, so update it explicitly
        # when the stateful BAM budget changes between substeps.
        efc_frictionloss = self.data.efc_frictionloss.clone()
        efc_frictionloss[: self.bundle.friction_dof_count] = friction
        self.data = self.data.replace(efc_frictionloss=efc_frictionloss)
        return torque

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
        self.data = mujoco_torch.forward(
            self.bundle.torch_model,
            self.data.replace(qvel=qvel),
            fixed_iterations=self.bundle.fixed_iterations,
        )
        self.state.next_push_step = self.step_count + self._next_interval_step(
            self.config.randomization.velocity_push_interval
        )

    def step(self, action: torch.Tensor) -> EnvStep:
        if self.data is None or self.state is None:
            self.reset()
        if self.data is None or self.state is None:
            raise RuntimeError("Call reset() before step()")
        bam_runtime = self._bam_runtime
        if self.actuator_mode == "bam" and bam_runtime is None:
            raise RuntimeError("BAM state is not initialized")
        action = torch.as_tensor(action, dtype=self.bundle.dtype, device=self.bundle.device)
        if action.shape != (self.bundle.action_size,):
            raise ValueError(f"Expected action shape (14,), got {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            raise ValueError("Action contains non-finite values")
        self._apply_push()
        previous_foot_contact = self.state.foot_contact.clone()
        previous_foot_air_time = self.state.foot_air_time.clone()
        self.state.previous_joint_velocity = self._encoder_velocity().clone()
        self.state.previous_action = self.state.last_action.clone()
        self.state.delay_buffer[self.step_count % len(self.state.delay_buffer)] = action.clone()
        delayed_index = (self.step_count - self.state.delay_lag) % len(self.state.delay_buffer)
        applied_action = self.state.delay_buffer[delayed_index]
        target = self.bundle.default_pose + self.action_scale * applied_action
        for _ in range(self.decimation):
            if self.actuator_mode == "bam":
                torque = self._bam_control(target)
                self.data = self.data.replace(ctrl=torque)
            else:
                self.data = self.data.replace(ctrl=target)
            self.data = mujoco_torch.step(
                self.bundle.torch_model,
                self.data,
                fixed_iterations=self.bundle.fixed_iterations,
            )
            if self.actuator_mode == "bam":
                if bam_runtime is None:
                    raise RuntimeError("BAM state is not initialized")
                bam_runtime.previous_torque = self.data.qfrc_actuator.index_select(
                    -1, self.bundle.qvel_indices
                ).clone()
        self.state.last_action = action.clone()
        self.last_action = action.clone()
        self.step_count += 1
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
        if self.domain_randomization and not self._fixed_command:
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
        bad_orientation = bool(cos_tilt < bad_limit)
        terminated = not finite or bad_orientation
        truncated = self.step_count >= self.config.episode_length_steps
        info: dict[str, Any] = {
            "step": self.step_count,
            "time": float(self.data.time),
            "finite": finite,
            "bad_orientation": bad_orientation,
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
