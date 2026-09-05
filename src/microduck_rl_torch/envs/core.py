"""Manager-based task environment and lifecycle implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .config import sample_uniform
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
from .physics import PhysicsBackend
from .rewards import foot_contact_mask
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
class SensorState:
    """Environment-owned sensor and actuator histories.

    This state is deliberately separate from task state. A future task can
    retain the same sensor buffers while adding posture, prop, or phase data
    under :class:`EnvironmentState.task_data`.
    """

    last_action: torch.Tensor
    previous_action: torch.Tensor
    previous_joint_velocity: torch.Tensor
    previous_foot_positions: torch.Tensor | None
    foot_air_time: torch.Tensor | None
    foot_contact: torch.Tensor | None
    imu_ang_vel_history: list[torch.Tensor]
    projected_gravity_history: list[torch.Tensor]
    delay_buffer: list[torch.Tensor]
    delay_lag: int
    imu_lag: int
    encoder_bias: torch.Tensor
    imu_quaternion: torch.Tensor


@dataclass(frozen=True)
class TransitionData:
    """Data produced by one transition and visible to manager terms."""

    action: torch.Tensor
    previous_action: torch.Tensor
    previous_foot_positions: torch.Tensor | None
    foot_air_time: torch.Tensor | None
    foot_contact: torch.Tensor | None
    foot_touchdown: torch.Tensor | None


@dataclass
class EnvironmentState:
    """Generic environment state owned by ``ManagerBasedTaskEnv``."""

    sensors: SensorState
    reward_terms: dict[str, torch.Tensor]
    manager_data: dict[str, Any] = field(default_factory=dict)
    task_data: dict[str, Any] = field(default_factory=dict)
    transition: TransitionData | None = None


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


class ManagerBasedTaskEnv:
    """Manager-based task environment with the complete lifecycle owner.

    The environment owns configuration, manager execution, task state, and
    lifecycle ordering. :class:`PhysicsBackend` owns only model/data creation
    and low-level simulation mechanics. This mirrors upstream's
    ``ManagerBasedRlEnv``: task factories configure manager terms, while the
    environment executes the same lifecycle for every task.
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
        **load_options: Any,
    ) -> None:
        if "robot" not in task_cfg.scene.entities:
            raise ValueError("Task scene must contain a named 'robot' entity")
        robot_cfg = task_cfg.scene.entities["robot"]
        scene_build: SceneBuild = SceneBuilder().build(task_cfg.scene)
        if bundle is None:
            model_options = dict(load_options)
            model_options.setdefault("timestep", task_cfg.physics_timestep)
            model_options.setdefault("decimation", task_cfg.decimation)
            bundle = load_microduck_model(
                xml_path=scene_build.xml_path,
                entity_cfg=robot_cfg,
                entities=task_cfg.scene.entities,
                device=device,
                dtype=dtype,
                actuator_mode=task_cfg.actions.actuator_mode,
                **model_options,
            )
        configured_decimation = load_options.get("decimation", bundle.decimation)
        self.physics = PhysicsBackend(
            bundle,
            actuator_mode=task_cfg.actions.actuator_mode,
            decimation=configured_decimation,
            actuator_delay_lag=task_cfg.actions.actuator_delay_lag,
        )
        if bundle.entity_cfg.xml_path.resolve() != robot_cfg.xml_path.resolve():
            raise ValueError(
                "Task entity and model bundle refer to different robot XMLs: "
                f"{robot_cfg.xml_path} != {bundle.entity_cfg.xml_path}"
            )
        missing_entities = set(task_cfg.scene.entities) - set(bundle.entities)
        if missing_entities:
            raise ValueError(
                f"Model bundle is missing configured scene entities: {sorted(missing_entities)!r}"
            )
        if task_cfg.action_size != bundle.action_size:
            raise ValueError(
                f"Task declares {task_cfg.action_size} actions, model exposes {bundle.action_size}"
            )
        self.task_cfg = task_cfg
        self.cfg = task_cfg
        self.scene_build = scene_build
        self.bundle = self.physics.bundle
        self.action_scale = task_cfg.actions.scale
        self.decimation = self.physics.decimation
        self.actuator_mode = self.physics.actuator_mode
        self.config = task_cfg.task
        self.action_manager = ActionManager(task_cfg.actions)
        self.command_manager = CommandManager(task_cfg.commands, command=command)
        self.observation_manager = ObservationManager(task_cfg.observations)
        self.reward_manager = RewardManager(
            task_cfg.rewards,
            scale_by_dt=task_cfg.reward_scale_by_dt,
        )
        self.termination_manager = TerminationManager(task_cfg.terminations)
        self.event_manager = EventManager(task_cfg.events)
        self.curriculum_manager = CurriculumManager(task_cfg.curriculum)
        action_delay_lag = task_cfg.actions.delay_lag
        if isinstance(action_delay_lag, tuple):
            low, high = action_delay_lag
            if low < 0 or low > high:
                raise ValueError("action delay range must satisfy 0 <= low <= high")
            self.action_delay_range = (low, high)
        else:
            if action_delay_lag < 0:
                raise ValueError("action_delay_lag must be non-negative")
            self.action_delay_range = (action_delay_lag, action_delay_lag)
        self.domain_randomization = (
            task_cfg.metadata.get("domain_randomization", False)
            if domain_randomization is None
            else domain_randomization
        )
        self.state: EnvironmentState | None = None
        self._generator = self.physics._generator
        self._velocity_reward_cache: dict[str, torch.Tensor] | None = None
        self._startup_events_applied = False

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
    def command(self) -> torch.Tensor:
        if self.command_manager.command is None:
            raise RuntimeError("Command manager has not been reset")
        return self.command_manager.command

    @command.setter
    def command(self, value: torch.Tensor) -> None:
        self.command_manager.set_command(value)

    @property
    def transition(self) -> TransitionData | None:
        """Expose the current transition without duplicating ownership."""

        return None if self.state is None else self.state.transition

    def entity(self, name: str) -> Any:
        """Return a resolved scene entity view for task terms."""

        return self.bundle.entity(name)

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

    def _initial_observation(self) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.data is None:
            raise RuntimeError("Call reset() before initializing observation state")
        imu_slice = self.bundle.sensor_slices.get("imu_ang_vel")
        base_ang_vel = (
            self.data.sensordata[..., imu_slice].clone() if imu_slice is not None else None
        )
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
            self.physics.configure_bam(
                vin=None,
                drop_gain=None,
                friction_scale=torch.ones((), dtype=self.bundle.dtype, device=self.bundle.device),
            )
            return
        base_dof_armature = self.physics.base_field("dof_armature")
        base_body_mass = self.physics.base_field("body_mass")
        base_body_inertia = self.physics.base_field("body_inertia")
        base_geom_friction = self.physics.base_field("geom_friction")
        base_native_dof_armature = self.physics.base_field("native_dof_armature")
        base_native_body_mass = self.physics.base_field("native_body_mass")
        base_native_body_inertia = self.physics.base_field("native_body_inertia")
        base_native_geom_friction = self.physics.base_field("native_geom_friction")
        randomization = getattr(self.config, "randomization", None)
        if randomization is None:
            self.physics.configure_bam(
                vin=None,
                drop_gain=None,
                friction_scale=torch.ones((), dtype=self.bundle.dtype, device=self.bundle.device),
            )
            return
        trunk = self.bundle.trunk_body_id
        if getattr(self.config, "randomize_mass_inertia", False):
            mass_scale = self._sample_range(*randomization.mass_inertia_range)
            self.bundle.torch_model.body_mass[trunk] = base_body_mass[trunk] * mass_scale
            self.bundle.torch_model.body_inertia[trunk] = base_body_inertia[trunk] * mass_scale
            self.bundle.native_model.body_mass[trunk] = base_native_body_mass[trunk] * float(
                mass_scale
            )
            self.bundle.native_model.body_inertia[trunk] = base_native_body_inertia[trunk] * float(
                mass_scale
            )
        if getattr(self.config, "randomize_com", False):
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
        if getattr(self.config, "randomize_head_com", False):
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
        if getattr(self.config, "randomize_foot_friction", False) and self.bundle.foot_geom_groups:
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
        if getattr(self.config, "randomize_armature", False):
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
                if getattr(self.config, "randomize_joint_friction", False)
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
            self.command_manager.set_command(command)
        self._restore_model_defaults()
        self._apply_domain_randomization()
        qpos = self.bundle.default_qpos.clone()
        initial_height_range = getattr(self.config, "initial_height_range", None)
        if self.domain_randomization and initial_height_range is not None:
            qpos[2] = self._sample_range(*initial_height_range)
        if self.domain_randomization and getattr(self.config, "randomize_base_orientation", False):
            randomization = getattr(self.config, "randomization", None)
            if randomization is None:
                raise RuntimeError("Base orientation randomization requires randomization ranges")
            roll = torch.deg2rad(
                self._sample_range(
                    -randomization.base_roll_degrees,
                    randomization.base_roll_degrees,
                )
            )
            pitch = torch.deg2rad(
                self._sample_range(
                    -randomization.base_pitch_degrees,
                    randomization.base_pitch_degrees,
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
        self._velocity_reward_cache = None
        base_ang_vel, gravity = self._initial_observation()
        randomization = getattr(self.config, "randomization", None)
        data = self.data
        if data is None:
            raise RuntimeError("Physics backend did not produce data during reset")
        foot_positions = (
            data.site_xpos[list(self.bundle.foot_site_ids)].clone()
            if self.bundle.foot_site_ids
            else None
        )
        zero_action = torch.zeros(
            self.bundle.action_size, dtype=self.bundle.dtype, device=self.bundle.device
        )
        low_delay, high_delay = self.action_delay_range
        delay_lag = self._sample_delay(low_delay, high_delay)
        if (
            self.domain_randomization
            and getattr(self.config, "randomize_actuator_delay", False)
            and self.action_delay_range == (0, 0)
        ):
            delay_lag = self._sample_delay(3, 6)
        imu_lag_range = getattr(self.config, "imu_delay_lag", (0, 0))
        imu_lag = self._sample_delay(*imu_lag_range) if self.domain_randomization else 0
        self.state = EnvironmentState(
            sensors=SensorState(
                last_action=zero_action.clone(),
                previous_action=zero_action.clone(),
                previous_joint_velocity=self._encoder_velocity().clone(),
                previous_foot_positions=foot_positions,
                foot_air_time=(
                    torch.zeros(
                        len(self.bundle.foot_geom_groups),
                        dtype=self.bundle.dtype,
                        device=self.bundle.device,
                    )
                    if self.bundle.foot_geom_groups
                    else None
                ),
                foot_contact=foot_contact_mask(self.data, self.bundle),
                imu_ang_vel_history=[] if base_ang_vel is None else [base_ang_vel],
                projected_gravity_history=[] if base_ang_vel is None else [gravity],
                delay_buffer=[zero_action.clone() for _ in range(max(delay_lag + 1, 1))],
                delay_lag=delay_lag,
                imu_lag=imu_lag,
                encoder_bias=(
                    sample_uniform(
                        (getattr(randomization, "encoder_bias_range", (0.0, 0.0)),)
                        * self.bundle.action_size,
                        generator=self._generator,
                        device=self.bundle.device,
                        dtype=self.bundle.dtype,
                    )
                    if self.domain_randomization
                    and getattr(self.config, "randomize_encoder_bias", False)
                    and randomization is not None
                    else torch.zeros_like(zero_action)
                ),
                imu_quaternion=torch.tensor(
                    [1.0, 0.0, 0.0, 0.0],
                    dtype=self.bundle.dtype,
                    device=self.bundle.device,
                ),
            ),
            reward_terms={},
        )
        if not self._startup_events_applied:
            self.event_manager.startup(self)
            self._startup_events_applied = True
        self.command_manager.reset(self)
        if (
            self.domain_randomization
            and getattr(self.config, "randomize_imu_orientation", False)
            and randomization is not None
            and getattr(randomization, "imu_angle_degrees", 0.0) > 0
        ):
            axis = torch.randn(
                3, generator=self._generator, device=self.bundle.device, dtype=self.bundle.dtype
            )
            axis = axis / (torch.linalg.vector_norm(axis) + 1e-8)
            angle = torch.deg2rad(self._random() * randomization.imu_angle_degrees)
            half = angle / 2.0
            self.state.sensors.imu_quaternion = torch.cat(
                (torch.cos(half).reshape(1), axis * torch.sin(half))
            )
        self.event_manager.reset(self)
        # Reset events are allowed to mutate qpos/qvel.  Refresh all derived
        # quantities and history baselines after those mutations so the first
        # policy step cannot observe a synthetic foot velocity/contact edge.
        self.physics.forward()
        base_ang_vel, gravity = self._initial_observation()
        data = self.data
        if data is None:
            raise RuntimeError("Physics backend lost data after reset event")
        self.state.sensors.previous_joint_velocity = self._encoder_velocity().clone()
        if self.bundle.foot_site_ids:
            self.state.sensors.previous_foot_positions = data.site_xpos[
                list(self.bundle.foot_site_ids)
            ].clone()
        self.state.sensors.foot_contact = foot_contact_mask(data, self.bundle)
        self.state.sensors.imu_ang_vel_history = [] if base_ang_vel is None else [base_ang_vel]
        self.state.sensors.projected_gravity_history = [] if base_ang_vel is None else [gravity]
        reset_ids = torch.zeros(1, dtype=torch.long, device=self.bundle.device)
        self.observation_manager.reset(self, reset_ids)
        self.reward_manager.reset(self, reset_ids)
        self.termination_manager.reset(self, reset_ids)
        self.curriculum_manager.reset(self, reset_ids)
        return self.observation()

    def _next_interval_step(self, interval: tuple[float, float]) -> int:
        seconds = float(self._sample_range(*interval))
        return max(1, int(round(seconds / (self.bundle.timestep * self.decimation))))

    def _observation_noise(self, shape: torch.Size, scale: float) -> torch.Tensor:
        """Generate one term's configured uniform observation noise."""

        if not self.domain_randomization or scale == 0.0:
            return torch.zeros(shape, dtype=self.bundle.dtype, device=self.bundle.device)
        return (
            torch.rand(
                shape,
                generator=self._generator,
                device=self.bundle.device,
                dtype=self.bundle.dtype,
            )
            * 2.0
            - 1.0
        ) * scale

    def observation(self, group: str = "actor") -> torch.Tensor:
        """Compute one named observation group."""

        return self.observation_manager.compute(self, group)

    def observations(self) -> dict[str, torch.Tensor]:
        """Compute every enabled configured observation group."""

        return {
            name: self.observation(name)
            for name, group in self.task_cfg.observations.groups.items()
            if group.enabled
        }

    def _apply_push(self) -> None:
        if self.data is None or self.state is None:
            return
        if not self.domain_randomization or not getattr(
            self.config, "randomize_velocity_pushes", False
        ):
            return
        randomization = getattr(self.config, "randomization", None)
        if randomization is None:
            return
        push_low, push_high = randomization.velocity_push_range
        push = torch.stack(
            (self._sample_range(push_low, push_high), self._sample_range(push_low, push_high))
        )
        qvel = self.data.qvel.clone()
        qvel[:2] += push
        self.physics.forward(qvel=qvel)

    def step(self, action: torch.Tensor) -> EnvStep:
        if self.data is None or self.state is None:
            self.reset()
        if self.data is None or self.state is None:
            raise RuntimeError("Call reset() before step()")
        sensor = self.state.sensors
        action = torch.as_tensor(action, dtype=self.bundle.dtype, device=self.bundle.device)
        if action.shape != (self.bundle.action_size,):
            raise ValueError(
                f"Expected action shape ({self.bundle.action_size},), got {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all():
            raise ValueError("Action contains non-finite values")
        self.event_manager.apply(self, "pre_physics")
        previous_foot_contact = (
            sensor.foot_contact.clone() if sensor.foot_contact is not None else None
        )
        previous_foot_air_time = (
            sensor.foot_air_time.clone() if sensor.foot_air_time is not None else None
        )
        applied_action, target = self.action_manager.prepare(self, action)
        self.physics.step(target)
        sensor.last_action = action.clone()
        current_contact = foot_contact_mask(self.data, self.bundle)
        touchdown = (
            current_contact & ~previous_foot_contact if previous_foot_contact is not None else None
        )
        if sensor.foot_air_time is not None:
            sensor.foot_air_time = torch.where(
                current_contact,
                torch.zeros_like(sensor.foot_air_time),
                sensor.foot_air_time + self.bundle.timestep * self.decimation,
            )
        sensor.foot_contact = current_contact
        base_ang_vel, gravity = self._initial_observation()
        if base_ang_vel is not None:
            sensor.imu_ang_vel_history.append(base_ang_vel)
            sensor.projected_gravity_history.append(gravity)
            if len(sensor.imu_ang_vel_history) > 4:
                sensor.imu_ang_vel_history.pop(0)
                sensor.projected_gravity_history.pop(0)
        if (
            self.domain_randomization
            and getattr(self.config, "delay_update_period", 0) > 0
            and self.step_count % getattr(self.config, "delay_update_period", 1) == 0
        ):
            sensor.imu_lag = self._sample_delay(*getattr(self.config, "imu_delay_lag", (0, 0)))
        # Post-physics events observe the freshly integrated state.  Reward and
        # termination terms use the command that produced this transition;
        # command resampling happens after them, matching upstream.
        self.event_manager.apply(self, "post_physics")
        transition = TransitionData(
            action=action,
            previous_action=sensor.previous_action,
            previous_foot_positions=sensor.previous_foot_positions,
            foot_air_time=previous_foot_air_time,
            foot_contact=current_contact,
            foot_touchdown=touchdown,
        )
        self.state.transition = transition
        self._velocity_reward_cache = None
        reward, terms = self.reward_manager.compute(self)
        self.state.reward_terms = terms
        if self.bundle.foot_site_ids:
            sensor.previous_foot_positions = self.data.site_xpos[
                list(self.bundle.foot_site_ids)
            ].clone()
        finite = bool(
            torch.isfinite(self.data.qpos).all()
            and torch.isfinite(self.data.qvel).all()
            and torch.isfinite(reward).all()
        )
        terminated, truncated, termination_values = self.termination_manager.evaluate(
            self, finite=finite
        )
        bad_orientation_value = termination_values.get("bad_orientation", False)
        self.curriculum_manager.step(self)
        self.command_manager.step(self)
        self.event_manager.apply(self, "step")
        self.event_manager.apply(self, "interval")
        observation = self.observation()
        if not torch.isfinite(observation).all():
            finite = False
            terminated = True
            termination_values["non_finite"] = True
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


__all__ = ["EnvStep", "EnvironmentState", "ManagerBasedTaskEnv", "SensorState", "TransitionData"]
