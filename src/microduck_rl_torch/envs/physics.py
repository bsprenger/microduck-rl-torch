"""Generic MuJoCo-Torch physics backend.

This module owns simulation mechanics only.  It deliberately has no knowledge
of commands, observations, rewards, terminations, or task curricula.  The
manager-based environment composes it and uses the model bundle's resolved
actuator mapping to express task-specific behavior above this boundary.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
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
from .model import ModelBundle, clone_model_bundle

mujoco_api: Any = mujoco

# Fields whose values are safe to restore and mutate between episodes.  The
# list intentionally covers the generic MuJoCo model surface rather than the
# handful of Microduck fields used by the first task.  Derived constants are
# recomputed after a mutation and are never exposed as mutation targets.
MUTABLE_MODEL_FIELDS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "body_pos",
    "body_quat",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_margin",
    "geom_gap",
    "geom_size",
    "geom_pos",
    "geom_quat",
    "dof_armature",
    "dof_frictionloss",
    "dof_damping",
    "jnt_stiffness",
    "jnt_damping",
    "jnt_frictionloss",
    "jnt_armature",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_gear",
    "actuator_ctrlrange",
    "actuator_forcerange",
    "tendon_stiffness",
    "tendon_damping",
    "tendon_frictionloss",
)

# Fields whose shape/topology and mujoco-torch collision/transmission
# precomputation remain valid after an in-place value update.  The broader
# registry above is retained for baseline capture/diagnostics, but the
# mutation manager only permits this subset between compilations.
SAFE_RUNTIME_MODEL_FIELDS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_margin",
    "geom_gap",
    "dof_armature",
    "dof_frictionloss",
    "dof_damping",
    "jnt_stiffness",
    "jnt_damping",
    "jnt_frictionloss",
    "jnt_armature",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_ctrlrange",
    "actuator_forcerange",
    "tendon_stiffness",
    "tendon_damping",
    "tendon_frictionloss",
)


@dataclass
class PhysicsState:
    """Backend-owned state that is independent of any task semantics."""

    data: Any | None = None
    step_count: int = 0
    previous_torque: torch.Tensor | None = None
    actuator_delay_buffer: list[torch.Tensor] = field(default_factory=list)
    actuator_delay_lag: int = 0
    actuator_delay_step: int = 0
    actuator_delay_phase: int = 0


class PhysicsBackend:
    """Device-resident MuJoCo-Torch simulation backend.

    The backend supports arbitrary compiled models that satisfy the model
    bundle's action/data contract.  Semantic task quantities such as feet,
    trunk pose, and commands are intentionally left to entity views and task
    terms above this layer.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        actuator_mode: str | None = None,
        decimation: int | None = None,
    ) -> None:
        self.bundle = bundle
        self.actuator_mode = actuator_mode or bundle.actuator_mode
        if self.actuator_mode not in {"bam", "xml"}:
            raise ValueError("actuator_mode must be 'bam' or 'xml'")
        if self.actuator_mode == "bam" and bundle.bam_parameters is None:
            raise ValueError("BAM backend requested with a non-BAM model bundle")
        self.decimation = decimation if decimation is not None else bundle.decimation
        if self.decimation < 1:
            raise ValueError("decimation must be positive")
        self.state = PhysicsState()
        self._generator: torch.Generator | None = None
        self._bam_runtime: BamRuntime | None = None
        self._bam_vin: torch.Tensor | None = None
        self._bam_drop_gain: torch.Tensor | float | None = None
        self._bam_friction_scale: torch.Tensor | float = 1.0
        self._bam_kp_scale: torch.Tensor | float = 1.0
        self._bam_kd_scale: torch.Tensor | float = 1.0
        self._base_fields: dict[str, Any] | None = None
        self._native_constant_data: Any | None = None
        self._substep_callback: Any | None = None

    @property
    def data(self) -> Any | None:
        return self.state.data

    @data.setter
    def data(self, value: Any | None) -> None:
        self.state.data = value

    @property
    def step_count(self) -> int:
        return self.state.step_count

    @step_count.setter
    def step_count(self, value: int) -> None:
        self.state.step_count = int(value)

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @property
    def dtype(self) -> torch.dtype:
        return self.bundle.dtype

    @property
    def timestep(self) -> float:
        return self.bundle.timestep

    @property
    def action_size(self) -> int:
        return self.bundle.action_size

    @property
    def generators(self) -> tuple[torch.Generator, ...]:
        """Expose the backend RNG using the batch-compatible interface."""

        if self._generator is None:
            self._generator = torch.Generator(device=self.device)
        return (self._generator,)

    @property
    def step_counts(self) -> torch.Tensor:
        """Return the scalar step counter with a leading environment dimension."""

        return torch.tensor([self.step_count], dtype=torch.long, device=self.device)

    def random(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Sample a scalar on the backend device using the backend RNG."""

        return torch.rand(
            (),
            generator=self._generator,
            dtype=dtype or self.dtype,
            device=self.device,
        )

    def random_tensor(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        normal: bool = False,
    ) -> torch.Tensor:
        """Sample a tensor on this backend's private RNG stream."""

        sampler = torch.randn if normal else torch.rand
        return sampler(
            shape,
            generator=self._generator,
            dtype=dtype or self.dtype,
            device=self.device,
        )

    def sample_range(
        self,
        low: float,
        high: float,
        env_ids: torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        del env_ids
        return self.random() * (high - low) + low

    def sample_delay(
        self,
        low: int,
        high: int,
        env_ids: torch.Tensor | slice | None = None,
        *,
        per_env: bool = True,
    ) -> int:
        del env_ids
        if low == high:
            return low
        return int(
            torch.randint(
                low,
                high + 1,
                (),
                generator=self._generator,
                device=self.device,
            ).item()
        )

    def set_seed(self, seed: int | None, env_ids: torch.Tensor | slice | None = None) -> None:
        del env_ids
        if seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(seed)

    def set_substep_callback(self, callback: Any | None) -> None:
        """Install a backend-owned callback executed after each physics substep.

        The callback is part of the backend lifecycle, rather than an argument
        threaded through every call site.  This keeps the public ``step`` call
        stable for wrappers and instrumentation while allowing scene-owned
        stateful sensors to observe every decimated substep.
        """

        self._substep_callback = callback

    def new_data(self) -> Any:
        """Allocate and forward fresh device-resident simulation data."""

        return self.bundle.new_data()

    def restore_model_defaults(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Restore mutable model fields before a new mutation sample."""

        del env_ids

        if self._base_fields is None:
            self._base_fields = {}
            for name in MUTABLE_MODEL_FIELDS:
                torch_value = getattr(self.bundle.torch_model, name, None)
                native_value = getattr(self.bundle.native_model, name, None)
                if torch_value is not None:
                    self._base_fields[name] = torch_value.clone()
                if native_value is not None:
                    self._base_fields[f"native_{name}"] = native_value.copy()
        fields = self._base_fields
        for name in MUTABLE_MODEL_FIELDS:
            torch_value = getattr(self.bundle.torch_model, name, None)
            native_value = getattr(self.bundle.native_model, name, None)
            if torch_value is not None and name in fields:
                torch_value[:] = fields[name]
            if native_value is not None and f"native_{name}" in fields:
                native_value[:] = fields[f"native_{name}"]
        self.recompute_model_constants()

    def recompute_model_constants(self) -> None:
        """Rebuild native and Torch derived model data after a mutation.

        MuJoCo exposes several cached quantities (subtree masses, inverse
        weights, mass-matrix diagonals, and address-dependent collision
        metadata) that are not updated by assigning a raw model field.  The
        Torch driver also specializes these values at ``device_put`` time.
        Recompute both layers at the mutation boundary so a batched child
        cannot accidentally retain row-zero physics constants.
        """

        if self._native_constant_data is None:
            self._native_constant_data = mujoco_api.MjData(self.bundle.native_model)
        mujoco_api.mj_setConst(self.bundle.native_model, self._native_constant_data)
        # Re-device-putting a mesh-heavy model on every reset is both wasteful
        # and leaks large driver-side precomputation buffers.  The current
        # backend's mutable randomization fields have stable addresses; copy
        # the native recomputed derived arrays into the already-specialized
        # Torch model instead.
        for name in (
            "body_subtreemass",
            "body_invweight0",
            "dof_invweight0",
            "dof_M0",
        ):
            native_value = getattr(self.bundle.native_model, name, None)
            torch_value = getattr(self.bundle.torch_model, name, None)
            if native_value is None or torch_value is None:
                continue
            native_tensor = torch.as_tensor(
                native_value,
                dtype=self.bundle.dtype,
                device=self.bundle.device,
            )
            if native_tensor.shape != torch_value.shape:
                raise RuntimeError(
                    f"Derived model field {name!r} changed shape from "
                    f"{tuple(torch_value.shape)} to {tuple(native_tensor.shape)}"
                )
            torch_value.copy_(native_tensor)

    def base_field(self, name: str) -> Any:
        """Return a read-only reference to a captured mutable model field."""

        if self._base_fields is None:
            self.restore_model_defaults()
        assert self._base_fields is not None
        try:
            return self._base_fields[name]
        except KeyError as exc:
            raise KeyError(f"Unknown captured model field {name!r}") from exc

    def model_field(self, name: str) -> Any:
        """Return a mutable Torch model field by declarative name."""

        if name not in MUTABLE_MODEL_FIELDS:
            raise KeyError(f"Unknown or derived mutable model field {name!r}")
        value = getattr(self.bundle.torch_model, name, None)
        if value is None:
            raise KeyError(f"Model does not expose mutable field {name!r}")
        return value

    def native_model_field(self, name: str) -> Any:
        """Return the corresponding mutable native MuJoCo model field."""

        if name not in MUTABLE_MODEL_FIELDS:
            raise KeyError(f"Unknown or derived mutable model field {name!r}")
        value = getattr(self.bundle.native_model, name, None)
        if value is None:
            raise KeyError(f"Native model does not expose mutable field {name!r}")
        return value

    def configure_bam(
        self,
        *,
        vin: torch.Tensor | None = None,
        drop_gain: torch.Tensor | float | None = None,
        friction_scale: torch.Tensor | float = 1.0,
        kp_scale: torch.Tensor | float = 1.0,
        kd_scale: torch.Tensor | float = 1.0,
    ) -> None:
        """Configure per-rollout BAM parameters without coupling to a task."""

        self._bam_vin = vin
        self._bam_drop_gain = drop_gain
        self._bam_friction_scale = friction_scale
        self._bam_kp_scale = kp_scale
        self._bam_kd_scale = kd_scale

    def reset(
        self,
        *,
        qpos: torch.Tensor | None = None,
        qvel: torch.Tensor | None = None,
        ctrl: torch.Tensor | None = None,
        seed: int | None = None,
        env_ids: torch.Tensor | slice | None = None,
        mocap_pos: torch.Tensor | None = None,
        mocap_quat: torch.Tensor | None = None,
        xfrc_applied: torch.Tensor | None = None,
    ) -> Any:
        """Reset only the physical state; task reset events run above this layer."""

        if env_ids is not None:
            ids = (
                torch.arange(1, device=self.device)[env_ids]
                if isinstance(env_ids, slice)
                else torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
            )
            if ids.numel() != 1 or int(ids[0]) != 0:
                raise ValueError("A scalar physics backend only accepts env_ids=[0]")
        self.set_seed(seed)
        data = self.new_data()
        qpos_value = data.qpos.clone() if qpos is None else qpos.to(self.device, self.dtype)
        qvel_value = (
            self.bundle.default_qvel.clone() if qvel is None else qvel.to(self.device, self.dtype)
        )
        if ctrl is None:
            ctrl_value = (
                torch.zeros_like(data.ctrl)
                if self.actuator_mode == "bam"
                else self.bundle.default_ctrl.clone()
            )
        else:
            ctrl_value = ctrl.to(self.device, self.dtype)
        replacements: dict[str, Any] = {"qpos": qpos_value, "qvel": qvel_value, "ctrl": ctrl_value}
        for name, value in (
            ("mocap_pos", mocap_pos),
            ("mocap_quat", mocap_quat),
            ("xfrc_applied", xfrc_applied),
        ):
            if value is not None:
                replacements[name] = value.to(self.device, self.dtype)
        self.data = mujoco_torch.forward(
            self.bundle.torch_model,
            data.replace(**replacements),
            fixed_iterations=self.bundle.fixed_iterations,
        )
        self.state.step_count = 0
        self.state.previous_torque = torch.zeros(
            self.action_size, dtype=self.dtype, device=self.device
        )
        self._reset_actuator_delay()
        self._bam_runtime = BamRuntime(
            previous_torque=self.state.previous_torque.clone(),
            friction_scale=self._bam_friction_scale,
        )
        return self.data

    def _reset_actuator_delay(self) -> None:
        """Reset the entity actuator delay at the physics-step boundary."""

        config = self.bundle.actuator_delay
        state = self.state
        state.actuator_delay_buffer = []
        state.actuator_delay_lag = 0
        state.actuator_delay_step = 0
        state.actuator_delay_phase = (
            int(
                torch.randint(
                    0,
                    config.delay_update_period,
                    (),
                    generator=self._generator,
                    device=self.device,
                ).item()
            )
            if config.delay_update_period > 0 and config.delay_per_env_phase
            else 0
        )

    def _apply_actuator_delay(self, target: torch.Tensor) -> torch.Tensor:
        """Append and serve a command using delayed-buffer semantics."""

        config = self.bundle.actuator_delay
        if config.delay_max_lag <= 0:
            return target
        state = self.state
        if config.delay_update_period > 0:
            should_update = (
                state.actuator_delay_step + state.actuator_delay_phase
            ) % config.delay_update_period == 0
        else:
            should_update = True
        if should_update:
            hold = bool(
                torch.rand(
                    (), generator=self._generator, device=self.device, dtype=self.dtype
                ).item()
                < config.delay_hold_prob
            )
            if not hold:
                state.actuator_delay_lag = self.sample_delay(
                    config.delay_min_lag, config.delay_max_lag
                )
        state.actuator_delay_buffer.append(target.detach().clone())
        max_length = config.delay_max_lag + 1
        if len(state.actuator_delay_buffer) > max_length:
            state.actuator_delay_buffer.pop(0)
        state.actuator_delay_step += 1
        available_lag = min(state.actuator_delay_lag, len(state.actuator_delay_buffer) - 1)
        return state.actuator_delay_buffer[-1 - available_lag]

    def forward(
        self,
        *,
        qpos: torch.Tensor | None = None,
        qvel: torch.Tensor | None = None,
        ctrl: torch.Tensor | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> Any:
        """Forward the current state after an external state mutation."""

        if env_ids is not None:
            ids = (
                torch.arange(1, device=self.device)[env_ids]
                if isinstance(env_ids, slice)
                else torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
            )
            if ids.numel() != 1 or int(ids[0]) != 0:
                raise ValueError("A scalar physics backend only accepts env_ids=[0]")
        if self.data is None:
            raise RuntimeError("Reset the physics backend before forwarding")
        replacements: dict[str, Any] = {}
        if qpos is not None:
            replacements["qpos"] = qpos
        if qvel is not None:
            replacements["qvel"] = qvel
        if ctrl is not None:
            replacements["ctrl"] = ctrl
        self.data = mujoco_torch.forward(
            self.bundle.torch_model,
            self.data.replace(**replacements),
            fixed_iterations=self.bundle.fixed_iterations,
        )
        return self.data

    def actuator_measurements(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return output-position and motor-side velocity in actuator order."""

        if self.data is None:
            raise RuntimeError("Reset the physics backend before reading measurements")
        safe_qpos = self.bundle.qpos_indices.clamp_min(0)
        safe_qvel = self.bundle.qvel_indices.clamp_min(0)
        position = self.data.qpos.index_select(-1, safe_qpos)
        velocity = self.data.qvel.index_select(-1, safe_qvel)
        joint_mask = self.bundle.actuator_joint_mask
        position = torch.where(joint_mask, position, torch.zeros_like(position))
        velocity = torch.where(joint_mask, velocity, torch.zeros_like(velocity))
        if self.bundle.has_backlash:
            backlash_position = self.data.qpos.index_select(-1, self.bundle.backlash_qpos_indices)
            position = position + backlash_position * self.bundle.backlash_mask
        return position, velocity

    def encoder_velocity(self) -> torch.Tensor:
        """Return output-side velocity for the resolved actuator mapping."""

        if self.data is None:
            raise RuntimeError("Reset the physics backend before reading measurements")
        velocity = self.data.qvel.index_select(-1, self.bundle.qvel_indices.clamp_min(0))
        velocity = torch.where(
            self.bundle.actuator_joint_mask, velocity, torch.zeros_like(velocity)
        )
        if self.bundle.has_backlash:
            backlash_velocity = self.data.qvel.index_select(-1, self.bundle.backlash_qvel_indices)
            velocity = velocity + backlash_velocity * self.bundle.backlash_mask
        return velocity

    def compute_control(self, target: torch.Tensor) -> torch.Tensor:
        """Convert an action target into the configured actuator control."""

        if self.data is None or self._bam_runtime is None or self.bundle.bam_parameters is None:
            raise RuntimeError("BAM state is not initialized")
        parameters = self.bundle.bam_parameters
        position, velocity = self.actuator_measurements()
        vin: Any = self._bam_vin if self._bam_vin is not None else parameters.vin
        drop_gain = (
            self._bam_drop_gain if self._bam_drop_gain is not None else parameters.vin_drop_gain
        )
        if drop_gain is not None:
            vin = vin - drop_gain * self._bam_runtime.previous_torque.abs().sum()
            if parameters.vin_min is not None:
                vin = torch.clamp(vin, min=parameters.vin_min)
        torque = motor_torque(
            target,
            position,
            velocity * self._bam_kd_scale,
            params=parameters,
            vin=vin,
            kp_fw=parameters.kp_fw * self._bam_kp_scale,
        )
        load = external_torque(
            self.data,
            self.bundle.qvel_indices,
            self.bundle.friction_dof_count,
        )
        friction, damping = friction_budget(
            self._bam_runtime.previous_torque,
            load,
            velocity,
            params=parameters,
            friction_scale=self._bam_runtime.friction_scale,
        )
        apply_bam_fields(self.bundle.torch_model, self.bundle.qvel_indices, friction, damping)
        efc_frictionloss = self.data.efc_frictionloss.clone()
        efc_frictionloss[: self.bundle.friction_dof_count] = friction
        self.data = self.data.replace(efc_frictionloss=efc_frictionloss)
        return torque

    def step(
        self,
        target: torch.Tensor,
        *,
        target_type: str | tuple[str, ...] = "position",
        direct_ctrl: torch.Tensor | None = None,
        direct_ctrl_mask: torch.Tensor | None = None,
        substep_callback: Any | None = None,
    ) -> Any:
        """Apply one control target for the configured physics decimation."""

        if self.data is None:
            raise RuntimeError("Reset the physics backend before stepping")
        target = torch.as_tensor(target, dtype=self.dtype, device=self.device)
        if target.shape != (self.action_size,):
            raise ValueError(
                f"Expected control target shape ({self.action_size},), got {tuple(target.shape)}"
            )
        if not torch.isfinite(target).all():
            raise ValueError("Control target contains non-finite values")
        if direct_ctrl is not None or direct_ctrl_mask is not None:
            if direct_ctrl is None or direct_ctrl_mask is None:
                raise ValueError("direct_ctrl and direct_ctrl_mask must be provided together")
            direct_ctrl = torch.as_tensor(direct_ctrl, dtype=self.dtype, device=self.device)
            direct_ctrl_mask = torch.as_tensor(
                direct_ctrl_mask, dtype=torch.bool, device=self.device
            )
            if direct_ctrl.shape != target.shape or direct_ctrl_mask.shape != target.shape:
                raise ValueError("Direct control sink must match the target shape")
            if not torch.isfinite(direct_ctrl[direct_ctrl_mask]).all():
                raise ValueError("Direct control contains non-finite values")
        target_types = self._normalize_target_types(target_type)
        if self.actuator_mode == "bam" and any(item != "position" for item in target_types):
            raise ValueError("BAM backend cannot apply velocity targets")
        for _ in range(self.decimation):
            self._step_once(
                target,
                target_type=target_types,
                direct_ctrl=direct_ctrl,
                direct_ctrl_mask=direct_ctrl_mask,
            )
            callback = substep_callback or self._substep_callback
            if callback is not None:
                callback()
        self.state.step_count += 1
        return self.data

    def _step_once(
        self,
        target: torch.Tensor,
        *,
        target_type: str | tuple[str, ...],
        direct_ctrl: torch.Tensor | None = None,
        direct_ctrl_mask: torch.Tensor | None = None,
    ) -> Any:
        """Advance exactly one MuJoCo substep without invoking callbacks."""

        data = self.data
        if data is None:
            raise RuntimeError("Physics backend has no data during step")
        delayed_target = self._apply_actuator_delay(target)
        ctrl = (
            self.compute_control(delayed_target)
            if self.actuator_mode == "bam"
            and all(item == "position" for item in self._normalize_target_types(target_type))
            else delayed_target
        )
        if direct_ctrl is not None and direct_ctrl_mask is not None:
            ctrl = torch.where(direct_ctrl_mask, direct_ctrl, ctrl)
        data = data.replace(ctrl=ctrl)
        data = mujoco_torch.step(
            self.bundle.torch_model,
            data,
            fixed_iterations=self.bundle.fixed_iterations,
        )
        self.data = data
        if self.actuator_mode == "bam":
            if self._bam_runtime is None:
                raise RuntimeError("BAM state is not initialized")
            self._bam_runtime.previous_torque = data.qfrc_actuator.index_select(
                -1, self.bundle.qvel_indices
            ).clone()
            self.state.previous_torque = self._bam_runtime.previous_torque.clone()
        return self.data

    def _normalize_target_types(self, target_type: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(target_type, str):
            result = (target_type,) * self.action_size
        else:
            result = tuple(target_type)
            if len(result) != self.action_size:
                raise ValueError(
                    f"Expected one target type per actuator ({self.action_size}), got {len(result)}"
                )
        invalid = set(result) - {"position", "velocity", "effort"}
        if invalid:
            raise ValueError(f"Unsupported action target types {sorted(invalid)!r}")
        return result

    def snapshot(self) -> dict[str, Any]:
        if self.data is None:
            raise RuntimeError("Reset the physics backend before taking a snapshot")
        return {
            "qpos": self.data.qpos.detach().clone(),
            "qvel": self.data.qvel.detach().clone(),
            "qacc": self.data.qacc.detach().clone(),
            "ctrl": self.data.ctrl.detach().clone(),
            "sensordata": self.data.sensordata.detach().clone(),
            "time": float(self.data.time),
        }


class BatchedPhysicsBackend:
    """Batch contract over independent scalar ``PhysicsBackend`` instances.

    The installed mujoco-torch release batches ``Data`` but deliberately keeps
    model metadata unbatched. Independent instances are therefore the only
    correct way to support per-environment friction, mass, actuator, and BAM
    mutations today. The wrapper exposes a leading environment dimension to
    managers while keeping the proven scalar solver path for each instance.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        num_envs: int,
        actuator_mode: str | None = None,
        decimation: int | None = None,
    ) -> None:
        if num_envs < 2:
            raise ValueError("BatchedPhysicsBackend requires at least two environments")
        self.bundle = bundle
        self.num_envs = num_envs
        self._instances = tuple(
            PhysicsBackend(
                clone_model_bundle(bundle),
                actuator_mode=actuator_mode,
                decimation=decimation,
            )
            for _ in range(num_envs)
        )
        self._substep_callback: Any | None = None

    @property
    def instances(self) -> tuple[PhysicsBackend, ...]:
        return self._instances

    @property
    def _generator(self) -> torch.Generator:
        """Expose a deterministic batch generator to manager term code.

        Each scalar child owns its own stream.  The first stream is exposed for
        APIs that only accept one generator; vectorized sampling methods below
        deliberately use every child stream so randomization cannot leak across
        environments.
        """

        generator = self._instances[0]._generator
        if generator is None:  # pragma: no cover - initialized in constructor
            generator = torch.Generator(device=self.device)
            self._instances[0]._generator = generator
        return generator

    @property
    def generators(self) -> tuple[torch.Generator, ...]:
        """Independent RNG streams, one for each environment instance."""

        result: list[torch.Generator] = []
        for instance in self._instances:
            if instance._generator is None:
                instance._generator = torch.Generator(device=self.device)
            result.append(instance._generator)
        return tuple(result)

    @property
    def _bam_vin(self) -> Any:
        values = [instance._bam_vin for instance in self._instances]
        return self._stack_optional(values)

    @_bam_vin.setter
    def _bam_vin(self, value: Any) -> None:
        for index, instance in enumerate(self._instances):
            instance._bam_vin = self._value_for_env(value, index)

    @property
    def _bam_drop_gain(self) -> Any:
        values = [instance._bam_drop_gain for instance in self._instances]
        return self._stack_optional(values)

    @_bam_drop_gain.setter
    def _bam_drop_gain(self, value: Any) -> None:
        for index, instance in enumerate(self._instances):
            instance._bam_drop_gain = self._value_for_env(value, index)

    @property
    def _bam_friction_scale(self) -> Any:
        values = [instance._bam_friction_scale for instance in self._instances]
        return self._stack_optional(values)

    @staticmethod
    def _stack_optional(values: list[Any]) -> Any:
        """Expose per-row BAM state without collapsing the batch to row zero."""

        if any(value is None for value in values):
            return None
        if all(isinstance(value, torch.Tensor) for value in values):
            return torch.stack(values)
        return torch.as_tensor(values)

    @_bam_friction_scale.setter
    def _bam_friction_scale(self, value: Any) -> None:
        for index, instance in enumerate(self._instances):
            instance._bam_friction_scale = self._value_for_env(value, index)

    @property
    def _bam_kp_scale(self) -> Any:
        return self._stack_optional([instance._bam_kp_scale for instance in self._instances])

    @_bam_kp_scale.setter
    def _bam_kp_scale(self, value: Any) -> None:
        for index, instance in enumerate(self._instances):
            instance._bam_kp_scale = self._value_for_env(value, index)

    @property
    def _bam_kd_scale(self) -> Any:
        return self._stack_optional([instance._bam_kd_scale for instance in self._instances])

    @_bam_kd_scale.setter
    def _bam_kd_scale(self, value: Any) -> None:
        for index, instance in enumerate(self._instances):
            instance._bam_kd_scale = self._value_for_env(value, index)

    @property
    def data(self) -> Any | None:
        rows = [instance.data for instance in self._instances]
        if any(row is None for row in rows):
            return None
        data_rows = [row for row in rows if row is not None]
        # ``Data`` marks ``ncon``/``nefc`` as unbatched metadata and its
        # nested ``Contact`` as a TensorClass.  The generic TensorClass stack
        # consequently retains row zero for those fields and emits a warning.
        # Stack once, then replace every shape-invariant/contact field with an
        # explicitly batched value before exposing the result to managers.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Stacking UnbatchedTensors with different data storage.*",
            )
            data: Any = torch.stack(data_rows)
        contacts = [row.contact for row in data_rows]
        contact_keys = tuple(contacts[0]._tensordict.keys())
        contact = type(contacts[0])(
            *(torch.stack([getattr(value, key) for value in contacts]) for key in contact_keys),
            batch_size=torch.Size([self.num_envs]),
            device=self.device,
        )
        # TensorClass marks these counts as shape-invariant and therefore
        # ``torch.stack`` keeps only row zero. Restore the true per-row values
        # immediately so contact/constraint managers never inspect a sibling's
        # count after a partial reset or heterogeneous terrain contact.
        counts = {
            name: torch.tensor(
                [int(getattr(row, name).item()) for row in data_rows],
                dtype=torch.int32,
                device=self.device,
            )
            for name in ("ncon", "nefc")
        }
        return data.replace(contact=contact, **counts)

    @data.setter
    def data(self, value: Any | None) -> None:
        if value is None:
            for instance in self._instances:
                instance.data = None
            return
        if value.batch_size != torch.Size([self.num_envs]):
            raise ValueError(f"Expected batched Data with batch size {self.num_envs}")
        for index, instance in enumerate(self._instances):
            instance.data = value[index]

    @property
    def state(self) -> PhysicsState:
        # The wrapper's state is represented by child states. This property is
        # only retained for code that reads ``physics.state`` diagnostics.
        return self._instances[0].state

    @property
    def step_count(self) -> int:
        # A vector reset can intentionally restart one episode while siblings
        # continue.  The scalar property remains a diagnostic summary;
        # per-environment consumers should use ``step_counts``.
        return max(instance.step_count for instance in self._instances)

    @property
    def step_counts(self) -> torch.Tensor:
        return torch.tensor(
            [instance.step_count for instance in self._instances],
            dtype=torch.long,
            device=self.device,
        )

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @property
    def dtype(self) -> torch.dtype:
        return self.bundle.dtype

    @property
    def timestep(self) -> float:
        return self.bundle.timestep

    @property
    def decimation(self) -> int:
        return self._instances[0].decimation

    @property
    def actuator_mode(self) -> str:
        return self._instances[0].actuator_mode

    @step_count.setter
    def step_count(self, value: int) -> None:
        for instance in self._instances:
            instance.step_count = value

    @property
    def action_size(self) -> int:
        return self.bundle.action_size

    def set_seed(self, seed: int | None, env_ids: torch.Tensor | slice | None = None) -> None:
        if seed is None:
            return
        ids = range(self.num_envs) if env_ids is None else self._ids(env_ids).tolist()
        for index in ids:
            instance = self._instances[index]
            instance.set_seed(seed + index)

    def random(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.stack([instance.random(dtype=dtype) for instance in self._instances])

    def random_tensor(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        normal: bool = False,
    ) -> torch.Tensor:
        """Sample one independent tensor per environment row."""

        row_shape = shape[1:] if shape and shape[0] == self.num_envs else shape
        return torch.stack(
            [
                instance.random_tensor(row_shape, dtype=dtype, normal=normal)
                for instance in self._instances
            ]
        )

    def sample_range(
        self,
        low: float,
        high: float,
        env_ids: torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else self._ids(env_ids)
        )
        return torch.stack(
            [self._instances[int(index)].sample_range(low, high) for index in ids.tolist()]
        )

    def sample_delay(
        self,
        low: int,
        high: int,
        env_ids: torch.Tensor | slice | None = None,
        *,
        per_env: bool = True,
    ) -> int | torch.Tensor:
        if not per_env:
            return self._instances[0].sample_delay(low, high)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else self._ids(env_ids)
        )
        return torch.tensor(
            [self._instances[int(index)].sample_delay(low, high) for index in ids.tolist()],
            dtype=torch.long,
            device=self.device,
        )

    def set_substep_callback(self, callback: Any | None) -> None:
        self._substep_callback = callback
        # Child callbacks are intentionally disabled. Calling a batched sensor
        # callback after only one child has advanced would expose mixed-time
        # state. The wrapper calls it after the complete decimated batch.
        for instance in self._instances:
            instance.set_substep_callback(None)

    def restore_model_defaults(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else self._ids(env_ids)
        )
        for index in ids.tolist():
            self._instances[index].restore_model_defaults()

    def base_field(self, name: str) -> Any:
        values = [instance.base_field(name) for instance in self._instances]
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values)
        return values

    def configure_bam(
        self,
        *,
        vin: Any = None,
        drop_gain: Any = None,
        friction_scale: Any = 1.0,
        kp_scale: Any = 1.0,
        kd_scale: Any = 1.0,
    ) -> None:
        for index, instance in enumerate(self._instances):
            instance.configure_bam(
                vin=self._value_for_env(vin, index),
                drop_gain=self._value_for_env(drop_gain, index),
                friction_scale=self._value_for_env(friction_scale, index),
                kp_scale=self._value_for_env(kp_scale, index),
                kd_scale=self._value_for_env(kd_scale, index),
            )

    def new_data(self) -> Any:
        return torch.stack([instance.new_data() for instance in self._instances])

    def reset(
        self,
        *,
        qpos: torch.Tensor | None = None,
        qvel: torch.Tensor | None = None,
        ctrl: torch.Tensor | None = None,
        seed: int | None = None,
        env_ids: torch.Tensor | slice | None = None,
        mocap_pos: torch.Tensor | None = None,
        mocap_quat: torch.Tensor | None = None,
        xfrc_applied: torch.Tensor | None = None,
    ) -> Any:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else self._ids(env_ids)
        )
        for index in ids.tolist():
            instance = self._instances[index]
            instance.reset(
                qpos=self._row(qpos, index),
                qvel=self._row(qvel, index),
                ctrl=self._row(ctrl, index),
                seed=None if seed is None else seed + index,
                mocap_pos=self._row(mocap_pos, index),
                mocap_quat=self._row(mocap_quat, index),
                xfrc_applied=self._row(xfrc_applied, index),
            )
        return self.data

    def forward(
        self,
        *,
        qpos: torch.Tensor | None = None,
        qvel: torch.Tensor | None = None,
        ctrl: torch.Tensor | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> Any:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else self._ids(env_ids)
        )
        for index in ids.tolist():
            self._instances[index].forward(
                qpos=self._row(qpos, index),
                qvel=self._row(qvel, index),
                ctrl=self._row(ctrl, index),
            )
        return self.data

    def actuator_measurements(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions, velocities = zip(
            *(instance.actuator_measurements() for instance in self._instances), strict=True
        )
        return torch.stack(positions), torch.stack(velocities)

    def encoder_velocity(self) -> torch.Tensor:
        return torch.stack([instance.encoder_velocity() for instance in self._instances])

    def step(
        self,
        target: torch.Tensor,
        *,
        target_type: str | tuple[str, ...] = "position",
        direct_ctrl: torch.Tensor | None = None,
        direct_ctrl_mask: torch.Tensor | None = None,
        substep_callback: Any | None = None,
    ) -> Any:
        target = torch.as_tensor(target, dtype=self.dtype, device=self.device)
        if target.shape != (self.num_envs, self.action_size):
            raise ValueError(
                f"Expected control target shape ({self.num_envs}, {self.action_size}), "
                f"got {tuple(target.shape)}"
            )
        if not torch.isfinite(target).all():
            raise ValueError("Control target contains non-finite values")
        if direct_ctrl is not None or direct_ctrl_mask is not None:
            if direct_ctrl is None or direct_ctrl_mask is None:
                raise ValueError("direct_ctrl and direct_ctrl_mask must be provided together")
            direct_ctrl = torch.as_tensor(direct_ctrl, dtype=self.dtype, device=self.device)
            direct_ctrl_mask = torch.as_tensor(
                direct_ctrl_mask, dtype=torch.bool, device=self.device
            )
            expected = (self.num_envs, self.action_size)
            if direct_ctrl.shape != expected or direct_ctrl_mask.shape != expected:
                raise ValueError(f"Direct control sink must have shape {expected}")
            if not torch.isfinite(direct_ctrl[direct_ctrl_mask]).all():
                raise ValueError("Direct control contains non-finite values")
        target_types = self._normalize_target_types(target_type)
        if self.actuator_mode == "bam" and any(item != "position" for item in target_types):
            raise ValueError("BAM backend cannot apply velocity targets")
        for _ in range(self.decimation):
            for index, instance in enumerate(self._instances):
                instance._step_once(
                    target[index],
                    target_type=target_types,
                    direct_ctrl=direct_ctrl[index] if direct_ctrl is not None else None,
                    direct_ctrl_mask=(
                        direct_ctrl_mask[index] if direct_ctrl_mask is not None else None
                    ),
                )
            callback = substep_callback or self._substep_callback
            if callback is not None:
                callback()
        for instance in self._instances:
            instance.state.step_count += 1
        return self.data

    def _normalize_target_types(self, target_type: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(target_type, str):
            result = (target_type,) * self.action_size
        else:
            result = tuple(target_type)
            if len(result) != self.action_size:
                raise ValueError(
                    f"Expected one target type per actuator ({self.action_size}), got {len(result)}"
                )
        invalid = set(result) - {"position", "velocity", "effort"}
        if invalid:
            raise ValueError(f"Unsupported action target types {sorted(invalid)!r}")
        return result

    def snapshot(self) -> dict[str, Any]:
        snapshots = [instance.snapshot() for instance in self._instances]
        return {
            name: (
                torch.stack([snapshot[name] for snapshot in snapshots])
                if isinstance(snapshots[0][name], torch.Tensor)
                else [snapshot[name] for snapshot in snapshots]
            )
            for name in snapshots[0]
        }

    def _value_for_env(self, value: Any, index: int) -> Any:
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return value[index]
        return value

    def _ids(self, env_ids: torch.Tensor | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)

    @staticmethod
    def _row(value: torch.Tensor | None, index: int) -> torch.Tensor | None:
        if value is None:
            return None
        value = torch.as_tensor(value)
        return value[index] if value.ndim > 1 else value


__all__ = ["BatchedPhysicsBackend", "PhysicsBackend", "PhysicsState"]
