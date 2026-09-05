"""Generic MuJoCo-Torch physics backend.

This module owns simulation mechanics only.  It deliberately has no knowledge
of commands, observations, rewards, terminations, or task curricula.  The
manager-based environment composes it and uses the model bundle's resolved
actuator mapping to express task-specific behavior above this boundary.
"""

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
from .model import ModelBundle

mujoco_api: Any = mujoco


@dataclass
class PhysicsState:
    """Backend-owned state that is independent of any task semantics."""

    data: Any | None = None
    step_count: int = 0
    previous_torque: torch.Tensor | None = None


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
        actuator_delay_lag: int | tuple[int, int] = 0,
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
        self._base_fields: dict[str, Any] | None = None
        if isinstance(actuator_delay_lag, tuple):
            low, high = actuator_delay_lag
            if low < 0 or low > high:
                raise ValueError("actuator delay range must satisfy 0 <= low <= high")
            self.actuator_delay_range = (low, high)
        else:
            if actuator_delay_lag < 0:
                raise ValueError("actuator_delay_lag must be non-negative")
            self.actuator_delay_range = (actuator_delay_lag, actuator_delay_lag)
        self.actuator_delay_lag = 0
        self._actuator_delay_buffer: list[torch.Tensor] = []

    @property
    def data(self) -> Any | None:
        return self.state.data

    @data.setter
    def data(self, value: Any | None) -> None:
        self.state.data = value

    @property
    def step_count(self) -> int:
        return self.state.step_count

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

    def random(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Sample a scalar on the backend device using the backend RNG."""

        return torch.rand(
            (),
            generator=self._generator,
            dtype=dtype or self.dtype,
            device=self.device,
        )

    def sample_range(self, low: float, high: float) -> torch.Tensor:
        return self.random() * (high - low) + low

    def sample_delay(self, low: int, high: int) -> int:
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

    def set_seed(self, seed: int | None) -> None:
        if seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(seed)

    def new_data(self) -> Any:
        """Allocate and forward fresh device-resident simulation data."""

        return self.bundle.new_data()

    def restore_model_defaults(self) -> None:
        """Restore mutable model fields before a new mutation sample."""

        if self._base_fields is None:
            self._base_fields = {
                "dof_armature": self.bundle.torch_model.dof_armature.clone(),
                "dof_frictionloss": self.bundle.torch_model.dof_frictionloss.clone(),
                "dof_damping": self.bundle.torch_model.dof_damping.clone(),
                "body_mass": self.bundle.torch_model.body_mass.clone(),
                "body_inertia": self.bundle.torch_model.body_inertia.clone(),
                "body_ipos": self.bundle.torch_model.body_ipos.clone(),
                "geom_friction": self.bundle.torch_model.geom_friction.clone(),
                "native_dof_armature": self.bundle.native_model.dof_armature.copy(),
                "native_dof_frictionloss": self.bundle.native_model.dof_frictionloss.copy(),
                "native_dof_damping": self.bundle.native_model.dof_damping.copy(),
                "native_body_mass": self.bundle.native_model.body_mass.copy(),
                "native_body_inertia": self.bundle.native_model.body_inertia.copy(),
                "native_body_ipos": self.bundle.native_model.body_ipos.copy(),
                "native_geom_friction": self.bundle.native_model.geom_friction.copy(),
            }
        fields = self._base_fields
        self.bundle.torch_model.dof_armature[:] = fields["dof_armature"]
        self.bundle.torch_model.dof_frictionloss[:] = fields["dof_frictionloss"]
        self.bundle.torch_model.dof_damping[:] = fields["dof_damping"]
        self.bundle.torch_model.body_mass[:] = fields["body_mass"]
        self.bundle.torch_model.body_inertia[:] = fields["body_inertia"]
        self.bundle.torch_model.body_ipos[:] = fields["body_ipos"]
        self.bundle.torch_model.geom_friction[:] = fields["geom_friction"]
        self.bundle.native_model.dof_armature[:] = fields["native_dof_armature"]
        self.bundle.native_model.dof_frictionloss[:] = fields["native_dof_frictionloss"]
        self.bundle.native_model.dof_damping[:] = fields["native_dof_damping"]
        self.bundle.native_model.body_mass[:] = fields["native_body_mass"]
        self.bundle.native_model.body_inertia[:] = fields["native_body_inertia"]
        self.bundle.native_model.body_ipos[:] = fields["native_body_ipos"]
        self.bundle.native_model.geom_friction[:] = fields["native_geom_friction"]
        mujoco_api.mj_setConst(
            self.bundle.native_model, mujoco_api.MjData(self.bundle.native_model)
        )

    def base_field(self, name: str) -> Any:
        """Return a read-only reference to a captured mutable model field."""

        if self._base_fields is None:
            self.restore_model_defaults()
        assert self._base_fields is not None
        try:
            return self._base_fields[name]
        except KeyError as exc:
            raise KeyError(f"Unknown captured model field {name!r}") from exc

    def configure_bam(
        self,
        *,
        vin: torch.Tensor | None = None,
        drop_gain: torch.Tensor | float | None = None,
        friction_scale: torch.Tensor | float = 1.0,
    ) -> None:
        """Configure per-rollout BAM parameters without coupling to a task."""

        self._bam_vin = vin
        self._bam_drop_gain = drop_gain
        self._bam_friction_scale = friction_scale

    def reset(
        self,
        *,
        qpos: torch.Tensor | None = None,
        qvel: torch.Tensor | None = None,
        ctrl: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> Any:
        """Reset only the physical state; task reset events run above this layer."""

        self.set_seed(seed)
        data = self.new_data()
        qpos_value = data.qpos.clone() if qpos is None else qpos.to(self.device, self.dtype)
        qvel_value = (
            torch.zeros_like(data.qvel) if qvel is None else qvel.to(self.device, self.dtype)
        )
        if ctrl is None:
            ctrl_value = (
                torch.zeros_like(data.ctrl)
                if self.actuator_mode == "bam"
                else self.bundle.default_pose.clone()
            )
        else:
            ctrl_value = ctrl.to(self.device, self.dtype)
        self.data = mujoco_torch.forward(
            self.bundle.torch_model,
            data.replace(qpos=qpos_value, qvel=qvel_value, ctrl=ctrl_value),
            fixed_iterations=self.bundle.fixed_iterations,
        )
        self.state.step_count = 0
        self.state.previous_torque = torch.zeros(
            self.action_size, dtype=self.dtype, device=self.device
        )
        self._bam_runtime = BamRuntime(
            previous_torque=self.state.previous_torque.clone(),
            friction_scale=self._bam_friction_scale,
        )
        self.actuator_delay_lag = self.sample_delay(*self.actuator_delay_range)
        self._actuator_delay_buffer = [
            self.bundle.default_pose.clone() for _ in range(self.actuator_delay_lag)
        ]
        return self.data

    def forward(
        self,
        *,
        qpos: torch.Tensor | None = None,
        qvel: torch.Tensor | None = None,
        ctrl: torch.Tensor | None = None,
    ) -> Any:
        """Forward the current state after an external state mutation."""

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
        position = self.data.qpos.index_select(-1, self.bundle.qpos_indices)
        velocity = self.data.qvel.index_select(-1, self.bundle.qvel_indices)
        if self.bundle.has_backlash:
            backlash_position = self.data.qpos.index_select(-1, self.bundle.backlash_qpos_indices)
            position = position + backlash_position * self.bundle.backlash_mask
        return position, velocity

    def encoder_velocity(self) -> torch.Tensor:
        """Return output-side velocity for the resolved actuator mapping."""

        if self.data is None:
            raise RuntimeError("Reset the physics backend before reading measurements")
        velocity = self.data.qvel.index_select(-1, self.bundle.qvel_indices)
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
        torque = motor_torque(target, position, velocity, params=parameters, vin=vin)
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

    def step(self, target: torch.Tensor) -> Any:
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
        data = self.data
        if data is None:
            raise RuntimeError("Physics backend has no data during step")
        for _ in range(self.decimation):
            if self.actuator_delay_lag:
                self._actuator_delay_buffer.append(target.clone())
                delayed_target = self._actuator_delay_buffer.pop(0)
            else:
                delayed_target = target
            ctrl = (
                self.compute_control(delayed_target)
                if self.actuator_mode == "bam"
                else delayed_target
            )
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
        self.state.step_count += 1
        return self.data

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


__all__ = ["PhysicsBackend", "PhysicsState"]
