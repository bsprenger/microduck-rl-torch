"""MicroDuck's upstream BAM M6 actuator equations.

The upstream task uses the ``bam`` M6 model rather than MuJoCo's XML position
actuator.  Keeping the equations here makes the policy-facing environment
independent of the optional upstream training stack while preserving the
native and ``mujoco-torch`` control semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

XL330_ERROR_GAIN = (4096.0 / (2.0 * np.pi)) / (256.0 * 885.0)


@dataclass(frozen=True)
class BamM6Parameters:
    """Calibrated BAM M6 XL330 parameters used by upstream MicroDuck."""

    kt: float = 0.36601349688984386
    resistance: float = 2.8113923539223227
    armature: float = 0.0018077432831600838
    q_offset: float = 0.0271132870444849
    friction_base: float = 0.004771183165566
    friction_stribeck: float = 0.004676345799486616
    load_friction_motor: float = 0.2667860954283698
    load_friction_external: float = 8.515871897059342e-06
    load_friction_motor_stribeck: float = 1.072291839509912e-05
    load_friction_external_stribeck: float = 0.08077928978935671
    load_friction_motor_quad: float = 0.009972471242139415
    load_friction_external_quad: float = 0.004902565732332559
    dtheta_stribeck: float = 2.890372094130307
    alpha: float = 8.683259907618984
    friction_viscous: float = 0.005359668274599504
    kp_fw: float = 200.0
    vin: float = 7.4
    vin_drop_gain: float | None = 0.1
    vin_min: float | None = 6.0
    max_current: float | None = None

    def force_limit(self, vin: float | None = None) -> float:
        """Maximum motor torque represented by the compiled motor actuator."""

        return (self.vin if vin is None else vin) * self.kt / self.resistance


@dataclass
class BamRuntime:
    """Mutable per-rollout state required by the BAM controller."""

    previous_torque: Any
    friction_scale: Any


def _clamp(value: Any, low: Any, high: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.clamp(value, min=low, max=high)
    return np.clip(value, low, high)


def motor_torque(
    target: Any,
    position: Any,
    velocity: Any,
    *,
    params: BamM6Parameters,
    vin: Any | None = None,
    kp_fw: Any | None = None,
    max_current: float | None = None,
) -> Any:
    """Evaluate the XL330 firmware P loop and DC motor equation."""

    vin_value = params.vin if vin is None else vin
    kp_value = params.kp_fw if kp_fw is None else kp_fw
    duty = (target - position) * kp_value * XL330_ERROR_GAIN
    current_limit = params.max_current if max_current is None else max_current
    if current_limit is not None:
        back_emf = params.kt * velocity
        duty_span = params.resistance * current_limit / vin_value
        duty = _clamp(duty, back_emf / vin_value - duty_span, back_emf / vin_value + duty_span)
    duty = _clamp(duty, -1.0, 1.0)
    voltage = vin_value * duty
    return params.kt * voltage / params.resistance - params.kt**2 * velocity / params.resistance


def friction_budget(
    motor: Any,
    external: Any,
    velocity: Any,
    *,
    params: BamM6Parameters,
    friction_scale: Any = 1.0,
) -> tuple[Any, Any]:
    """Return the BAM M6 ``dof_frictionloss`` and ``dof_damping`` values."""

    if isinstance(motor, torch.Tensor):
        stribeck = torch.exp(-torch.pow(torch.abs(velocity) / params.dtheta_stribeck, params.alpha))
        gearbox = torch.abs(
            external * params.load_friction_external - motor * params.load_friction_motor
        )
        gearbox_stribeck = torch.abs(
            external * params.load_friction_external_stribeck
            - motor * params.load_friction_motor_stribeck
        )
        drive = torch.abs(motor) > torch.abs(external)
        quadratic = torch.where(
            drive,
            params.load_friction_external_quad * torch.abs(external) ** 2,
            params.load_friction_motor_quad * torch.abs(motor) ** 2,
        )
        opposing = torch.sign(external) != torch.sign(motor)
        friction = (
            params.friction_base
            + stribeck * params.friction_stribeck
            + gearbox
            + stribeck * gearbox_stribeck
            + stribeck * torch.where(opposing, quadratic, torch.zeros_like(quadratic))
        )
    else:
        stribeck = np.exp(-np.power(np.abs(velocity) / params.dtheta_stribeck, params.alpha))
        gearbox = np.abs(
            external * params.load_friction_external - motor * params.load_friction_motor
        )
        gearbox_stribeck = np.abs(
            external * params.load_friction_external_stribeck
            - motor * params.load_friction_motor_stribeck
        )
        drive = np.abs(motor) > np.abs(external)
        quadratic = np.where(
            drive,
            params.load_friction_external_quad * np.abs(external) ** 2,
            params.load_friction_motor_quad * np.abs(motor) ** 2,
        )
        opposing = np.sign(external) != np.sign(motor)
        friction = (
            params.friction_base
            + stribeck * params.friction_stribeck
            + gearbox
            + stribeck * gearbox_stribeck
            + stribeck * np.where(opposing, quadratic, 0.0)
        )
    return friction * friction_scale, params.friction_viscous * friction_scale


def external_torque(data: Any, dof_indices: torch.Tensor, friction_rows: int) -> torch.Tensor:
    """Compute BAM's load torque, excluding injected DOF-friction rows.

    ``mujoco-torch`` stores friction constraints first in the EFC arrays.  The
    first ``friction_rows`` rows therefore give the exact equivalent of
    MuJoCo's ``efc_id``/``efc_type`` selection without adding a backend-specific
    field to the policy environment.
    """

    qfrc_bias = data.qfrc_bias[..., dof_indices]
    qfrc_constraint = data.qfrc_constraint[..., dof_indices]
    if friction_rows <= 0:
        return -qfrc_bias + qfrc_constraint
    friction_force = (
        data.efc_J[..., :friction_rows, :].transpose(-1, -2) @ data.efc_force[..., :friction_rows]
    )
    return -qfrc_bias + qfrc_constraint - friction_force[..., dof_indices]


def apply_bam_fields(
    model: Any, dof_indices: torch.Tensor, frictionloss: torch.Tensor, damping: torch.Tensor
) -> None:
    """Write scalar BAM friction fields into a device model."""

    model.dof_frictionloss[dof_indices] = frictionloss
    model.dof_damping[dof_indices] = damping
