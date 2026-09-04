import numpy as np
import torch

from microduck_rl_torch.envs.actuation import (
    XL330_ERROR_GAIN,
    BamM6Parameters,
    friction_budget,
    motor_torque,
)


def test_bam_motor_equation_matches_numpy_and_torch():
    params = BamM6Parameters()
    target = np.array([0.1, -0.2])
    position = np.array([0.0, -0.1])
    velocity = np.array([0.3, -0.4])
    expected_duty = np.clip((target - position) * params.kp_fw * XL330_ERROR_GAIN, -1.0, 1.0)
    expected = params.kt * params.vin * expected_duty / params.resistance
    expected -= params.kt**2 * velocity / params.resistance
    np.testing.assert_allclose(
        motor_torque(target, position, velocity, params=params), expected, rtol=0.0, atol=1e-14
    )
    torch_result = motor_torque(
        torch.from_numpy(target),
        torch.from_numpy(position),
        torch.from_numpy(velocity),
        params=params,
    )
    np.testing.assert_allclose(torch_result.numpy(), expected, rtol=0.0, atol=1e-14)


def test_bam_friction_budget_scales_all_terms():
    params = BamM6Parameters()
    motor = torch.tensor([0.2, -0.4], dtype=torch.float64)
    external = torch.tensor([0.1, 0.5], dtype=torch.float64)
    velocity = torch.tensor([0.0, 1.0], dtype=torch.float64)
    nominal, damping = friction_budget(motor, external, velocity, params=params, friction_scale=1.0)
    scaled, scaled_damping = friction_budget(
        motor, external, velocity, params=params, friction_scale=1.25
    )
    torch.testing.assert_close(scaled, nominal * 1.25, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(scaled_damping, damping * 1.25, rtol=0.0, atol=1e-14)
