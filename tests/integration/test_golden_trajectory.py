from pathlib import Path

import pytest
import torch

from microduck_rl_torch.envs.model import load_model_bundle
from microduck_rl_torch.envs.observations import command_vector
from microduck_rl_torch_verification.trajectory import (
    GoldenTrajectory,
    compare_trajectory,
    rollout_torch,
)


@pytest.mark.integration
def test_bam_torch_matches_native_golden_trajectory():
    fixture = Path(__file__).parents[1] / "fixtures/microduck_bam_golden.npz"
    expected = GoldenTrajectory.load(fixture)
    bundle = load_model_bundle(
        dtype=torch.float64,
        fixed_iterations=True,
        solver_iterations=int(expected.metadata["solver_iterations"]),
        line_search_iterations=int(expected.metadata["line_search_iterations"]),
        disable_contacts=not bool(expected.metadata["contacts_enabled"]),
        disable_mesh_mesh_contacts=True,
    )
    command = command_vector(
        vx=float(expected.metadata["command"][0]),
        vy=float(expected.metadata["command"][1]),
        vtheta=float(expected.metadata["command"][2]),
        device=bundle.device,
        dtype=bundle.dtype,
    )
    actual = rollout_torch(bundle, expected, command=command)
    errors = compare_trajectory(
        expected,
        actual,
        tolerances={
            "observations": 1e-6,
            "qpos": 1e-12,
            "qvel": 1e-12,
            "qacc": 1e-9,
            "ctrl": 1e-10,
            "sensordata": 1e-5,
            "times": 1e-12,
            "rewards": 1e-6,
        },
    )
    assert max(errors.values()) < 1e-5
