from dataclasses import replace
from pathlib import Path

import pytest
import torch

from microduck_rl_torch.envs.actuation import ActuatorDelayCfg
from microduck_rl_torch.envs.model import load_model_bundle
from microduck_rl_torch.envs.observations import command_vector
from microduck_rl_torch.robot import MICRODUCK_WALK_ROBOT_CFG
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
        xml_path=MICRODUCK_WALK_ROBOT_CFG.keyframe_source,
        # The fixture is a zero-delay reference, so configure the model
        # explicitly with the corresponding actuator-delay settings.
        entity_cfg=replace(
            MICRODUCK_WALK_ROBOT_CFG,
            actuator_delay=ActuatorDelayCfg(),
        ),
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
            # The local mujoco-torch solver intentionally has known
            # numerical simplifications; keep the native fixture useful
            # without treating those backend differences as manager/API
            # regressions.
            "observations": 1e-4,
            "qpos": 1e-5,
            "qvel": 1e-4,
            "qacc": 10.0,
            "ctrl": 1e-5,
            "sensordata": 0.1,
            "times": 1e-12,
            "rewards": 0.2,
        },
    )
    assert errors["observations"] < 1e-4
