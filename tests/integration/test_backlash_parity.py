from pathlib import Path

import numpy as np
import pytest
import torch

from microduck_rl_torch.envs import ManagerBasedTaskEnv
from microduck_rl_torch.envs.model import load_microduck_model
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg
from microduck_rl_torch_verification.native import NativeMicroDuckEnv


@pytest.mark.integration
def test_backlash_encoder_and_bam_control_match_native():
    scene = Path(__file__).parents[2] / "assets/robot/microduck/scene_walk_backlash.xml"
    bundle = load_microduck_model(
        scene,
        dtype=torch.float64,
        fixed_iterations=True,
        solver_iterations=30,
        line_search_iterations=30,
        disable_contacts=True,
    )
    assert bundle.has_backlash
    torch_env = ManagerBasedTaskEnv(make_microduck_velocity_env_cfg(), bundle=bundle)
    native_env = NativeMicroDuckEnv(
        bundle=bundle,
        disable_contacts=True,
        solver_iterations=30,
        line_search_iterations=30,
    )
    torch_observation = torch_env.reset()
    native_observation = native_env.reset()
    np.testing.assert_allclose(torch_observation.numpy(), native_observation, rtol=0.0, atol=1e-6)
    for action in (np.zeros(14), np.full(14, 0.02), np.full(14, -0.015)):
        torch_result = torch_env.step(torch.from_numpy(action).to(dtype=bundle.dtype))
        native_observation = native_env.step(action)
        torch_data = torch_env.data
        assert torch_data is not None
        np.testing.assert_allclose(
            torch_result.observation.numpy(), native_observation, rtol=0.0, atol=2e-6
        )
        np.testing.assert_allclose(
            torch_data.qpos.numpy(), native_env.data.qpos, rtol=0.0, atol=2e-8
        )
        np.testing.assert_allclose(
            torch_data.qvel.numpy(), native_env.data.qvel, rtol=0.0, atol=2e-6
        )
