from pathlib import Path

import numpy as np
import pytest
import torch

from microduck_rl_torch.envs import ManagerBasedTaskEnv
from microduck_rl_torch.envs.model import load_model_bundle
from microduck_rl_torch.robot import MICRODUCK_WALK_BACKLASH_ROBOT_CFG
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg
from microduck_rl_torch.tasks.backlash import make_backlash_variant
from microduck_rl_torch_verification.native import NativeMicroDuckEnv


@pytest.mark.integration
def test_backlash_encoder_and_bam_control_match_native():
    scene = Path(__file__).parents[2] / "assets/robot/microduck/scene_walk_backlash.xml"
    cfg = make_backlash_variant(
        make_microduck_velocity_env_cfg(), MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    )
    bundle = load_model_bundle(
        scene,
        entity_cfg=MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
        entities=cfg.scene.entities,
        dtype=torch.float64,
        fixed_iterations=True,
        solver_iterations=30,
        line_search_iterations=30,
        disable_contacts=True,
    )
    assert bundle.has_backlash
    # The native diagnostic intentionally starts with a zero command. The
    # manager default now samples configured commands even with DR disabled,
    # matching upstream command-manager semantics, so pin this parity fixture.
    torch_env = ManagerBasedTaskEnv(
        cfg,
        bundle=bundle,
        command=torch.zeros(13, dtype=bundle.dtype),
    )
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
