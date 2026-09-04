from pathlib import Path

import numpy as np
import pytest
import torch

from microduck_rl_torch.envs import NominalMicroDuckEnv
from microduck_rl_torch.envs.model import load_microduck_model


@pytest.mark.integration
def test_head_pose_reward_uses_home_relative_command_and_backlash_output():
    scene = Path("assets/robot/microduck/scene_walk_backlash.xml")
    bundle = load_microduck_model(
        scene,
        fixed_iterations=True,
        solver_iterations=30,
        line_search_iterations=30,
        disable_contacts=True,
    )
    command = torch.zeros(13, dtype=bundle.dtype)
    environment = NominalMicroDuckEnv(bundle, command=command)
    environment.reset()
    result = environment.step(torch.zeros(bundle.action_size, dtype=bundle.dtype))
    assert environment.data is not None
    measured = environment.data.qpos.index_select(-1, bundle.qpos_indices[5:9])
    measured = (
        measured
        + environment.data.qpos.index_select(-1, bundle.backlash_qpos_indices[5:9])
        * bundle.backlash_mask[5:9]
    )
    expected = torch.exp(-torch.square((measured - bundle.default_pose[5:9]) / 0.5)).mean()
    np.testing.assert_allclose(
        result.info["reward_terms"]["head_pose_tracking"], float(expected), rtol=0.0, atol=1e-7
    )
