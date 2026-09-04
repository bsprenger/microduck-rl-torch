from pathlib import Path

import numpy as np
import pytest
import torch

from microduck_rl_torch.envs import ManagerBasedTaskEnv
from microduck_rl_torch.envs.model import load_model_bundle
from microduck_rl_torch.robot import MICRODUCK_WALK_BACKLASH_ROBOT_CFG
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg
from microduck_rl_torch.tasks.backlash import make_backlash_variant


@pytest.mark.integration
def test_head_pose_reward_uses_home_relative_command_and_backlash_output():
    scene = Path("assets/robot/microduck/scene_walk_backlash.xml")
    cfg = make_backlash_variant(
        make_microduck_velocity_env_cfg(), MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    )
    bundle = load_model_bundle(
        scene,
        entity_cfg=MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
        entities=cfg.scene.entities,
        fixed_iterations=True,
        solver_iterations=30,
        line_search_iterations=30,
        disable_contacts=True,
    )
    command = torch.zeros(13, dtype=bundle.dtype)
    environment = ManagerBasedTaskEnv(cfg, bundle=bundle, command=command)
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
