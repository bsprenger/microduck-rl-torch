from __future__ import annotations

import pytest
import torch

from microduck_rl_torch.envs import ManagerBasedTaskEnv, NominalMicroDuckEnv
from microduck_rl_torch.envs.model import load_microduck_model
from microduck_rl_torch.envs.observations import command_vector
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg


@pytest.mark.integration
def test_manager_velocity_matches_legacy_runtime_for_fixed_trace():
    bundle = load_microduck_model(
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    cfg = make_microduck_velocity_env_cfg()
    command = command_vector(vx=0.15, device=bundle.device, dtype=bundle.dtype)
    legacy = NominalMicroDuckEnv(bundle, command=command)
    manager = ManagerBasedTaskEnv(cfg, bundle=bundle, command=command)
    legacy_obs = legacy.reset(seed=17)
    manager_obs = manager.reset(seed=17)
    assert torch.equal(legacy_obs, manager_obs)

    for index in range(6):
        action = torch.sin(torch.arange(14, dtype=bundle.dtype) + index) * 0.05
        legacy_step = legacy.step(action)
        manager_step = manager.step(action)
        torch.testing.assert_close(manager_step.observation, legacy_step.observation)
        torch.testing.assert_close(manager_step.reward, legacy_step.reward)
        assert manager_step.terminated == legacy_step.terminated
        assert manager_step.truncated == legacy_step.truncated
        assert manager_step.info["terminations"] == {
            "non_finite": False,
            "bad_orientation": False,
            "timeout": False,
        }


@pytest.mark.integration
def test_semantic_model_selectors_support_roller_and_backlash_assets():
    from microduck_rl_torch.robot import (
        MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
        MICRODUCK_WALK_ROLLERS_ROBOT_CFG,
    )

    roller = load_microduck_model(
        entity_cfg=MICRODUCK_WALK_ROLLERS_ROBOT_CFG,
        actuator_mode="xml",
        disable_contacts=True,
    )
    backlash = load_microduck_model(
        entity_cfg=MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
        actuator_mode="xml",
        disable_contacts=True,
    )
    assert roller.native_model.nq == 25
    assert len(roller.foot_geom_groups[0]) > 1
    assert len(roller.foot_geom_groups[1]) > 1
    assert roller.has_backlash is False
    assert backlash.native_model.nq == 35
    assert backlash.has_backlash is True
