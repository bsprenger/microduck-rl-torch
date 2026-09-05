from __future__ import annotations

import pytest
import torch

from microduck_rl_torch.envs import (
    EntityView,
    EventTermCfg,
    ManagerBasedTaskEnv,
    PhysicsBackend,
    TermCfg,
    TermCollection,
)
from microduck_rl_torch.envs.managers.events import EventManager
from microduck_rl_torch.envs.model import load_microduck_model
from microduck_rl_torch.envs.observations import command_vector
from microduck_rl_torch.envs.rewards import foot_contact_mask
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg


@pytest.mark.integration
def test_manager_environment_is_deterministic_for_fixed_trace():
    bundle_reference = load_microduck_model(
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    bundle_manager = load_microduck_model(
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    cfg = make_microduck_velocity_env_cfg()
    reference_command = command_vector(
        vx=0.15, device=bundle_reference.device, dtype=bundle_reference.dtype
    )
    manager_command = command_vector(
        vx=0.15, device=bundle_manager.device, dtype=bundle_manager.dtype
    )
    reference = ManagerBasedTaskEnv(cfg.clone(), bundle=bundle_reference, command=reference_command)
    manager = ManagerBasedTaskEnv(cfg, bundle=bundle_manager, command=manager_command)
    assert isinstance(manager.physics, PhysicsBackend)
    assert not hasattr(manager, "runtime")
    reference_obs = reference.reset(seed=17)
    manager_obs = manager.reset(seed=17)
    torch.testing.assert_close(reference_obs, manager_obs)

    for index in range(6):
        action = torch.sin(torch.arange(14, dtype=bundle_manager.dtype) + index) * 0.05
        reference_step = reference.step(action)
        manager_step = manager.step(action)
        torch.testing.assert_close(manager_step.observation, reference_step.observation)
        torch.testing.assert_close(manager_step.reward, reference_step.reward)
        assert manager_step.terminated == reference_step.terminated
        assert manager_step.truncated == reference_step.truncated
        assert manager_step.info["terminations"] == {
            "non_finite": False,
            "bad_orientation": False,
            "timeout": False,
        }


@pytest.mark.integration
def test_manager_environment_owns_lifecycle_order(monkeypatch):
    bundle = load_microduck_model(
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    command = command_vector(vx=0.15, device=bundle.device, dtype=bundle.dtype)
    environment = ManagerBasedTaskEnv(
        make_microduck_velocity_env_cfg(), bundle=bundle, command=command
    )
    environment.reset(seed=17)
    order: list[str] = []

    event_apply = environment.event_manager.apply
    monkeypatch.setattr(
        environment.event_manager,
        "apply",
        lambda env, stage: (order.append(f"event:{stage}"), event_apply(env, stage))[1],
    )
    action_prepare = environment.action_manager.prepare
    monkeypatch.setattr(
        environment.action_manager,
        "prepare",
        lambda env, action: (order.append("action"), action_prepare(env, action))[1],
    )
    physics_step = environment.physics.step
    monkeypatch.setattr(
        environment.physics,
        "step",
        lambda target: (order.append("physics"), physics_step(target))[1],
    )
    reward_compute = environment.reward_manager.compute
    monkeypatch.setattr(
        environment.reward_manager,
        "compute",
        lambda env, **values: (order.append("reward"), reward_compute(env, **values))[1],
    )
    termination_evaluate = environment.termination_manager.evaluate
    monkeypatch.setattr(
        environment.termination_manager,
        "evaluate",
        lambda env, **values: (
            order.append("termination"),
            termination_evaluate(env, **values),
        )[1],
    )
    curriculum_step = environment.curriculum_manager.step
    monkeypatch.setattr(
        environment.curriculum_manager,
        "step",
        lambda env: (order.append("curriculum"), curriculum_step(env))[1],
    )
    command_step = environment.command_manager.step
    monkeypatch.setattr(
        environment.command_manager,
        "step",
        lambda env: (order.append("command"), command_step(env))[1],
    )
    observation_compute = environment.observation_manager.compute
    monkeypatch.setattr(
        environment.observation_manager,
        "compute",
        lambda env, group="actor": (
            order.append("observation"),
            observation_compute(env, group),
        )[1],
    )

    environment.step(torch.zeros(14, dtype=bundle.dtype))

    assert order == [
        "event:pre_physics",
        "action",
        "physics",
        "event:post_physics",
        "reward",
        "termination",
        "curriculum",
        "command",
        "event:step",
        "event:interval",
        "observation",
    ]


@pytest.mark.integration
def test_reset_events_refresh_environment_state_baselines():
    bundle = load_microduck_model(
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    cfg = make_microduck_velocity_env_cfg()

    def mutate_reset(env):
        qpos = env.data.qpos.clone()
        qpos[env.bundle.qpos_indices[0]] += 0.02
        env.physics.forward(qpos=qpos)
        return torch.zeros((), dtype=qpos.dtype, device=qpos.device)

    cfg.events.add("mutate_reset", TermCfg(func=mutate_reset, params={"stage": "reset"}))
    environment = ManagerBasedTaskEnv(cfg, bundle=bundle)
    environment.reset(seed=17)

    assert environment.state is not None
    data = environment.data
    assert data is not None
    sensors = environment.state.sensors
    torch.testing.assert_close(
        sensors.previous_joint_velocity,
        environment.physics.encoder_velocity(),
    )
    torch.testing.assert_close(
        sensors.previous_foot_positions,
        data.site_xpos[list(bundle.foot_site_ids)],
    )
    torch.testing.assert_close(
        sensors.foot_contact,
        foot_contact_mask(data, bundle),
    )


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


@pytest.mark.integration
def test_rough_factory_materializes_a_concrete_scene_and_critic_group():
    environment = ManagerBasedTaskEnv(make_microduck_velocity_env_cfg(rough=True))
    assert environment.scene_build.terrain_kind == "generator"
    assert environment.scene_build.xml_path != environment.task_cfg.scene.scene_xml
    assert environment.bundle.native_model.ngeom > 0
    assert environment.reset(seed=3).shape == (61,)
    assert environment.observation("critic").shape == (64,)
    assert set(environment.observations()) == {"actor", "critic"}


@pytest.mark.integration
def test_multi_entity_bundle_exposes_prop_views_from_compiled_scene():
    from pathlib import Path

    from microduck_rl_torch.robot import MICRODUCK_BALL_CFG, MICRODUCK_STANDUP_ROBOT_CFG

    bundle = load_microduck_model(
        Path("assets/robot/microduck/scene_ball.xml"),
        entity_cfg=MICRODUCK_STANDUP_ROBOT_CFG,
        entities={
            "robot": MICRODUCK_STANDUP_ROBOT_CFG,
            "ball": MICRODUCK_BALL_CFG,
        },
        actuator_mode="xml",
        disable_contacts=True,
    )
    ball = bundle.entity("ball")
    assert isinstance(ball, EntityView)
    assert ball.kind == "prop"
    assert len(ball.body_ids) == 1
    assert len(ball.geom_ids) == 1
    assert tuple(ball.qpos_indices.shape) == (7,)
    assert tuple(ball.qvel_indices.shape) == (6,)


def test_event_manager_supports_upstream_modes_and_manager_owned_schedules():
    class FakeState:
        def __init__(self):
            self.manager_data = {}

    class FakeEnv:
        state = FakeState()
        step_count = 0
        calls: list[str] = []

        @staticmethod
        def _next_interval_step(_interval):
            return 1

    env = FakeEnv()
    events = TermCollection(
        {
            "startup": EventTermCfg(
                func=lambda current: current.calls.append("startup"), mode="startup"
            ),
            "reset": EventTermCfg(func=lambda current: current.calls.append("reset"), mode="reset"),
            "step": EventTermCfg(func=lambda current: current.calls.append("step"), mode="step"),
            "interval": EventTermCfg(
                func=lambda current: current.calls.append("interval"),
                mode="interval",
                interval_range_s=(0.0, 0.0),
            ),
            "post": EventTermCfg(
                func=lambda current: current.calls.append("post"), mode="post_physics"
            ),
        }
    )
    manager = EventManager(events)

    manager.startup(env)
    manager.reset(env)
    manager.apply(env, "pre_physics")
    env.step_count = 1
    manager.apply(env, "step")
    manager.apply(env, "interval")
    manager.apply(env, "post_physics")

    assert env.calls == ["startup", "reset", "step", "interval", "post"]
