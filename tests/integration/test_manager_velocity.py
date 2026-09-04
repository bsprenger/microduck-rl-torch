from __future__ import annotations

import pytest
import torch

from microduck_rl_torch.envs import (
    EntityView,
    EventTermCfg,
    ManagerBasedTaskEnv,
    PhysicsBackend,
    TermCollection,
)
from microduck_rl_torch.envs.managers.events import EventManager
from microduck_rl_torch.envs.model import load_model_bundle
from microduck_rl_torch.envs.observations import command_vector
from microduck_rl_torch.envs.rewards import foot_contact_mask
from microduck_rl_torch.robot import MICRODUCK_WALK_ROBOT_CFG
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg


@pytest.mark.integration
def test_manager_environment_is_deterministic_for_fixed_trace():
    bundle_reference = load_model_bundle(
        entity_cfg=MICRODUCK_WALK_ROBOT_CFG,
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    bundle_manager = load_model_bundle(
        entity_cfg=MICRODUCK_WALK_ROBOT_CFG,
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
    bundle = load_model_bundle(
        entity_cfg=MICRODUCK_WALK_ROBOT_CFG,
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
    action_process = environment.action_manager.process_action
    monkeypatch.setattr(
        environment.action_manager,
        "process_action",
        lambda action: (order.append("action"), action_process(action))[1],
    )
    physics_step = environment.physics.step
    monkeypatch.setattr(
        environment.physics,
        "step",
        lambda target, **kwargs: (order.append("physics"), physics_step(target, **kwargs))[1],
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
    curriculum_compute = environment.curriculum_manager.compute
    monkeypatch.setattr(
        environment.curriculum_manager,
        "compute",
        lambda env, env_ids=None: (
            order.append("curriculum"),
            curriculum_compute(env, env_ids),
        )[1],
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
        lambda env, group="actor", *, update_history=False: (
            order.append("observation"),
            observation_compute(env, group, update_history=update_history),
        )[1],
    )

    environment.step(torch.zeros(14, dtype=bundle.dtype))

    assert order == [
        "event:pre_physics",
        "action",
        "physics",
        "event:post_physics",
        "termination",
        "reward",
        "command",
        "event:step",
        "event:interval",
        "observation",
        "observation",
    ]


def test_event_manager_preserves_global_timers_and_throttles_partial_resets():
    class FakeBundle:
        device = torch.device("cpu")
        timestep = 0.1

    class FakeState:
        def __init__(self):
            self.manager_data = {}

    class FakeEnv:
        bundle = FakeBundle()
        state = FakeState()
        num_envs = 2
        decimation = 1
        step_count = 0
        step_counts = torch.zeros(2, dtype=torch.long)
        reset_calls: list[torch.Tensor | slice | None] = []
        interval_calls = 0

        @staticmethod
        def _next_interval_step(_interval):
            return 2

    def reset_callback(env, env_ids):
        env.reset_calls.append(env_ids)

    def interval_callback(env, _env_ids):
        env.interval_calls += 1

    events = TermCollection(
        {
            "reset": EventTermCfg(
                func=reset_callback,
                mode="reset",
                min_step_count_between_reset=2,
            ),
            "global": EventTermCfg(
                func=interval_callback,
                mode="interval",
                interval_range_s=(0.0, 0.0),
                is_global_time=True,
            ),
        }
    )
    env = FakeEnv()
    manager = EventManager(events)

    manager.reset(env)
    assert len(env.reset_calls) == 1
    first_schedule = env.state.manager_data["event_next_steps"]["global"]
    assert first_schedule == 2

    env.step_count = 1
    env.step_counts[:] = 1
    manager.reset(env, torch.tensor([0]))
    assert len(env.reset_calls) == 1
    assert env.state.manager_data["event_next_steps"]["global"] == first_schedule

    env.step_count = 2
    env.step_counts[:] = 2
    manager.reset(env, torch.tensor([0]))
    assert len(env.reset_calls) == 2
    assert isinstance(env.reset_calls[-1], torch.Tensor)
    torch.testing.assert_close(env.reset_calls[-1], torch.tensor([0]))

    manager.apply(env, "interval")
    assert env.interval_calls == 1
    interval_schedule = env.state.manager_data["event_next_steps"]["global"]
    assert interval_schedule == 4
    env.step_count = 10
    env.step_counts[:] = 10
    manager.reset(env, torch.tensor([1]))
    assert env.state.manager_data["event_next_steps"]["global"] == interval_schedule


@pytest.mark.integration
def test_reset_events_refresh_environment_state_baselines():
    bundle = load_model_bundle(
        entity_cfg=MICRODUCK_WALK_ROBOT_CFG,
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    cfg = make_microduck_velocity_env_cfg()

    def mutate_reset(env, env_ids):
        del env_ids
        qpos = env.data.qpos.clone()
        qpos[env.bundle.qpos_indices[0]] += 0.02
        env.physics.forward(qpos=qpos)
        return torch.zeros((), dtype=qpos.dtype, device=qpos.device)

    cfg.events.add("mutate_reset", EventTermCfg(func=mutate_reset, mode="reset"))
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

    roller = load_model_bundle(
        entity_cfg=MICRODUCK_WALK_ROLLERS_ROBOT_CFG,
        actuator_mode="xml",
        disable_contacts=True,
    )
    backlash = load_model_bundle(
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
    cfg = make_microduck_velocity_env_cfg(rough=True)
    assert cfg.fixed_iterations is True
    assert cfg.solver_iterations == 30
    assert cfg.line_search_iterations == 50
    assert cfg.nconmax == 200
    environment = ManagerBasedTaskEnv(cfg)
    assert environment.scene_build.terrain_kind == "generator"
    assert environment.scene_build.xml_path != environment.task_cfg.scene.scene_xml
    assert environment.bundle.native_model.ngeom > 0
    assert environment.bundle.nconmax == 200
    actor_observation = environment.reset(seed=3)
    assert isinstance(actor_observation, torch.Tensor)
    assert actor_observation.shape == (61,)
    critic_observation = environment.observation("critic")
    assert isinstance(critic_observation, torch.Tensor)
    assert critic_observation.shape == (76,)
    assert set(environment.observations()) == {"actor", "critic"}


def test_velocity_task_preserves_mesh_mesh_contacts_by_default():
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.default_disable_mesh_mesh_contacts() is False


@pytest.mark.integration
def test_multi_entity_bundle_exposes_prop_views_from_compiled_scene():
    from pathlib import Path

    from microduck_rl_torch.robot import MICRODUCK_BALL_CFG, MICRODUCK_STANDUP_ROBOT_CFG

    bundle = load_model_bundle(
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


def test_event_manager_supports_all_modes_and_manager_owned_schedules():
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

    def record(current, env_ids=None):
        del env_ids
        current.calls.append("event")

    def require_startup_env_ids(current, env_ids):
        assert env_ids is None
        current.calls.append("startup")

    env = FakeEnv()
    events = TermCollection(
        {
            "startup": EventTermCfg(func=require_startup_env_ids, mode="startup"),
            "reset": EventTermCfg(
                func=lambda current, env_ids: current.calls.append("reset"), mode="reset"
            ),
            "step": EventTermCfg(
                func=lambda current, _env_ids: current.calls.append("step"), mode="step"
            ),
            "pre": EventTermCfg(
                func=lambda current, _env_ids: current.calls.append("pre"), mode="pre_physics"
            ),
            "interval": EventTermCfg(
                func=lambda current, env_ids: current.calls.append("interval"),
                mode="interval",
                interval_range_s=(0.0, 0.0),
            ),
            "post": EventTermCfg(
                func=lambda current, _env_ids: current.calls.append("post"), mode="post_physics"
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

    assert env.calls == ["startup", "reset", "pre", "step", "interval", "post"]
