from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import torch

from microduck_rl_torch.envs import (
    CommandConfig,
    CommandTermCfg,
    EntityInitStateCfg,
    EventTermCfg,
    ManagerBasedTaskEnv,
    TaskRuntimeCfg,
    TermCollection,
)
from microduck_rl_torch.envs.core import _EntityDataView
from microduck_rl_torch.envs.model import load_model_bundle
from microduck_rl_torch.robot import MICRODUCK_BALL_CFG
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg


def test_body_velocity_transport_matches_subtree_com_convention():
    class Data:
        xpos = torch.tensor([[[1.0, 0.0, 0.0]]])
        xipos = torch.tensor([[[0.0, 0.0, 0.0]]])
        subtree_com = torch.tensor([[[0.0, 0.0, 0.0]]])
        cvel = torch.tensor([[[0.0, 0.0, 1.0, 2.0, 0.0, 0.0]]])

    result = _EntityDataView._world_velocity(Data(), (0,), Data.subtree_com[:, 0, :])

    # v(point) = v(com) + omega x (point - com).
    torch.testing.assert_close(result, torch.tensor([[[2.0, 1.0, 0.0, 0.0, 0.0, 1.0]]]))


def test_entity_initial_velocity_matches_root_writer_layout():
    entity = replace(
        MICRODUCK_BALL_CFG,
        init_state=EntityInitStateCfg(
            linear_velocity=(1.0, 2.0, 3.0),
            angular_velocity=(4.0, 5.0, 6.0),
        ),
    )
    bundle = load_model_bundle(entity_cfg=entity, actuator_mode="xml", disable_contacts=True)
    view = bundle.entity("ball")
    torch.testing.assert_close(
        bundle.default_qvel[view.free_qvel_indices],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )


def test_ground_pick_style_stateful_command_is_initialized_before_first_observation():
    cfg = make_microduck_velocity_env_cfg().clone()
    cfg.runtime = TaskRuntimeCfg(episode_length_steps=1000, bad_orientation_degrees=70.0)
    cfg.observations.groups["actor"].expected_size = None
    cfg.observations.groups["critic"].expected_size = None
    for group in cfg.observations.groups.values():
        group.terms.remove("head_command")
        group.terms.remove("body_command")

    class GroundPickStyleCommand:
        def __init__(self, term_cfg, env):
            self.command = torch.zeros(term_cfg.size, dtype=env.bundle.dtype)
            self.phase = 0.25
            self.compute_dts: list[float] = []

        def reset(self, env_ids):
            del env_ids
            self.command.zero_()

        def compute(self, dt):
            self.compute_dts.append(float(dt))
            self.command[:] = torch.tensor(
                [self.phase + dt, 1.0 - self.phase, 0.0],
                dtype=self.command.dtype,
            )

    cfg.commands = CommandConfig(
        terms=OrderedDict(
            (
                (
                    "twist",
                    CommandTermCfg(func=GroundPickStyleCommand, size=3),
                ),
            )
        )
    )

    # The actor graph is intentionally retained: this test checks that a
    # task may replace the velocity command without changing env lifecycle.
    env = ManagerBasedTaskEnv(cfg)
    env.reset(seed=9)
    term = env.command_manager.get_term("twist")
    assert term.compute_dts == [0.0]
    torch.testing.assert_close(env.command, torch.tensor([0.25, 0.75, 0.0]))


def test_stateful_command_width_can_be_derived_from_term_instance():
    cfg = make_microduck_velocity_env_cfg().clone()
    cfg.observations.groups["actor"].expected_size = None
    cfg.observations.groups["critic"].expected_size = None
    for group in cfg.observations.groups.values():
        group.terms.remove("head_command")
        group.terms.remove("body_command")

    class CommandWithSelfDescribingWidth:
        def __init__(self, _term_cfg, env):
            self.command = torch.zeros(3, dtype=env.bundle.dtype)

        def reset(self, _env_ids):
            self.command.zero_()

        def compute(self, _dt):
            self.command[:] = torch.tensor((0.1, 0.2, 0.3), dtype=self.command.dtype)

    cfg.commands = CommandConfig(
        terms=OrderedDict((("twist", CommandTermCfg(func=CommandWithSelfDescribingWidth)),))
    )
    env = ManagerBasedTaskEnv(cfg)
    env.reset(seed=11)
    torch.testing.assert_close(env.command, torch.tensor((0.1, 0.2, 0.3)))


def test_mutating_runtime_event_owns_the_forward_barrier():
    class State:
        manager_data = {"event_next_steps": {}}

    class Physics:
        def __init__(self):
            self.forward_count = 0

        def forward(self):
            self.forward_count += 1

    class Env:
        state = State()
        physics = Physics()
        num_envs = 1
        step_count = 0
        step_counts = torch.zeros(1, dtype=torch.long)

    def write_state(env, env_ids):
        del env, env_ids

    from microduck_rl_torch.envs.managers.events import EventManager

    manager = EventManager(
        TermCollection(
            {
                "write": EventTermCfg(
                    func=write_state,
                    mode="step",
                    mutates_physics=True,
                )
            }
        )
    )
    assert manager.apply(Env(), "step") is True
    assert Env.physics.forward_count == 1
