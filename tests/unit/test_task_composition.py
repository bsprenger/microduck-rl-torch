from __future__ import annotations

from collections import OrderedDict

import torch

from microduck_rl_torch.envs import (
    CommandTermCfg,
    CurriculumTermCfg,
    ObservationGroupCfg,
    ObservationGroupsCfg,
    ObservationTermCfg,
    TermCfg,
    TermCollection,
)
from microduck_rl_torch.envs.config import CommandConfig
from microduck_rl_torch.envs.managers.commands import CommandManager
from microduck_rl_torch.envs.managers.curriculum import CurriculumManager
from microduck_rl_torch.envs.managers.events import EventManager
from microduck_rl_torch.envs.managers.observations import ObservationManager
from microduck_rl_torch.envs.managers.rewards import RewardManager
from microduck_rl_torch.envs.managers.terminations import TerminationManager
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg
from microduck_rl_torch.tasks.backlash import make_microduck_velocity_backlash_env_cfg
from microduck_rl_torch.tasks.names import (
    BACKLASH_TASK_NAMES,
    BASE_TASK_NAMES,
    MJLAB_VELOCITY_FLAT_MICRODUCK,
    MJLAB_VELOCITY_ROUGH_MICRODUCK,
)


def test_task_names_are_explicit_and_registration_free():
    assert len(BASE_TASK_NAMES) == 18
    assert len(BACKLASH_TASK_NAMES) == 15
    assert MJLAB_VELOCITY_FLAT_MICRODUCK == "Mjlab-Velocity-Flat-MicroDuck"
    assert MJLAB_VELOCITY_ROUGH_MICRODUCK == "Mjlab-Velocity-Rough-MicroDuck"


def test_velocity_factory_composes_fresh_flat_and_rough_configs():
    flat = make_microduck_velocity_env_cfg()
    rough = make_microduck_velocity_env_cfg(rough=True)
    play = make_microduck_velocity_env_cfg(play=True)

    assert flat.task_name == "Mjlab-Velocity-Flat-MicroDuck"
    assert rough.task_name == "Mjlab-Velocity-Rough-MicroDuck"
    assert flat.scene.terrain.kind == "plane"
    assert rough.scene.terrain.kind == "generator"
    assert play.play
    assert flat.observations.groups["actor"].expected_size == 61
    assert flat.actions.size == 14
    assert flat.metadata["rl_cfg"].runner == "MicroduckOnPolicyRunner"
    assert flat.rewards.names == (
        "pose",
        "upright",
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "head_pose_tracking",
        "foot_slip",
        "body_ang_vel",
        "angular_momentum",
        "dof_pos_limits",
        "action_rate_l2",
        "foot_clearance",
        "foot_swing_height",
        "self_collisions",
    )
    assert flat.terminations.names == ("non_finite", "bad_orientation", "timeout")

    flat.rewards.replace("pose", flat.rewards["pose"].clone())
    flat.rewards["pose"].weight = 123.0
    assert rough.rewards["pose"].weight == 1.0


def test_backlash_is_a_model_overlay_not_a_new_task_graph():
    base = make_microduck_velocity_env_cfg()
    backlash = make_microduck_velocity_backlash_env_cfg(base)

    assert backlash.task_name == "Mjlab-Velocity-Flat-Backlash-MicroDuck"
    assert backlash.metadata["backlash"] is True
    assert backlash.metadata["base_task_name"] == base.task_name
    assert backlash.scene.entities["robot"].xml_path.name == "robot_walk_backlash.xml"
    scene_xml = backlash.scene.entities["robot"].keyframe_source
    assert scene_xml is not None and scene_xml.name == "scene_walk_backlash.xml"
    assert backlash.rewards.names == base.rewards.names
    assert backlash.actions.size == base.actions.size


def test_command_manager_composes_arbitrary_term_dimensions():
    class FakeBundle:
        dtype = torch.float32
        device = torch.device("cpu")

    class FakeState:
        manager_data = {}

    class FakeEnv:
        bundle = FakeBundle()
        state = FakeState()
        domain_randomization = True
        step_count = 0
        _generator = None

        @staticmethod
        def _next_interval_step(_interval):
            return 1

    env = FakeEnv()
    manager = CommandManager(
        CommandConfig(
            terms=OrderedDict(
                (
                    (
                        "posture",
                        CommandTermCfg(func=lambda _env, env_ids: torch.ones(2), size=2),
                    ),
                    (
                        "phase",
                        CommandTermCfg(
                            func=lambda _env, env_ids: torch.tensor([2.0]),
                            size=1,
                            resampling_time_range=(0.0, 0.0),
                        ),
                    ),
                )
            )
        )
    )
    manager.reset(env)
    assert manager.size == 3
    torch.testing.assert_close(manager.command, torch.tensor([1.0, 1.0, 2.0]))
    env.step_count = 1
    manager.step(env)
    torch.testing.assert_close(manager.command, torch.tensor([1.0, 1.0, 2.0]))


def test_command_manager_samples_without_domain_randomization_and_supports_class_terms():
    class FakeBundle:
        dtype = torch.float32
        device = torch.device("cpu")
        timestep = 0.1

    class FakeEnv:
        bundle = FakeBundle()
        state = type("State", (), {})()
        domain_randomization = False
        step_count = 0
        decimation = 1
        _generator = torch.Generator().manual_seed(4)

        @staticmethod
        def _next_interval_step(_interval):
            return 1

    class StatefulCommand:
        def __init__(self, cfg, env):
            self.command = torch.zeros(cfg.size, dtype=env.bundle.dtype)
            self.reset_ids = None
            self.compute_calls = 0

        def reset(self, env_ids):
            self.reset_ids = env_ids.clone()
            self.command.fill_(2.0)

        def compute(self, _dt):
            self.compute_calls += 1
            self.command.add_(1.0)

    env = FakeEnv()
    function_manager = CommandManager(
        CommandConfig(
            terms=OrderedDict(
                (
                    (
                        "sampled",
                        CommandTermCfg(func=lambda _env, env_ids: torch.tensor([3.0]), size=1),
                    ),
                )
            )
        )
    )
    function_manager.reset(env)
    torch.testing.assert_close(function_manager.command, torch.tensor([3.0]))

    class_manager = CommandManager(
        CommandConfig(
            terms=OrderedDict((("stateful", CommandTermCfg(func=StatefulCommand, size=1)),))
        )
    )
    class_manager.reset(env)
    term = class_manager.get_term("stateful")
    assert isinstance(term, StatefulCommand)
    torch.testing.assert_close(class_manager.get_command("stateful"), torch.tensor([2.0]))
    class_manager.step(env)
    torch.testing.assert_close(class_manager.get_command("stateful"), torch.tensor([3.0]))
    assert term.compute_calls == 1


def test_empty_command_manager_is_a_valid_zero_width_composition():
    class FakeBundle:
        dtype = torch.float32
        device = torch.device("cpu")

    class FakeEnv:
        bundle = FakeBundle()
        step_count = 0

        @staticmethod
        def _next_interval_step(_interval):
            return 1

    manager = CommandManager(CommandConfig())
    manager.reset(FakeEnv())
    assert manager.command is not None
    assert manager.command.shape == (0,)


def test_observation_and_reward_managers_are_term_driven():
    class FakeBundle:
        dtype = torch.float32
        device = torch.device("cpu")
        timestep = 0.1

    class FakeEnv:
        bundle = FakeBundle()
        decimation = 2

    env = FakeEnv()
    observations = ObservationManager(
        ObservationGroupsCfg(
            groups={
                "actor": ObservationGroupCfg(
                    terms=TermCollection(
                        OrderedDict(
                            (
                                (
                                    "first",
                                    ObservationTermCfg(func=lambda _env: torch.tensor([1.0])),
                                ),
                                (
                                    "second",
                                    ObservationTermCfg(
                                        func=lambda _env: torch.tensor([2.0, 3.0]), scale=2.0
                                    ),
                                ),
                            )
                        )
                    ),
                    expected_size=3,
                )
            }
        )
    )
    torch.testing.assert_close(observations.compute(env), torch.tensor([1.0, 4.0, 6.0]))

    calls: list[str] = []
    rewards = RewardManager(
        TermCollection(
            OrderedDict(
                (
                    ("a", TermCfg(func=lambda _env: calls.append("a") or torch.tensor(2.0))),
                    (
                        "b",
                        TermCfg(
                            func=lambda _env: calls.append("b") or torch.tensor(3.0), weight=2.0
                        ),
                    ),
                )
            )
        ),
        scale_by_dt=True,
    )
    reward, terms = rewards.compute(env)
    torch.testing.assert_close(reward, torch.tensor(1.6))
    assert calls == ["a", "b"]
    assert set(terms) == {"a", "b"}


def test_observation_manager_supports_object_noise_and_non_concatenated_groups():
    class FakeBundle:
        dtype = torch.float32
        device = torch.device("cpu")

    class FakeEnv:
        bundle = FakeBundle()
        num_envs = 2

    class AddOneNoise:
        def apply(self, value):
            return value + 1.0

    manager = ObservationManager(
        ObservationGroupsCfg(
            groups={
                "critic": ObservationGroupCfg(
                    concatenate_terms=False,
                    terms=TermCollection(
                        {
                            "first": ObservationTermCfg(
                                func=lambda _env: torch.tensor([1.0]),
                                noise=AddOneNoise(),
                            ),
                            "second": ObservationTermCfg(
                                func=lambda _env: torch.tensor([2.0]),
                            ),
                        }
                    ),
                )
            }
        )
    )

    result = manager.compute(FakeEnv(), "critic")
    assert isinstance(result, dict)
    torch.testing.assert_close(result["first"], torch.tensor([[2.0], [2.0]]))
    torch.testing.assert_close(result["second"], torch.tensor([[2.0], [2.0]]))
    cached = manager.compute(FakeEnv(), "critic")
    assert cached is result


def test_observation_manager_adds_callback_noise_and_can_read_without_advancing_delay():
    class FakeBundle:
        dtype = torch.float32
        device = torch.device("cpu")

    class FakeEnv:
        bundle = FakeBundle()
        num_envs = 1
        value = torch.tensor([1.0])

        @staticmethod
        def _sample_delay(_minimum, _maximum, *, per_env=False):
            del per_env
            return 1

        @staticmethod
        def _random_tensor(shape, *, dtype):
            return torch.zeros(shape, dtype=dtype)

    env = FakeEnv()
    manager = ObservationManager(
        ObservationGroupsCfg(
            groups={
                "actor": ObservationGroupCfg(
                    terms=TermCollection(
                        {
                            "value": ObservationTermCfg(
                                func=lambda current: current.value,
                                noise=lambda _env, _value: torch.full((1,), 0.5),
                                delay_max_lag=1,
                            )
                        }
                    )
                )
            }
        )
    )

    torch.testing.assert_close(manager.compute(env, update_history=True), torch.tensor([1.5]))
    env.value = torch.tensor([2.0])
    torch.testing.assert_close(manager.compute(env), torch.tensor([1.5]))
    buffer_length = len(manager._temporal[("actor", "value")]["buffer"])
    torch.testing.assert_close(manager.compute(env, update_history=False), torch.tensor([1.5]))
    assert len(manager._temporal[("actor", "value")]["buffer"]) == buffer_length


def test_observation_manager_resamples_partial_delay_phase():
    class FakeBundle:
        dtype = torch.float32
        device = torch.device("cpu")

    class FakePhysics:
        generators = [torch.Generator().manual_seed(7)]

    class FakeEnv:
        bundle = FakeBundle()
        physics = FakePhysics()
        num_envs = 2

        @staticmethod
        def _sample_delay(_minimum, _maximum, *, per_env=False):
            assert per_env
            return torch.ones(2, dtype=torch.long)

        @staticmethod
        def _random_tensor(shape, *, dtype):
            return torch.zeros(shape, dtype=dtype)

    env = FakeEnv()
    manager = ObservationManager(
        ObservationGroupsCfg(
            groups={
                "actor": ObservationGroupCfg(
                    terms=TermCollection(
                        {
                            "value": ObservationTermCfg(
                                func=lambda _env: torch.ones(2),
                                delay_max_lag=1,
                                delay_update_period=4,
                            )
                        }
                    )
                )
            }
        )
    )
    manager.compute(env, update_history=True)
    state = manager._temporal[("actor", "value")]
    state["phase"] = torch.tensor([9, 8], dtype=torch.long)
    state["step"] = torch.tensor([4, 5], dtype=torch.long)
    manager.reset(env, torch.tensor([0]))

    assert 0 <= state["phase"][0] < 4
    assert state["phase"][1] == 8
    assert state["step"].tolist() == [0, 5]


def test_interval_scheduling_uses_elapsed_seconds_ceiling():
    class FakeBundle:
        device = torch.device("cpu")
        timestep = 0.1

    class FakeEnv:
        bundle = FakeBundle()
        decimation = 1
        num_envs = 2
        step_counts = torch.zeros(2, dtype=torch.long)

        @staticmethod
        def _sample_range(_low, _high, *, env_ids=None):
            size = 2 if env_ids is None else torch.as_tensor(env_ids).numel()
            return torch.full((size,), 0.14)

    schedule = EventManager(TermCollection())._next_interval_steps(FakeEnv(), (0.14, 0.14))
    torch.testing.assert_close(schedule, torch.tensor([2, 2]))


def test_curriculum_manager_uses_explicit_env_ids_and_publishes_extras():
    class FakeState:
        manager_data: dict[str, object] = {}

    class FakeEnv:
        state = FakeState()

    calls: list[torch.Tensor | slice] = []

    def curriculum(env, env_ids, *, increment):
        del env
        calls.append(env_ids)
        return {"increment": increment}

    manager = CurriculumManager(
        TermCollection({"schedule": CurriculumTermCfg(func=curriculum, params={"increment": 0.25})})
    )
    env = FakeEnv()
    result = manager.compute(env, torch.tensor([1]))

    assert isinstance(calls[0], torch.Tensor)
    torch.testing.assert_close(calls[0], torch.tensor([1]))
    assert result == {"schedule": {"increment": 0.25}}
    assert env.state.manager_data["curriculum"] == result
    assert not hasattr(manager, "step")


def test_termination_manager_supports_stateful_terms_and_explicit_timeouts():
    class FakeBundle:
        device = torch.device("cpu")

    class FakeEnv:
        bundle = FakeBundle()

    class Fails:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.env = env

        def __call__(self, _env):
            return torch.tensor(True)

    manager = TerminationManager(
        TermCollection(
            OrderedDict(
                (
                    ("failure", TermCfg(func=Fails)),
                    (
                        "episode_limit",
                        TermCfg(func=lambda _env: torch.tensor(True), time_out=True),
                    ),
                )
            )
        )
    )
    terminated, truncated, values = manager.evaluate(FakeEnv(), finite=True)

    assert terminated
    assert truncated
    assert values == {"failure": True, "episode_limit": True}
