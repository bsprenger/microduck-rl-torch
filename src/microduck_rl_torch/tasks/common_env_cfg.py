"""Common configuration composition for MicroDuck task factories."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from microduck_rl_torch.envs.config import CommandConfig, MicroDuckVelocityConfig
from microduck_rl_torch.envs.managers import bad_orientation, timeout
from microduck_rl_torch.envs.scene import SceneCfg, SensorCfg, TerrainCfg
from microduck_rl_torch.envs.task_config import (
    ActionCfg,
    ObservationGroupCfg,
    ObservationGroupsCfg,
    TaskEnvCfg,
    TermCfg,
    TermCollection,
)
from microduck_rl_torch.robot import MICRODUCK_WALK_ROBOT_CFG


@dataclass(frozen=True)
class MicroduckRlCfg:
    """Training metadata placeholder matching the upstream velocity name."""

    algorithm: str = "on_policy"
    runner: str = "MicroduckOnPolicyRunner"
    actor_observation_group: str = "actor"
    critic_observation_group: str = "critic"


def _actor_observation(ctx):  # type: ignore[no-untyped-def]
    return ctx.env._build_actor_observation()


def _push_event(ctx):  # type: ignore[no-untyped-def]
    ctx.env._apply_push()


def _velocity_rewards() -> TermCollection:
    return TermCollection(
        OrderedDict(
            (
                ("pose", TermCfg(weight=1.0)),
                ("upright", TermCfg(weight=2.0)),
                ("track_linear_velocity", TermCfg(weight=2.0)),
                ("track_angular_velocity", TermCfg(weight=2.0)),
                ("air_time", TermCfg(weight=3.0)),
                ("head_pose_tracking", TermCfg(weight=2.0)),
                ("foot_slip", TermCfg(weight=-0.1)),
                ("body_ang_vel", TermCfg(weight=-0.05)),
                ("angular_momentum", TermCfg(weight=-0.02)),
                ("action_rate_l2", TermCfg(weight=-0.1)),
                ("foot_clearance", TermCfg(weight=-2.0)),
                ("foot_swing_height", TermCfg(weight=-0.25)),
                ("self_collisions", TermCfg(weight=-1.0)),
            )
        )
    )


def _velocity_terminations() -> TermCollection:
    return TermCollection(
        OrderedDict(
            (
                ("non_finite", TermCfg()),
                ("bad_orientation", TermCfg(func=bad_orientation)),
                ("timeout", TermCfg(func=timeout)),
            )
        )
    )


def make_velocity_env_cfg(*, play: bool = False) -> TaskEnvCfg:
    """Create a fresh generic velocity configuration to mutate per robot."""

    runtime = MicroDuckVelocityConfig()
    robot = MICRODUCK_WALK_ROBOT_CFG
    return TaskEnvCfg(
        task_name="Velocity",
        scene=SceneCfg(
            entities={robot.name: robot},
            terrain=TerrainCfg(kind="plane"),
            sensors={
                "imu_ang_vel": SensorCfg("imu_ang_vel", expected_dim=3),
                "imu_accel": SensorCfg("imu_accel", expected_dim=3),
            },
            scene_xml=robot.load_path,
        ),
        actions=ActionCfg(size=14, scale=1.0, delay_lag=0, actuator_mode="bam"),
        commands=CommandConfig(),
        observations=ObservationGroupsCfg(
            groups={
                "actor": ObservationGroupCfg(builder=_actor_observation, expected_size=61),
                # The upstream policy is actor-facing today, but retaining a
                # named critic group prevents asymmetric observations from
                # becoming an architectural afterthought in future tasks.
                # The first Torch milestone has no trainer/critic contract;
                # keep the group named and disabled until its privileged
                # upstream terms are implemented rather than pretending it is
                # identical to the actor.
                "critic": ObservationGroupCfg(
                    builder=None,
                    expected_size=None,
                    enabled=False,
                ),
            }
        ),
        rewards=_velocity_rewards(),
        terminations=_velocity_terminations(),
        events=TermCollection(
            OrderedDict(
                (
                    (
                        "velocity_push",
                        TermCfg(func=_push_event, params={"stage": "pre_physics"}),
                    ),
                )
            )
        ),
        curriculum=TermCollection(),
        runtime=runtime,
        play=play,
        metadata={
            "family": "velocity",
            "rl_cfg": MicroduckRlCfg(),
            "domain_randomization": False,
        },
    )
