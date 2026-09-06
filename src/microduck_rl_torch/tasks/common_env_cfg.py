"""Common configuration composition for MicroDuck task factories."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from microduck_rl_torch.envs.config import CommandConfig, CommandTermCfg, MicroDuckVelocityConfig
from microduck_rl_torch.envs.managers import (
    bad_orientation,
    body_pose_command,
    head_pose_command,
    timeout,
    velocity_command,
)
from microduck_rl_torch.envs.observations import (
    base_ang_vel,
    base_lin_vel,
    command,
    joint_position,
    joint_velocity,
    last_action,
    projected_gravity,
)
from microduck_rl_torch.envs.rewards import velocity_term
from microduck_rl_torch.envs.scene import SceneCfg, SemanticSelector, SensorCfg, TerrainCfg
from microduck_rl_torch.envs.task_config import (
    ActionCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationGroupsCfg,
    ObservationTermCfg,
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


def _observation_noise(env, value, scale: float):  # type: ignore[no-untyped-def]
    return env._observation_noise(value.shape, scale)


def _push_event(env):  # type: ignore[no-untyped-def]
    env._apply_push()


def _velocity_rewards() -> TermCollection:
    return TermCollection(
        OrderedDict(
            (
                ("pose", TermCfg(func=velocity_term("pose"), weight=1.0)),
                ("upright", TermCfg(func=velocity_term("upright"), weight=2.0)),
                (
                    "track_linear_velocity",
                    TermCfg(func=velocity_term("track_linear_velocity"), weight=2.0),
                ),
                (
                    "track_angular_velocity",
                    TermCfg(func=velocity_term("track_angular_velocity"), weight=2.0),
                ),
                ("air_time", TermCfg(func=velocity_term("air_time"), weight=3.0)),
                (
                    "head_pose_tracking",
                    TermCfg(func=velocity_term("head_pose_tracking"), weight=2.0),
                ),
                ("foot_slip", TermCfg(func=velocity_term("foot_slip"), weight=-0.1)),
                ("body_ang_vel", TermCfg(func=velocity_term("body_ang_vel"), weight=-0.05)),
                (
                    "angular_momentum",
                    TermCfg(func=velocity_term("angular_momentum"), weight=-0.02),
                ),
                ("action_rate_l2", TermCfg(func=velocity_term("action_rate_l2"), weight=-0.1)),
                (
                    "foot_clearance",
                    TermCfg(func=velocity_term("foot_clearance"), weight=-2.0),
                ),
                (
                    "foot_swing_height",
                    TermCfg(func=velocity_term("foot_swing_height"), weight=-0.25),
                ),
                (
                    "self_collisions",
                    TermCfg(func=velocity_term("self_collisions"), weight=-1.0),
                ),
            )
        )
    )


def _velocity_terminations() -> TermCollection:
    return TermCollection(
        OrderedDict(
            (
                ("non_finite", TermCfg()),
                ("bad_orientation", TermCfg(func=bad_orientation)),
                ("timeout", TermCfg(func=timeout, time_out=True)),
            )
        )
    )


def make_velocity_env_cfg(*, play: bool = False) -> TaskEnvCfg:
    """Create a fresh generic velocity configuration to mutate per robot."""

    task_config = MicroDuckVelocityConfig()
    robot = MICRODUCK_WALK_ROBOT_CFG
    actor_terms = TermCollection(
        OrderedDict(
            (
                (
                    "base_ang_vel",
                    ObservationTermCfg(
                        func=base_ang_vel,
                        noise=_observation_noise,
                        noise_params={"scale": task_config.actor_noise[0]},
                        params={"misaligned": True},
                    ),
                ),
                (
                    "projected_gravity",
                    ObservationTermCfg(
                        func=projected_gravity,
                        noise=_observation_noise,
                        noise_params={"scale": task_config.actor_noise[1]},
                        params={"misaligned": True},
                    ),
                ),
                (
                    "joint_position",
                    ObservationTermCfg(
                        func=joint_position,
                        noise=_observation_noise,
                        noise_params={"scale": task_config.actor_noise[2]},
                        params={"biased": True},
                    ),
                ),
                (
                    "joint_velocity",
                    ObservationTermCfg(
                        func=joint_velocity,
                        noise=_observation_noise,
                        noise_params={"scale": task_config.actor_noise[3]},
                        params={"delayed": True},
                    ),
                ),
                ("last_action", ObservationTermCfg(func=last_action)),
                ("command", ObservationTermCfg(func=command)),
            )
        )
    )
    critic_terms = actor_terms.clone()
    for term in critic_terms.values():
        if isinstance(term, ObservationTermCfg):
            term.noise = None
            term.noise_params = {}
    critic_terms["base_ang_vel"].params["misaligned"] = False
    critic_terms["projected_gravity"].params["misaligned"] = False
    critic_terms["joint_position"].params["biased"] = False
    critic_terms["joint_velocity"].params["delayed"] = False
    critic_terms.add("base_lin_vel", ObservationTermCfg(func=base_lin_vel))
    return TaskEnvCfg(
        task_name="Velocity",
        scene=SceneCfg(
            entities={robot.name: robot},
            terrain=TerrainCfg(kind="plane"),
            sensors={
                "imu_ang_vel": SensorCfg("imu_ang_vel", expected_dim=3),
                "imu_accel": SensorCfg("imu_accel", expected_dim=3),
                "left_foot_contact": SensorCfg(
                    "left_foot_contact",
                    kind="contact",
                    primary=SemanticSelector(names=("left_foot_collision",)),
                    secondary=SemanticSelector(mode="regex", pattern=r"^(floor|terrain.*)$"),
                    expected_dim=1,
                ),
                "right_foot_contact": SensorCfg(
                    "right_foot_contact",
                    kind="contact",
                    primary=SemanticSelector(names=("right_foot_collision",)),
                    secondary=SemanticSelector(mode="regex", pattern=r"^(floor|terrain.*)$"),
                    expected_dim=1,
                ),
            },
            scene_xml=robot.load_path,
        ),
        actions=ActionCfg(size=14, scale=1.0, delay_lag=0, actuator_mode="bam"),
        commands=CommandConfig(
            terms=OrderedDict(
                (
                    (
                        "twist",
                        CommandTermCfg(
                            func=velocity_command,
                            size=3,
                            resample_interval_s=task_config.command.twist_resample_seconds,
                            params={
                                "twist_ranges": task_config.command.twist_ranges,
                                "turn_in_place_fraction": (
                                    task_config.command.turn_in_place_fraction
                                ),
                                "standing_fraction": task_config.command.standing_fraction,
                            },
                        ),
                    ),
                    (
                        "head_pose",
                        CommandTermCfg(
                            func=head_pose_command,
                            size=4,
                            resample_interval_s=task_config.command.head_resample_seconds,
                            params={"ranges": task_config.command.head_ranges},
                        ),
                    ),
                    (
                        "body_pose",
                        CommandTermCfg(
                            func=body_pose_command,
                            size=6,
                            resample_interval_s=task_config.command.body_resample_seconds,
                            params={"ranges": task_config.command.body_ranges},
                        ),
                    ),
                )
            )
        ),
        observations=ObservationGroupsCfg(
            groups={
                "actor": ObservationGroupCfg(
                    terms=actor_terms,
                    expected_size=61,
                ),
                "critic": ObservationGroupCfg(
                    terms=critic_terms,
                    expected_size=64,
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
                        EventTermCfg(
                            func=_push_event,
                            mode="interval",
                            interval_range_s=task_config.randomization.velocity_push_interval,
                            requires_domain_randomization=True,
                        ),
                    ),
                )
            )
        ),
        curriculum=TermCollection(),
        task=task_config,
        play=play,
        metadata={
            "family": "velocity",
            "rl_cfg": MicroduckRlCfg(),
            "domain_randomization": False,
            # Velocity histories are a task component; ManagerBasedTaskEnv
            # remains usable for tasks with no feet, IMU, or commands.
            "velocity_state": True,
        },
    )
