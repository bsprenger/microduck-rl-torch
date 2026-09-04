"""Common configuration composition for Microduck task factories."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from microduck_rl_torch.envs.config import CommandConfig, CommandTermCfg, MicroDuckVelocityConfig
from microduck_rl_torch.envs.managers import (
    ResetState,
    SensorStateCfg,
    bad_orientation,
    body_pose_command,
    head_pose_command,
    timeout,
    velocity_command,
)
from microduck_rl_torch.envs.observations import (
    base_ang_vel,
    base_lin_vel,
    command_term,
    foot_air_time,
    foot_contact,
    foot_contact_forces,
    foot_height,
    joint_position,
    joint_velocity,
    last_action,
    projected_gravity,
)
from microduck_rl_torch.envs.rewards import velocity_term
from microduck_rl_torch.envs.scene import SceneCfg, SemanticSelector, SensorCfg, TerrainCfg
from microduck_rl_torch.envs.sensors import ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from microduck_rl_torch.envs.task_config import (
    ActionCfg,
    EventTermCfg,
    JointPositionActionTermCfg,
    ModelMutationTermCfg,
    MutationDistributionCfg,
    ObservationGroupCfg,
    ObservationGroupsCfg,
    ObservationTermCfg,
    ResetStateTermCfg,
    TaskEnvCfg,
    TaskRuntimeCfg,
    TermCfg,
    TermCollection,
)
from microduck_rl_torch.robot import MICRODUCK_WALK_ROBOT_CFG


@dataclass(frozen=True)
class MicroduckRlCfg:
    """Training metadata associated with the velocity task."""

    algorithm: str = "on_policy"
    runner: str = "MicroduckOnPolicyRunner"
    actor_observation_group: str = "actor"
    critic_observation_group: str = "critic"


def _observation_noise(env, value, scale: float):  # type: ignore[no-untyped-def]
    return env._observation_noise(value.shape, scale)


def _push_event(env, env_ids=None, *, entity_name: str = "robot"):  # type: ignore[no-untyped-def]
    del env_ids
    if not env.domain_randomization or not getattr(env.config, "randomize_velocity_pushes", False):
        return
    randomization = getattr(env.config, "randomization", None)
    if randomization is None:
        return
    entity = env.entity(entity_name)
    low, high = randomization.velocity_push_range
    push = torch.stack((env._sample_range(low, high), env._sample_range(low, high)), dim=-1)
    velocity = entity.data.root_link_vel_w.clone()
    velocity[..., :2] += push
    entity.data.write_root_velocity(velocity)


def _quat_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack(
        (
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ),
        dim=-1,
    )


def _velocity_reset_state(
    env,
    reset_state: ResetState,
    env_ids=None,  # type: ignore[no-untyped-def]
) -> None:
    """Velocity-task reset policy, expressed as a reset-state mutation term."""

    config = env.config
    randomization = getattr(config, "randomization", None)
    if not env.domain_randomization or randomization is None:
        return
    entity = env.bundle.entity(env.bundle.primary_entity_name)
    free = entity.free_qpos_indices
    if free.numel() != 7:
        raise ValueError("Velocity reset requires a free primary entity")
    if getattr(config, "initial_height_range", None) is not None:
        height = env._sample_range(*config.initial_height_range, env_ids=env_ids)
        if env_ids is None or env.num_envs == 1:
            reset_state.qpos[..., free[2]] += height
        else:
            reset_state.qpos[env_ids, free[2]] += height
    if getattr(config, "randomize_base_orientation", False):
        roll = torch.deg2rad(
            env._sample_range(
                -randomization.base_roll_degrees, randomization.base_roll_degrees, env_ids
            )
        )
        pitch = torch.deg2rad(
            env._sample_range(
                -randomization.base_pitch_degrees, randomization.base_pitch_degrees, env_ids
            )
        )
        quat = _quat_from_euler(roll, pitch, torch.zeros_like(roll))
        if env_ids is None or env.num_envs == 1:
            reset_state.qpos[..., free[3:7]] = quat
        else:
            reset_state.qpos[env_ids, free[3:7]] = quat


def _velocity_model_mutations(task_config: MicroDuckVelocityConfig) -> TermCollection:
    """Declare reset-time physical randomization as composable transforms."""

    randomization = task_config.randomization
    terms = OrderedDict(
        (
            (
                "body_pseudo_inertia_scale",
                ModelMutationTermCfg(
                    field="body_pseudo_inertia",
                    selector="all",
                    operation="scale",
                    distribution=MutationDistributionCfg(
                        low=randomization.mass_inertia_range[0],
                        high=randomization.mass_inertia_range[1],
                    ),
                    enabled=task_config.randomize_mass_inertia,
                ),
            ),
            (
                "body_com_offset",
                ModelMutationTermCfg(
                    field="body_ipos",
                    selector="all",
                    operation="add",
                    distribution=MutationDistributionCfg(
                        low=-randomization.com_range,
                        high=randomization.com_range,
                    ),
                    enabled=task_config.randomize_com,
                ),
            ),
            (
                "head_com_offset",
                ModelMutationTermCfg(
                    field="body_ipos",
                    selector="head",
                    operation="add",
                    distribution=MutationDistributionCfg(
                        low=-randomization.head_com_range,
                        high=randomization.head_com_range,
                    ),
                    enabled=task_config.randomize_head_com,
                ),
            ),
            (
                "joint_armature_scale",
                ModelMutationTermCfg(
                    field="dof_armature",
                    selector="all",
                    operation="scale",
                    distribution=MutationDistributionCfg(
                        low=randomization.armature_range[0],
                        high=randomization.armature_range[1],
                    ),
                    enabled=task_config.randomize_armature,
                ),
            ),
            (
                "bam_friction_scale",
                ModelMutationTermCfg(
                    field="bam.friction_scale",
                    operation="set",
                    distribution=MutationDistributionCfg(
                        low=randomization.joint_friction_range[0],
                        high=randomization.joint_friction_range[1],
                    ),
                    enabled=task_config.randomize_joint_friction,
                ),
            ),
            (
                "foot_friction_scale",
                ModelMutationTermCfg(
                    field="geom_friction",
                    selector=SemanticSelector(mode="regex", pattern=r"foot.*collision$"),
                    operation="scale",
                    distribution=MutationDistributionCfg(
                        low=randomization.foot_friction_range[0],
                        high=randomization.foot_friction_range[1],
                    ),
                    enabled=task_config.randomize_foot_friction,
                ),
            ),
        )
    )
    return TermCollection(terms)


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
                (
                    "dof_pos_limits",
                    TermCfg(func=velocity_term("dof_pos_limits"), weight=-1.0),
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
                        delay_min_lag=0,
                        delay_max_lag=1,
                        delay_update_period=64,
                    ),
                ),
                (
                    "projected_gravity",
                    ObservationTermCfg(
                        func=projected_gravity,
                        noise=_observation_noise,
                        noise_params={"scale": task_config.actor_noise[1]},
                        params={"misaligned": True},
                        delay_min_lag=0,
                        delay_max_lag=1,
                        delay_update_period=64,
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
                ("actions", ObservationTermCfg(func=last_action)),
                (
                    "command",
                    ObservationTermCfg(func=command_term, params={"name": "twist"}),
                ),
                (
                    "head_command",
                    ObservationTermCfg(func=command_term, params={"name": "head_pose"}),
                ),
                (
                    "body_command",
                    ObservationTermCfg(func=command_term, params={"name": "body_pose"}),
                ),
            )
        )
    )
    critic_terms = TermCollection(
        OrderedDict(
            (
                ("base_lin_vel", ObservationTermCfg(func=base_lin_vel)),
                *(
                    (name, actor_terms[name].clone())
                    for name in (
                        "base_ang_vel",
                        "projected_gravity",
                        "joint_position",
                        "joint_velocity",
                        "actions",
                        "command",
                    )
                ),
            )
        )
    )
    for term in critic_terms.values():
        if isinstance(term, ObservationTermCfg):
            term.noise = None
            term.noise_params = {}
    critic_terms["base_ang_vel"].params["misaligned"] = False
    critic_terms["projected_gravity"].params["misaligned"] = False
    critic_terms["joint_position"].params["biased"] = False
    critic_terms["joint_velocity"].params["delayed"] = False
    critic_terms.add(
        "foot_height",
        ObservationTermCfg(func=foot_height, params={"sensor_name": "foot_height_scan"}),
    )
    critic_terms.add(
        "foot_air_time",
        ObservationTermCfg(func=foot_air_time, params={"sensor_name": "feet_ground_contact"}),
    )
    critic_terms.add(
        "foot_contact",
        ObservationTermCfg(func=foot_contact, params={"sensor_name": "feet_ground_contact"}),
    )
    critic_terms.add(
        "foot_contact_forces",
        ObservationTermCfg(func=foot_contact_forces, params={"sensor_name": "feet_ground_contact"}),
    )
    critic_terms.add("head_command", actor_terms["head_command"].clone())
    critic_terms.add("body_command", actor_terms["body_command"].clone())
    # Force normalization and sensor cadence still require a reference-
    # trajectory check before claiming critic-level parity.
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
                "feet_ground_contact": SensorCfg(
                    "feet_ground_contact",
                    kind="contact",
                    primary=SemanticSelector(names=("left_foot_collision", "right_foot_collision")),
                    secondary=SemanticSelector(mode="regex", pattern=r"^(floor|terrain.*)$"),
                    fields=("found", "force"),
                    reduce="netforce",
                    num_slots=1,
                    track_air_time=True,
                    expected_dim=8,
                ),
                "foot_height_scan": TerrainHeightSensorCfg(
                    name="foot_height_scan",
                    frame=(
                        ObjRef(type="site", name="left_foot", entity="robot"),
                        ObjRef(type="site", name="right_foot", entity="robot"),
                    ),
                    pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
                    ray_alignment="yaw",
                    max_distance=1.0,
                    exclude_parent_body=True,
                    include_geom_groups=(0,),
                    reduction="min",
                    expected_dim=2,
                ),
            },
            # The task source is the robot entity plus the declarative terrain;
            scene_xml=None,
        ),
        actions=ActionCfg(
            terms=OrderedDict(
                (
                    (
                        "joint_pos",
                        JointPositionActionTermCfg(
                            entity="robot", size=14, scale=1.0, offset="default"
                        ),
                    ),
                )
            ),
            actuator_mode="bam",
        ),
        commands=CommandConfig(
            terms=OrderedDict(
                (
                    (
                        "twist",
                        CommandTermCfg(
                            func=velocity_command,
                            size=3,
                            resampling_time_range=task_config.command.twist_resample_seconds,
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
                            resampling_time_range=task_config.command.head_resample_seconds,
                            params={"ranges": task_config.command.head_ranges},
                        ),
                    ),
                    (
                        "body_pose",
                        CommandTermCfg(
                            func=body_pose_command,
                            size=6,
                            resampling_time_range=task_config.command.body_resample_seconds,
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
                    expected_size=76,
                ),
            }
        ),
        rewards=_velocity_rewards(),
        terminations=_velocity_terminations(),
        reset_state=TermCollection(
            OrderedDict((("velocity_start", ResetStateTermCfg(func=_velocity_reset_state)),))
        ),
        model_mutations=_velocity_model_mutations(task_config),
        sensor_state=SensorStateCfg(
            enabled=True,
            randomize_encoder_bias=task_config.randomize_encoder_bias,
            encoder_bias_range=task_config.randomization.encoder_bias_range,
            randomize_imu_orientation=task_config.randomize_imu_orientation,
            imu_angle_degrees=task_config.randomization.imu_angle_degrees,
        ),
        events=TermCollection(
            OrderedDict(
                (
                    (
                        "velocity_push",
                        EventTermCfg(
                            func=_push_event,
                            mode="interval",
                            interval_range_s=task_config.randomization.velocity_push_interval,
                            params={"entity_name": "robot"},
                            mutates_physics=True,
                            requires_domain_randomization=True,
                        ),
                    ),
                )
            )
        ),
        curriculum=TermCollection(),
        task=task_config,
        runtime=TaskRuntimeCfg(
            episode_length_steps=task_config.episode_length_steps,
            bad_orientation_degrees=task_config.bad_orientation_degrees,
        ),
        play=play,
        reward_scale_by_dt=True,
        metadata={
            "family": "velocity",
            "rl_cfg": MicroduckRlCfg(),
            "domain_randomization": True,
            # Velocity histories are a task component; ManagerBasedTaskEnv
            # remains usable for tasks with no feet, IMU, or commands.
            "velocity_state": True,
        },
    )
