from __future__ import annotations

from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg
from microduck_rl_torch.tasks.backlash import make_microduck_velocity_backlash_env_cfg
from microduck_rl_torch.tasks.names import (
    BACKLASH_TASK_NAMES,
    BASE_TASK_NAMES,
    MJLAB_VELOCITY_FLAT_MICRODUCK,
    MJLAB_VELOCITY_ROUGH_MICRODUCK,
)


def test_upstream_task_names_are_explicit_and_registration_free():
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
    scene_xml = backlash.scene.entities["robot"].scene_xml_path
    assert scene_xml is not None and scene_xml.name == "scene_walk_backlash.xml"
    assert backlash.rewards.names == base.rewards.names
    assert backlash.actions.size == base.actions.size
