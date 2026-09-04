"""Upstream-compatible MicroDuck entity/model variants."""

from __future__ import annotations

from pathlib import Path

from microduck_rl_torch.envs.scene import EntityCfg, SemanticSelector

SERVO_JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


def _asset_root() -> Path:
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[3] / "assets/robot/microduck",
        module_path.parents[2] / "assets/robot/microduck",
        Path.cwd() / "assets/robot/microduck",
    )
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def _entity(
    name: str,
    robot_xml: str,
    scene_xml: str,
    *,
    foot_selectors: tuple[SemanticSelector, SemanticSelector] | None = None,
    keyframe_name: str | None = "STAND",
) -> EntityCfg:
    root = _asset_root()
    return EntityCfg(
        name=name,
        xml_path=root / robot_xml,
        scene_xml_path=root / scene_xml,
        keyframe_name=keyframe_name,
        foot_contact_selectors=foot_selectors
        or (
            SemanticSelector(names=("left_foot_collision",)),
            SemanticSelector(names=("right_foot_collision",)),
        ),
        actuator_joint_names=SERVO_JOINT_NAMES,
    )


_NAMED_FOOT_SELECTORS = (
    SemanticSelector(names=("left_foot_collision",)),
    SemanticSelector(names=("right_foot_collision",)),
)
_ROLLER_FOOT_SELECTORS = (
    SemanticSelector(mode="body_subtree", pattern=r"ankle_l_v1"),
    SemanticSelector(mode="body_subtree", pattern=r"ankle_r_v1"),
)


MICRODUCK_WALK_ROBOT_CFG = _entity(
    "robot",
    "robot_walk.xml",
    "scene_walk.xml",
    foot_selectors=_NAMED_FOOT_SELECTORS,
)
MICRODUCK_STANDUP_ROBOT_CFG = _entity(
    "robot",
    "robot_groundcontact.xml",
    "scene.xml",
    foot_selectors=_NAMED_FOOT_SELECTORS,
)
MICRODUCK_GROUND_PICK_ROBOT_CFG = _entity(
    "robot",
    "robot_groundcontact.xml",
    "scene.xml",
    foot_selectors=_NAMED_FOOT_SELECTORS,
)
MICRODUCK_WALK_ROLLERS_ROBOT_CFG = _entity(
    "robot",
    "robot_groundcontact_rollers.xml",
    "scene_rollers.xml",
    foot_selectors=_ROLLER_FOOT_SELECTORS,
)
MICRODUCK_BACKLASH_ROBOT_CFG = _entity(
    "robot",
    "robot_groundcontact_backlash.xml",
    "scene_backlash.xml",
    foot_selectors=_NAMED_FOOT_SELECTORS,
)
MICRODUCK_WALK_BACKLASH_ROBOT_CFG = _entity(
    "robot",
    "robot_walk_backlash.xml",
    "scene_walk_backlash.xml",
    foot_selectors=_NAMED_FOOT_SELECTORS,
)
MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG = _entity(
    "robot",
    "robot_groundcontact_rollers_backlash.xml",
    "robot_groundcontact_rollers_backlash.xml",
    foot_selectors=_ROLLER_FOOT_SELECTORS,
)
