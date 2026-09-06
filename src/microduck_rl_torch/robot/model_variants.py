"""Upstream-compatible MicroDuck entity/model variants."""

from __future__ import annotations

from pathlib import Path

from microduck_rl_torch.envs.scene import EntityCfg, SemanticSelector

from .constants import SERVO_JOINT_NAMES


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
        root_body_name="trunk_base",
        head_body_names=(
            "neck",
            "neck_pitch",
            "yaw_roll_motion",
            "bottom_head_shell",
            "jaw_soft",
            "bearing_roll",
        ),
        foot_contact_selectors=foot_selectors
        or (
            SemanticSelector(names=("left_foot_collision",)),
            SemanticSelector(names=("right_foot_collision",)),
        ),
        foot_site_selector=SemanticSelector(names=("left_foot", "right_foot")),
        collision_name_suffix="_collision",
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
MICRODUCK_ALLCOLLISIONS_ROBOT_CFG = _entity(
    "robot",
    "robot_allcollisions.xml",
    "scene_apartment.xml",
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
    "scene_rollers_backlash.xml",
    foot_selectors=_ROLLER_FOOT_SELECTORS,
)

MICRODUCK_BALL_CFG = EntityCfg(
    name="ball",
    xml_path=_asset_root() / "ball.xml",
    kind="prop",
    keyframe_name=None,
    root_body_name="ball",
    foot_site_selector=None,
    foot_contact_selectors=None,
    actuator_joint_names=(),
)
