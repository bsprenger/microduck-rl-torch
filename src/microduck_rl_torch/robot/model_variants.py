"""Microduck entity and model variants."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from microduck_rl_torch.envs.actuation import ActuatorDelayCfg
from microduck_rl_torch.envs.scene import (
    AssetSwapTransform,
    EntityCfg,
    SemanticSelector,
)

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
    keyframe_source: str,
    *,
    foot_selectors: tuple[SemanticSelector, SemanticSelector] | None = None,
    keyframe_name: str | None = "STAND",
) -> EntityCfg:
    root = _asset_root()
    return EntityCfg(
        name=name,
        xml_path=root / robot_xml,
        keyframe_source=root / keyframe_source,
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
        actuator_mode="bam",
        actuator_joint_names=SERVO_JOINT_NAMES,
        actuator_delay=ActuatorDelayCfg(delay_min_lag=3, delay_max_lag=6),
    )


def with_model_transform(entity: EntityCfg, transform: object) -> EntityCfg:
    """Compose a topology/model variant without changing the task graph."""

    return replace(entity, transforms=(*entity.transforms, transform))


def asset_swap(entity: EntityCfg, xml_path: Path) -> EntityCfg:
    """Convenience transform for an alternate MJCF entity asset."""

    return with_model_transform(entity, AssetSwapTransform(xml_path.resolve()))


def _variant(
    base: EntityCfg,
    robot_xml: str,
    keyframe_source: str,
    *,
    foot_selectors: tuple[SemanticSelector, SemanticSelector] | None = None,
) -> EntityCfg:
    """Build a named model variant as an ordered transform over a base entity."""

    replacement = _asset_root() / robot_xml
    return replace(
        base,
        # Keep the base source stable.  The replacement is a transform so
        # scene composition, root discovery, and semantic indexing all use
        # the transformed artifact rather than a parallel hard-coded scene.
        xml_path=replacement,
        keyframe_source=_asset_root() / keyframe_source,
        foot_contact_selectors=foot_selectors or base.foot_contact_selectors,
        transforms=(),
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
MICRODUCK_STANDUP_ROBOT_CFG = _variant(
    MICRODUCK_WALK_ROBOT_CFG,
    "robot_groundcontact.xml",
    "scene.xml",
    foot_selectors=_NAMED_FOOT_SELECTORS,
)
MICRODUCK_ALLCOLLISIONS_ROBOT_CFG = _variant(
    MICRODUCK_WALK_ROBOT_CFG,
    "robot_allcollisions.xml",
    "scene_apartment.xml",
    foot_selectors=_NAMED_FOOT_SELECTORS,
)
MICRODUCK_GROUND_PICK_ROBOT_CFG = _variant(
    MICRODUCK_WALK_ROBOT_CFG,
    "robot_groundcontact.xml",
    "scene.xml",
    foot_selectors=_NAMED_FOOT_SELECTORS,
)
MICRODUCK_WALK_ROLLERS_ROBOT_CFG = _variant(
    MICRODUCK_WALK_ROBOT_CFG,
    "robot_groundcontact_rollers.xml",
    "scene_rollers.xml",
    foot_selectors=_ROLLER_FOOT_SELECTORS,
)


def _backlash_variant(base: EntityCfg, robot_xml: str, scene_xml: str) -> EntityCfg:
    """Select an authored backlash asset through one asset mutation.

    The repository ships backlash MJCFs because their passive joints and
    actuator-side topology are part of the authored model.  Do not append
    ``BacklashTransform`` to those files: doing so would attempt to inject a
    second set of passive joints. ``BacklashTransform`` remains available for
    entities that start from a non-backlash asset.
    """

    replacement = _asset_root() / robot_xml
    return replace(
        base,
        xml_path=replacement,
        keyframe_source=_asset_root() / scene_xml,
        transforms=(),
    )


MICRODUCK_BACKLASH_ROBOT_CFG = _backlash_variant(
    MICRODUCK_WALK_ROBOT_CFG,
    "robot_groundcontact_backlash.xml",
    "scene_backlash.xml",
)
MICRODUCK_WALK_BACKLASH_ROBOT_CFG = _backlash_variant(
    MICRODUCK_WALK_ROBOT_CFG,
    "robot_walk_backlash.xml",
    "scene_walk_backlash.xml",
)
MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG = _backlash_variant(
    MICRODUCK_WALK_ROLLERS_ROBOT_CFG,
    "robot_groundcontact_rollers_backlash.xml",
    "scene_rollers_backlash.xml",
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
