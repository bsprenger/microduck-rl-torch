"""Declarative scene and entity specifications.

The upstream project separates task configuration from the MuJoCo entity that a
task uses.  This module provides the small, dependency-free equivalent used by
the Torch runtime.  The specifications are intentionally immutable: task
factories clone them and mutate the containing scene configuration instead of
mutating a shared robot constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


SelectorMode = Literal["names", "regex", "body_subtree"]


@dataclass(frozen=True)
class SemanticSelector:
    """Resolve one or more MuJoCo objects by semantic names or body subtrees."""

    mode: SelectorMode = "names"
    names: tuple[str, ...] = ()
    pattern: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "names" and not self.names:
            raise ValueError("A names selector requires at least one name")
        if self.mode in {"regex", "body_subtree"} and not self.pattern:
            raise ValueError(f"A {self.mode} selector requires a pattern")


@dataclass(frozen=True)
class EntityCfg:
    """One scene entity and the semantic handles required by task terms."""

    name: str
    xml_path: Path
    scene_xml_path: Path | None = None
    kind: str = "robot"
    keyframe_name: str | None = "STAND"
    trunk_body_name: str = "trunk_base"
    head_body_names: tuple[str, ...] = (
        "neck",
        "neck_pitch",
        "yaw_roll_motion",
        "bottom_head_shell",
        "jaw_soft",
        "bearing_roll",
    )
    foot_site_selector: SemanticSelector = field(
        default_factory=lambda: SemanticSelector(names=("left_foot", "right_foot"))
    )
    foot_contact_selectors: tuple[SemanticSelector, SemanticSelector] = field(
        default_factory=lambda: (
            SemanticSelector(names=("left_foot_collision",)),
            SemanticSelector(names=("right_foot_collision",)),
        )
    )
    collision_name_suffix: str = "_collision"
    actuator_mode: Literal["bam", "xml"] = "bam"
    actuator_joint_names: tuple[str, ...] = ()

    @property
    def load_path(self) -> Path:
        """Return the compatibility scene wrapper when one is provided."""

        return (self.scene_xml_path or self.xml_path).resolve()


@dataclass
class TerrainCfg:
    """Terrain declaration; generators are deliberately opaque to the core."""

    kind: Literal["plane", "generator", "ramp", "apartment"] = "plane"
    generator: Any | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorCfg:
    """Named sensor contract used by an environment."""

    name: str
    required: bool = True
    expected_dim: int | None = None


@dataclass
class SceneCfg:
    """Composable scene containing named entities, terrain, and sensors."""

    entities: dict[str, EntityCfg] = field(default_factory=dict)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    sensors: dict[str, SensorCfg] = field(default_factory=dict)
    scene_xml: Path | None = None
    contact_options: dict[str, Any] = field(default_factory=dict)

