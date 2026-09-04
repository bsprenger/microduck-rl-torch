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


@dataclass(frozen=True)
class SceneBuild:
    """Concrete scene source selected from a declarative ``SceneCfg``."""

    xml_path: Path
    entity_names: tuple[str, ...]
    terrain_kind: str


class SceneBuilder:
    """Resolve scene wrappers without leaking XML policy into task code.

    The current backend consumes one compiled MuJoCo XML source, so an
    explicit scene wrapper is preferred. A single entity's root XML is a valid
    fallback. Multi-entity composition remains at this boundary for future
    prop tasks instead of leaking an XML merge policy into the environment.
    """

    def __init__(self, config: SceneCfg | None = None) -> None:
        self.config = config

    def build(self, config: SceneCfg | None = None) -> SceneBuild:
        config = config or self.config
        if config is None:
            raise ValueError("SceneBuilder requires a SceneCfg")
        if not config.entities:
            raise ValueError("A scene must contain at least one entity")
        entity_paths = {
            name: entity.load_path.resolve() for name, entity in config.entities.items()
        }
        if config.scene_xml is not None:
            xml_path = config.scene_xml.resolve()
        elif len(entity_paths) == 1:
            xml_path = next(iter(entity_paths.values()))
        else:
            raise ValueError(
                "A multi-entity scene requires an explicit scene_xml or a future scene composer"
            )
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        for name, path in entity_paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Entity {name!r} XML source does not exist: {path}")
        return SceneBuild(
            xml_path=xml_path,
            entity_names=tuple(config.entities),
            terrain_kind=config.terrain.kind,
        )
