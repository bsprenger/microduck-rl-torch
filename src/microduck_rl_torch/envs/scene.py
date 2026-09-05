"""Declarative scene and entity specifications.

The upstream project separates task configuration from the MuJoCo entity that a
task uses.  This module provides the small, dependency-free equivalent used by
the Torch environment.  The specifications are intentionally immutable: task
factories clone them and mutate the containing scene configuration instead of
mutating a shared robot constant.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import xml.etree.ElementTree as ET
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
    root_body_name: str | None = None
    trunk_body_name: str = "trunk_base"
    head_body_names: tuple[str, ...] = (
        "neck",
        "neck_pitch",
        "yaw_roll_motion",
        "bottom_head_shell",
        "jaw_soft",
        "bearing_roll",
    )
    foot_site_selector: SemanticSelector | None = field(
        default_factory=lambda: SemanticSelector(names=("left_foot", "right_foot"))
    )
    foot_contact_selectors: tuple[SemanticSelector, SemanticSelector] | None = field(
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
    scene_xml: Path | None = None
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
        terrain = config.terrain
        if terrain.scene_xml is not None:
            xml_path = terrain.scene_xml.resolve()
        elif callable(terrain.generator):
            xml_path = Path(terrain.generator(config)).resolve()
        elif config.scene_xml is not None:
            xml_path = config.scene_xml.resolve()
        elif len(entity_paths) == 1:
            xml_path = next(iter(entity_paths.values()))
        else:
            raise ValueError(
                "A multi-entity scene requires an explicit scene_xml or a future scene composer"
            )
        if (
            terrain.kind != "plane"
            and terrain.scene_xml is None
            and not callable(terrain.generator)
        ):
            raise ValueError(
                f"Terrain kind {terrain.kind!r} requires a concrete scene_xml or callable generator"
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


def make_microduck_rough_scene(config: SceneCfg) -> Path:
    """Build a deterministic MuJoCo rough-terrain scene for a task config.

    Upstream creates its terrain through ``TerrainGeneratorCfg`` before the
    vectorized scene is compiled.  The Torch backend has one compiled XML
    boundary, so the equivalent operation is an explicit pre-compilation
    scene generator.  The generated scene preserves the robot include,
    materials, lights, and keyframes from the task's base scene and adds a
    bounded representative obstacle.

    This is intentionally a real scene transformation, not a ``rough`` flag
    carried as metadata. A compact scalar obstacle keeps the
    ``mujoco-torch`` contact graph bounded; upstream's vectorized terrain
    importer can afford many terrain cells because it compiles a different
    scene representation. The generator's parameters are part of the cache
    key, so changing terrain settings cannot accidentally reuse an old scene.
    """

    if not config.entities:
        raise ValueError("A rough scene needs at least one configured entity")
    base_path = config.scene_xml
    if base_path is None:
        robot = config.entities.get("robot") or next(iter(config.entities.values()))
        base_path = robot.load_path
    base_path = base_path.resolve()
    params = {
        "size": config.terrain.params.get("size", (8.0, 8.0)),
        # Retain upstream's grid settings as declarative metadata. The scalar
        # backend materializes one representative obstacle because compiling
        # the full vectorized grid is a backend-level resource mismatch.
        "rows": int(config.terrain.params.get("rows", 10)),
        "cols": int(config.terrain.params.get("cols", 20)),
        "max_height": float(config.terrain.params.get("max_height", 0.015)),
        "seed": int(config.terrain.params.get("seed", 0)),
    }
    if params["rows"] < 1 or params["cols"] < 1:
        raise ValueError("Rough terrain rows and cols must be positive")
    if params["max_height"] < 0:
        raise ValueError("Rough terrain max_height must be non-negative")
    cache_key = hashlib.sha256(
        b"microduck-rough-scene-v8\0"
        + base_path.read_bytes()
        + json.dumps(params, sort_keys=True).encode()
    ).hexdigest()[:16]
    output_dir = Path(tempfile.gettempdir()) / "microduck_rl_torch" / "scenes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rough_{cache_key}.xml"
    if output_path.is_file():
        return output_path

    root = ET.parse(base_path).getroot()
    for include in root.findall("include"):
        include_path = Path(include.attrib["file"])
        if not include_path.is_absolute():
            include_path = (base_path.parent / include_path).resolve()
        # MuJoCo applies the included file's own compiler settings.  Copy the
        # robot wrapper into the cache with an absolute meshdir so compilation
        # remains valid even though the rough scene itself is generated under
        # the system temporary directory.
        included_root = ET.parse(include_path).getroot()
        compiler = included_root.find("compiler")
        if compiler is not None and compiler.get("meshdir") is not None:
            meshdir = Path(compiler.get("meshdir", ""))
            if not meshdir.is_absolute():
                meshdir = (include_path.parent / meshdir).resolve()
            compiler.set("meshdir", str(meshdir))
        included_output = output_dir / f"{include_path.stem}_{cache_key}.xml"
        ET.ElementTree(included_root).write(included_output, encoding="utf-8", xml_declaration=True)
        include.set("file", str(included_output))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Terrain base scene has no worldbody: {base_path}")
    width_x, width_y = (float(value) for value in params["size"])
    # A fixed seed and bounded boxes make this deterministic and reproducible
    # on CPU, MPS, and CUDA without involving the simulator RNG.
    generator = random.Random(params["seed"])
    height = generator.uniform(0.0, params["max_height"])
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "terrain_obstacle",
            "type": "box",
            "pos": f"{width_x / 4.0:.6f} 0 {height / 2.0:.6f}",
            "size": f"{width_x / 8.0:.6f} {width_y / 2.0:.6f} {max(height / 2.0, 0.0005):.6f}",
            "material": "groundplane",
            "contype": "1",
            "conaffinity": "1",
            "solref": "0.04 1",
            "solimp": "0.85 0.95 0.001 0.5 2",
        },
    )
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
