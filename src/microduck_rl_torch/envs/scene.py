"""Declarative scene and entity specifications.

The upstream project separates task configuration from the MuJoCo entity that a
task uses.  This module provides the small, dependency-free equivalent used by
the Torch environment.  The specifications are intentionally immutable: task
factories clone them and mutate the containing scene configuration instead of
mutating a shared robot constant.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
import struct
import tempfile
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .dispatch import construct, invoke_compatible

SelectorMode = Literal["names", "regex", "body_subtree", "body", "subtree"]


@dataclass(frozen=True)
class SemanticSelector:
    """Resolve one or more MuJoCo objects by semantic names or body subtrees."""

    mode: SelectorMode = "names"
    names: tuple[str, ...] = ()
    pattern: str | None = None

    def __post_init__(self) -> None:
        # Match mjlab's ContactMatch spelling while retaining the original
        # selector names used by the Torch task configs.
        if self.mode == "body" or self.mode == "subtree":
            object.__setattr__(self, "mode", "body_subtree")
        if self.mode == "names" and not self.names:
            raise ValueError("A names selector requires at least one name")
        if self.mode in {"regex", "body_subtree"} and not self.pattern:
            raise ValueError(f"A {self.mode} selector requires a pattern")


@dataclass(frozen=True)
class EntityInitStateCfg:
    """Upstream-style per-entity reset state."""

    pos: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None
    joint_pos: dict[str, float] = field(default_factory=dict)
    joint_vel: dict[str, float] = field(default_factory=dict)
    linear_velocity: tuple[float, float, float] | None = None
    angular_velocity: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class EntityCfg:
    """One scene entity and the semantic handles required by task terms."""

    name: str
    xml_path: Path
    scene_xml_path: Path | None = None
    kind: str = "robot"
    keyframe_name: str | None = None
    root_body_name: str | None = None
    trunk_body_name: str | None = None
    head_body_names: tuple[str, ...] = ()
    foot_site_selector: SemanticSelector | None = None
    foot_contact_selectors: tuple[SemanticSelector, SemanticSelector] | None = None
    collision_name_suffix: str | None = None
    actuator_mode: Literal["bam", "xml"] = "xml"
    actuator_joint_names: tuple[str, ...] = ()
    # Spawn transforms are applied by the scene composer to the entity root
    # body.  Task reset terms may still override dynamic qpos/qvel afterward.
    spawn_pos: tuple[float, float, float] | None = None
    spawn_quat: tuple[float, float, float, float] | None = None
    init_state: EntityInitStateCfg = field(default_factory=EntityInitStateCfg)
    # Upstream's entity boundary is a pre-attach MjSpec mutation hook.  It is
    # deliberately typed as a callable rather than a MicroDuck-specific
    # editor so articulated robots, props, and fixed obstacles share it.
    spec_fn: Callable[..., Any] | None = None

    @property
    def load_path(self) -> Path:
        """Return the compatibility scene wrapper when one is provided."""

        return (self.scene_xml_path or self.xml_path).resolve()


@dataclass
class TerrainGeometry:
    """One MuJoCo geometry returned by a sub-terrain generator."""

    geom: Any | None = None
    hfield: Any | None = None
    color: tuple[float, float, float, float] | None = None


@dataclass
class TerrainOutput:
    """Typed result of an upstream-style terrain generation pass.

    Upstream sub-terrain functions return one ``TerrainOutput`` per patch,
    while the generator assembles those patches into a grid.  The Torch scene
    boundary accepts both forms: ``origin``/``geometries`` describe one
    generated patch and ``origins``/``types`` describe the complete runtime
    assignment table.  This keeps geometry generation and reset placement
    coupled without making ``TerrainManager`` know how terrain was built.
    """

    origin: Any | None = None
    geometries: list[Any] = field(default_factory=list)
    origins: Any | None = None
    types: Any | None = None
    difficulties: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerrainCfg:
    """Terrain declaration; generators are deliberately opaque to the core."""

    kind: Literal["plane", "generator", "ramp", "apartment"] = "plane"
    generator: Any | None = None
    scene_xml: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)
    # A terrain generator may be an upstream-style object with ``compile`` or
    # ``function(difficulty, spec, rng)``, a scene-level callable returning a
    # TerrainOutput, or the legacy path-producing callable.  SceneBuilder
    # normalizes all forms through one boundary for single and composed scenes.
    spec_fn: Callable[..., Any] | None = None
    curriculum: bool = False
    max_init_level: int | None = None
    # Filled by the deterministic generator at scene materialization time.
    # Keeping the generated spawn metadata on the terrain config makes the
    # runtime origin table describe the actual support geometry rather than a
    # fabricated z=0 grid.
    generated_origins: Any | None = field(default=None, repr=False, compare=False)
    generated_types: tuple[tuple[str, ...], ...] | None = field(
        default=None, repr=False, compare=False
    )
    generated_output: TerrainOutput | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class SensorCfg:
    """Declarative first-class sensor contract.

    ``kind="mujoco"`` reads a named sensor declared in the compiled XML and
    preserves the original two-argument constructor.  The other kinds are
    resolved by ``SensorManager`` from semantic entity selectors, so task
    terms never need to know MuJoCo addresses.  ``reader`` is the explicit
    escape hatch for backend-native sensors such as a future raycaster while
    retaining the same lifecycle and history contract.
    """

    name: str
    required: bool = True
    expected_dim: int | None = None
    kind: Literal[
        "mujoco",
        "body_pose",
        "body_velocity",
        "site_position",
        "site_velocity",
        "joint_position",
        "joint_velocity",
        "contact",
        "custom",
        "raycast",
        "terrain_height",
    ] = "mujoco"
    # Builtin MuJoCo sensor declaration fields.  Keeping these on the generic
    # config lets a sensor be authored before compilation, like mjlab's
    # BuiltinSensorCfg, while ``kind="mujoco"`` still wraps an existing XML
    # sensor when ``sensor_type`` is omitted.
    sensor_type: str | None = None
    class_type: type[Any] | None = None
    object_type: str | None = None
    object_name: str | None = None
    reference_type: str | None = None
    reference_name: str | None = None
    cutoff: float = 0.0
    entity: str | None = None
    source: str | None = None
    selector: SemanticSelector | None = None
    joint_names: tuple[str, ...] = ()
    primary: SemanticSelector | None = None
    secondary: SemanticSelector | None = None
    primary_entity: str | None = None
    secondary_entity: str | None = None
    exclude: tuple[SemanticSelector, ...] = ()
    secondary_policy: Literal["first", "any", "error"] = "any"
    global_frame: bool = False
    update_period: int = 1
    debug_visualization: bool = False
    fields: tuple[str, ...] = ("found",)
    reduce: Literal["none", "mindist", "maxforce", "netforce", "any"] = "any"
    num_slots: int = 1
    track_air_time: bool = False
    reader: Any | None = None
    params: dict[str, Any] = field(default_factory=dict)
    history_length: int = 0

    @property
    def prefixed_name(self) -> str:
        """Name used by a newly authored MuJoCo sensor declaration.

        Existing XML sensors are intentionally looked up by their logical
        name.  Only sensors authored through ``sensor_type`` receive the same
        entity namespace that upstream's ``BuiltinSensorCfg`` uses.
        """

        if self.sensor_type is not None and self.entity:
            return f"{self.entity}/{self.name}"
        return self.name

    def build(self) -> Any:
        """Build the runtime sensor object without importing it at config time."""

        from .sensors import Sensor

        if self.class_type is not None:
            factory = self.class_type
            return construct(factory, self)
        return Sensor(self)


@dataclass
class SceneCfg:
    """Composable scene containing named entities, terrain, and sensors."""

    entities: dict[str, EntityCfg] = field(default_factory=dict)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    sensors: dict[str, SensorCfg] = field(default_factory=dict)
    scene_xml: Path | None = None
    contact_options: dict[str, Any] = field(default_factory=dict)
    # Upstream scene state is explicitly batched.  A scalar scene is simply
    # the B=1 specialization and keeps the existing public behavior.
    num_envs: int = 1
    env_spacing: float = 2.0
    spec_fn: Callable[..., Any] | None = None


@dataclass(frozen=True)
class SceneBuild:
    """Concrete scene source selected from a declarative ``SceneCfg``."""

    xml_path: Path
    entity_names: tuple[str, ...]
    terrain_kind: str
    composed: bool = False
    source_paths: tuple[Path, ...] = ()
    provenance: str = ""


def _copy_world_template(
    template_path: Path | None,
    entity_root_names: set[str] | None = None,
) -> ET.Element:
    """Create a world-only XML root from a scene/terrain template.

    Entity declarations are intentionally omitted because they are attached
    below through ``MjSpec.attach``.  All other scene-level sections are
    retained.  This makes a configured scene path a world template for a
    multi-entity scene, which is the same ownership split used by upstream.
    """

    if template_path is None or not template_path.is_file():
        return ET.Element("mujoco", {"model": "microduck_composed_scene"})
    root = _expand_includes(template_path.resolve())
    result = ET.Element("mujoco", {"model": "microduck_composed_scene"})
    # ``compiler`` is deliberately omitted.  Attached entity resources are
    # made absolute below and a single global meshdir cannot represent several
    # independent entity asset roots.
    copy_tags = {
        "option",
        "size",
        "visual",
        "asset",
        "worldbody",
        "contact",
        "equality",
        "tendon",
        "tuple",
        "custom",
        "statistic",
        "default",
        "exclude",
        "plugin",
        "sensor",
    }
    for child in root:
        if child.tag in copy_tags:
            copied = ET.fromstring(ET.tostring(child))
            if copied.tag == "worldbody":
                # The template is the world layer.  Its static bodies,
                # lights, and terrain are retained; configured entities are
                # attached as separate prefixed bodies below.
                roots = entity_root_names or set()
                for body in list(copied):
                    if body.tag == "body" and body.get("name") in roots:
                        copied.remove(body)
            # Entity keyframes have incompatible qpos widths once the world
            # and additional entities are assembled.  They are overlaid by
            # the model loader from each EntityCfg instead.
            if copied.tag == "keyframe":
                continue
            result.append(copied)
    _remove_entity_references(result, entity_root_names or set())
    return result


def _remove_entity_references(root: ET.Element, entity_root_names: set[str]) -> None:
    """Remove template-local references invalidated by entity attachment.

    A scene XML in the asset tree may include a robot *and* a world.  Once the
    robot is removed from the world template and re-attached with an entity
    prefix, its local contact/equality references are no longer valid.  Keep
    references whose objects still belong to the world layer and drop only
    references to removed entity roots or objects.  This is intentionally
    performed on the XML boundary, before ``MjSpec`` validates names.
    """

    worldbody = root.find("worldbody")
    world_bodies = (
        {element.get("name") for element in worldbody.iter("body") if element.get("name")}
        if worldbody is not None
        else set()
    )
    world_geoms = (
        {element.get("name") for element in worldbody.iter("geom") if element.get("name")}
        if worldbody is not None
        else set()
    )
    world_sites = (
        {element.get("name") for element in worldbody.iter("site") if element.get("name")}
        if worldbody is not None
        else set()
    )
    world_joints = (
        {element.get("name") for element in worldbody.iter("joint") if element.get("name")}
        if worldbody is not None
        else set()
    )
    world_tendons = {
        element.get("name") for element in root.findall("tendon/*") if element.get("name")
    }

    def keep_pair(element: ET.Element, left: str, right: str, names: set[str]) -> bool:
        left_name, right_name = element.get(left), element.get(right)
        return (
            left_name is None or right_name is None or (left_name in names and right_name in names)
        )

    contact = root.find("contact")
    if contact is not None:
        for element in list(contact):
            invalid_exclude = element.tag == "exclude" and not keep_pair(
                element, "body1", "body2", world_bodies
            )
            invalid_pair = element.tag == "pair" and not keep_pair(
                element, "geom1", "geom2", world_geoms
            )
            if invalid_exclude or invalid_pair:
                contact.remove(element)

    equality = root.find("equality")
    if equality is not None:
        object_sets = {
            "body1": world_bodies,
            "body2": world_bodies,
            "joint1": world_joints,
            "joint2": world_joints,
            "site1": world_sites,
            "site2": world_sites,
            "tendon1": world_tendons,
            "tendon2": world_tendons,
        }
        for element in list(equality):
            if any(
                attribute in element.attrib and element.get(attribute) not in object_sets[attribute]
                for attribute in object_sets
            ):
                equality.remove(element)

    # A full scene template commonly includes the entity's XML sensor block.
    # The entity is removed from the world layer and reattached with a prefix,
    # so keep only sensors whose object references still belong to the world.
    sensors = root.find("sensor")
    if sensors is not None:
        reference_attributes = {
            "objname",
            "refname",
            "site",
            "body",
            "joint",
            "tendon",
            "actuator",
            "geom",
            "camera",
            "xbody",
            "pair",
        }
        valid_names = world_bodies | world_geoms | world_sites | world_joints | world_tendons
        for element in list(sensors):
            if any(
                attribute in element.attrib
                and element.get(attribute)
                and element.get(attribute) not in valid_names
                for attribute in reference_attributes
            ):
                sensors.remove(element)

    # ``entity_root_names`` is part of the signature to make the ownership
    # boundary explicit and to catch a template that accidentally keeps a
    # removed root body as a static world body.
    del entity_root_names


def _expand_includes(path: Path, _stack: tuple[Path, ...] = ()) -> ET.Element:
    """Recursively expand MJCF includes while resolving resource paths.

    ``MjSpec.from_file`` expands includes but also validates keyframes before
    entities are attached.  World templates such as the apartment contain a
    robot include and therefore have an intentionally different qpos width.
    Expanding the XML first lets us remove the embedded entity and preserve the
    world includes without ever asking MuJoCo to compile the intermediate
    invalid keyframe set.
    """

    path = path.resolve()
    if path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"Cyclic MJCF include detected: {chain}")
    root = ET.parse(path).getroot()
    expanded_children: list[ET.Element] = []
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
    assetdir = Path(compiler.get("assetdir", "")) if compiler is not None else Path()
    for child in list(root):
        if child.tag == "include":
            include_path = Path(child.attrib["file"])
            if not include_path.is_absolute():
                include_path = path.parent / include_path
            included = _expand_includes(include_path, (*_stack, path))
            expanded_children.extend(list(included))
            continue
        copied = ET.fromstring(ET.tostring(child))
        for element in copied.iter():
            if element.tag in {"mesh", "texture", "hfield"} and "file" in element.attrib:
                resource = Path(element.attrib["file"])
                if not resource.is_absolute():
                    base = meshdir if element.tag == "mesh" else assetdir
                    candidate = (path.parent / base / resource).resolve()
                    if candidate.is_file():
                        element.set("file", str(candidate))
        expanded_children.append(copied)
    result = ET.Element(root.tag, dict(root.attrib))
    result.extend(expanded_children)
    return result


def _run_spec_hook(hook: Callable[..., Any] | None, spec: Any, cfg: Any) -> Any:
    """Run an entity/scene mutation hook with explicit supported signatures."""

    if hook is None:
        return
    result = invoke_compatible(
        hook,
        (
            ((), {}),
            ((spec, cfg), {}),
            ((spec,), {}),
            ((), {"spec": spec, "cfg": cfg}),
            ((), {"spec": spec}),
        ),
    )
    # A zero-argument upstream ``spec_fn`` is a factory.  Entity callers use
    # the returned spec explicitly; mutation hooks conventionally return
    # ``None`` and mutate the passed object in place.
    return result


def _apply_spec_hook(spec: Any, hook: Callable[..., Any] | None, cfg: Any, label: str) -> Any:
    """Apply a scene mutation hook and honor an explicit replacement spec."""

    result = _run_spec_hook(hook, spec, cfg)
    if result is None:
        return spec
    import mujoco

    if not isinstance(result, mujoco.MjSpec):
        raise TypeError(
            f"{label} spec_fn returned {type(result).__name__}; expected mujoco.MjSpec or None"
        )
    return result


def _is_scene_path_generator(generator: Any) -> bool:
    """Identify the explicitly supported legacy ``generator(config)`` form."""

    if not callable(generator):
        return False
    try:
        parameters = tuple(inspect.signature(generator).parameters.values())
    except (TypeError, ValueError):
        return False
    required = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        and parameter.default is inspect.Parameter.empty
    )
    return len(required) == 1 and required[0].name in {"config", "cfg"}


def _apply_terrain_generator(spec: Any, config: SceneCfg) -> Any:
    """Apply one generic terrain generator to an assembled scene.

    This is the single terrain extension boundary used by both one-entity and
    multi-entity scenes.  It accepts mjlab's ``compile(spec)`` generator,
    per-patch ``function(difficulty, spec, rng)`` generators, or a callable
    scene mutator.  Legacy path generators are intentionally rejected here:
    they must be resolved before composition because their result is a whole
    scene, not a terrain output.
    """

    generator = config.terrain.generator
    if generator is None:
        return spec
    if _is_scene_path_generator(generator):
        raise TypeError(
            "A generator(config) path factory cannot run after entity composition; "
            "use an upstream-style compile(spec) or function(difficulty, spec, rng) generator"
        )
    if callable(getattr(generator, "compile", None)):
        result = generator.compile(spec)
    elif getattr(generator, "sub_terrains", None) or getattr(
        getattr(generator, "cfg", None), "sub_terrains", None
    ):
        _compile_terrain_grid(generator, spec, config)
        result = None
    elif callable(getattr(generator, "function", None)):
        result = invoke_compatible(
            generator.function,
            (
                ((0.0, spec, random.Random(config.terrain.params.get("seed", 0))), {}),
                ((spec,), {}),
            ),
        )
    elif callable(generator):
        result = invoke_compatible(
            generator,
            (
                ((0.0, spec, random.Random(config.terrain.params.get("seed", 0))), {}),
                ((spec, config), {}),
                ((spec,), {}),
            ),
        )
    else:
        raise TypeError(
            "Terrain generator must implement compile(spec), function(difficulty, spec, rng), "
            "or be a callable scene mutator"
        )
    import mujoco

    if isinstance(result, mujoco.MjSpec):
        spec = result
    elif isinstance(result, TerrainOutput):
        _record_terrain_output(config, result)
    elif result is not None:
        raise TypeError(
            f"Terrain generator returned {type(result).__name__}; expected TerrainOutput, "
            "MjSpec, or None"
        )
    # mjlab's grid generator stores assignment metadata on the generator
    # object while mutating the spec. Normalize it into the same typed output.
    if config.terrain.generated_output is None:
        origins = getattr(generator, "terrain_origins", None)
        if origins is not None:
            output = TerrainOutput(
                origins=origins,
                types=getattr(generator, "terrain_types", None),
                difficulties=getattr(generator, "terrain_difficulties", None),
            )
            config.terrain.generated_output = output
            config.terrain.generated_origins = origins
            if output.types is not None:
                config.terrain.generated_types = tuple(tuple(row) for row in output.types)
    return spec


def _record_terrain_output(config: SceneCfg, output: TerrainOutput) -> None:
    """Normalize one patch or a complete grid into the runtime terrain table."""

    config.terrain.generated_output = output
    origins = output.origins
    if origins is None and output.origin is not None:
        origin = np.asarray(output.origin, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError(
                "TerrainOutput.origin must contain exactly three coordinates, "
                f"got shape {origin.shape}"
            )
        origins = origin.reshape(1, 1, 3)
    if origins is not None:
        origin_array = np.asarray(origins, dtype=np.float64)
        if origin_array.shape == (3,):
            origin_array = origin_array.reshape(1, 1, 3)
        elif origin_array.ndim == 2 and origin_array.shape[-1] == 3:
            origin_array = origin_array.reshape(1, *origin_array.shape)
        if origin_array.ndim != 3 or origin_array.shape[-1] != 3:
            raise ValueError(
                f"TerrainOutput.origins must have shape (rows, cols, 3), got {origin_array.shape}"
            )
        config.terrain.generated_origins = tuple(
            tuple(tuple(float(value) for value in origin) for origin in row) for row in origin_array
        )
    if output.types is not None:
        types = output.types
        if isinstance(types, str):
            normalized_types = ((types,),)
        else:
            type_array = np.asarray(types, dtype=object)
            if type_array.ndim == 0:
                normalized_types = ((str(type_array.item()),),)
            elif type_array.ndim == 1:
                normalized_types = (tuple(str(value) for value in type_array),)
            else:
                normalized_types = tuple(
                    tuple(str(value) for value in row) for row in type_array.tolist()
                )
        config.terrain.generated_types = normalized_types


def _compile_terrain_grid(generator: Any, spec: Any, config: SceneCfg) -> None:
    """Run upstream-style ``SubTerrainCfg.function`` over a terrain grid."""

    generator_cfg = getattr(generator, "cfg", generator)
    sub_terrains = getattr(generator_cfg, "sub_terrains", None)
    if not sub_terrains:
        raise ValueError("Terrain grid generator requires a non-empty sub_terrains mapping")
    sub_items = tuple(sub_terrains.items())
    curriculum = bool(getattr(generator_cfg, "curriculum", False))
    rows = int(getattr(generator_cfg, "num_rows", config.terrain.params.get("rows", 1)))
    cols = int(
        len(sub_items)
        if curriculum
        else getattr(generator_cfg, "num_cols", config.terrain.params.get("cols", 1))
    )
    if rows < 1 or cols < 1:
        raise ValueError("Terrain generator rows and columns must be positive")
    size = tuple(
        float(value)
        for value in getattr(generator_cfg, "size", config.terrain.params.get("size", (8.0, 8.0)))
    )
    if len(size) != 2 or any(value <= 0 for value in size):
        raise ValueError("Terrain generator size must contain two positive values")
    seed = getattr(generator_cfg, "seed", config.terrain.params.get("seed", 0))
    rng = np.random.default_rng(seed)
    difficulty_range = tuple(getattr(generator_cfg, "difficulty_range", (0.0, 1.0)))
    if len(difficulty_range) != 2:
        raise ValueError("Terrain difficulty_range must contain two values")
    origin_table = np.zeros((rows, cols, 3), dtype=np.float64)
    type_table: list[list[str]] = [["" for _ in range(cols)] for _ in range(rows)]
    difficulty_table = np.zeros(rows, dtype=np.float64)
    body = None
    with suppress(KeyError, ValueError):
        body = spec.body("terrain")
    if body is None:
        body = spec.worldbody.add_body(name="terrain")
    del body  # sub-terrain functions resolve the shared body by its stable name
    proportions = np.asarray(
        [float(getattr(sub_cfg, "proportion", 1.0)) for _, sub_cfg in sub_items],
        dtype=np.float64,
    )
    if (proportions < 0).any() or float(proportions.sum()) <= 0:
        raise ValueError("Terrain proportions must be non-negative and non-zero")
    probabilities = proportions / proportions.sum()
    lower, upper = (float(value) for value in difficulty_range)
    for row in range(rows):
        row_fraction = (row + float(rng.uniform())) / rows
        difficulty = lower + (upper - lower) * row_fraction
        difficulty_table[row] = difficulty
        for col in range(cols):
            if curriculum:
                name, sub_cfg = sub_items[col]
            else:
                index = int(rng.choice(len(sub_items), p=probabilities))
                name, sub_cfg = sub_items[index]
            function = getattr(sub_cfg, "function", None)
            if not callable(function):
                raise TypeError(f"Terrain sub-terrain {name!r} has no callable function")
            output = invoke_compatible(
                function,
                (((difficulty, spec, rng), {}), ((spec,), {})),
            )
            if not isinstance(output, TerrainOutput):
                raise TypeError(
                    f"Terrain sub-terrain {name!r} returned {type(output).__name__}; "
                    "expected TerrainOutput"
                )
            patch_origin = np.asarray(output.origin, dtype=np.float64)
            if patch_origin.shape != (3,):
                raise ValueError(f"Terrain sub-terrain {name!r} returned an invalid origin")
            world_position = np.asarray(
                (
                    row * size[0] - rows * size[0] / 2.0,
                    col * size[1] - cols * size[1] / 2.0,
                    0.0,
                ),
                dtype=np.float64,
            )
            for geometry in output.geometries:
                geom = getattr(geometry, "geom", geometry)
                if geom is not None and hasattr(geom, "pos"):
                    geom.pos = np.asarray(geom.pos, dtype=np.float64) + world_position
                if (
                    getattr(geometry, "color", None) is not None
                    and geom is not None
                    and hasattr(geom, "rgba")
                ):
                    geom.rgba = geometry.color
            origin_table[row, col] = patch_origin + world_position
            type_table[row][col] = str(name)
    output = TerrainOutput(
        origins=origin_table,
        types=type_table,
        difficulties=difficulty_table,
        metadata={"size": size, "curriculum": curriculum},
    )
    _record_terrain_output(config, output)


def _source_manifest(path: Path, seen: set[Path] | None = None) -> list[tuple[str, str]]:
    """Hash XML includes and referenced resources for a safe composition cache."""

    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        return []
    if not path.is_file():
        raise FileNotFoundError(path)
    seen.add(path)
    result = [(str(path), hashlib.sha256(path.read_bytes()).hexdigest())]
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return result
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
    assetdir = Path(compiler.get("assetdir", "")) if compiler is not None else Path()
    for include in root.iter("include"):
        include_path = Path(include.attrib["file"])
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        result.extend(_source_manifest(include_path, seen))
    for element in root.iter():
        if element.tag not in {"mesh", "texture", "hfield"} or "file" not in element.attrib:
            continue
        resource = Path(element.attrib["file"])
        if not resource.is_absolute():
            resource = path.parent / (meshdir if element.tag == "mesh" else assetdir) / resource
        result.extend(_source_manifest(resource, seen))
    return result


def _hook_identity(hook: Any) -> str | None:
    """Return a stable cache identity for a callable mutation boundary."""

    if hook is None:
        return None
    code = getattr(hook, "__code__", None)
    if code is not None:
        closure_values = ()
        closure = getattr(hook, "__closure__", None)
        if closure is not None:
            # Closure/default values are part of a mutation hook's behavior.
            # Use repr rather than serializing arbitrary user objects, while
            # keeping the cache key deterministic for normal config values.
            closure_values = tuple(repr(cell.cell_contents) for cell in closure)
        payload = repr(
            (
                code.co_code,
                code.co_consts,
                getattr(hook, "__defaults__", None),
                getattr(hook, "__kwdefaults__", None),
                closure_values,
            )
        ).encode()
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return f"{hook.__module__}.{hook.__qualname__}:{digest}"
    return f"{type(hook).__module__}.{type(hook).__qualname__}:{repr(hook)}"


def _normalize_spec_resources(spec: Any, source_path: Path) -> None:
    """Make entity/template assets independent of the generated XML location."""

    compiler = spec.compiler
    meshdir = Path(getattr(compiler, "meshdir", "") or "")
    assetdir = Path(getattr(compiler, "assetdir", "") or "")
    for mesh in spec.meshes:
        if mesh.file and not Path(mesh.file).is_absolute():
            mesh.file = str((source_path.parent / meshdir / mesh.file).resolve())
    for collection, _tag in ((spec.textures, "texture"), (spec.hfields, "hfield")):
        for item in collection:
            if item.file and not Path(item.file).is_absolute():
                item.file = str((source_path.parent / assetdir / item.file).resolve())


def _normalize_scene_resources(spec: Any, source_paths: tuple[Path, ...]) -> None:
    """Normalize resources in a legacy scene whose includes were expanded."""

    candidates: list[Path] = []
    for source_path in source_paths:
        root = ET.parse(source_path).getroot()
        compiler = root.find("compiler")
        meshdir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
        assetdir = Path(compiler.get("assetdir", "")) if compiler is not None else Path()
        candidates.extend((source_path.parent / meshdir, source_path.parent / assetdir))
        candidates.append(source_path.parent)
    for mesh in spec.meshes:
        if mesh.file and not Path(mesh.file).is_absolute():
            match = next(
                (base / mesh.file for base in candidates if (base / mesh.file).is_file()), None
            )
            if match is not None:
                mesh.file = str(match.resolve())
    for collection in (spec.textures, spec.hfields):
        for item in collection:
            if item.file and not Path(item.file).is_absolute():
                match = next(
                    (base / item.file for base in candidates if (base / item.file).is_file()), None
                )
                if match is not None:
                    item.file = str(match.resolve())


def _copy_entity_actuators(scene_spec: Any, source_spec: Any, entity_name: str) -> None:
    """Copy entity actuators into the assembled global actuator namespace.

    ``MjSpec.attach`` intentionally attaches the body subtree and child
    objects, but MuJoCo actuators live in the global top-level ``actuator``
    section and are not transferred by that operation.  Treating them as an
    entity-owned resource here preserves XML actuator models as well as BAM's
    later actuator conversion.  References and default classes are rewritten
    to the same namespace used by ``attach``.
    """

    array_fields = (
        "gainprm",
        "biasprm",
        "dynprm",
        "velrange",
        "ffrange",
        "gear",
        "lengthrange",
        "ctrlrange",
        "forcerange",
        "actrange",
        "userdata",
    )
    scalar_fields = (
        "gaintype",
        "biastype",
        "dyntype",
        "actdim",
        "ctrlspec",
        "actearly",
        "cranklength",
        "inheritrange",
        "damping",
        "armature",
        "ctrllimited",
        "forcelimited",
        "actlimited",
        "group",
        "nsample",
        "interp",
        "delay",
    )
    reference_fields = {"refsite", "slidersite"}
    prefix = f"{entity_name}/"
    existing_names = {item.name for item in scene_spec.actuators}
    for source_actuator in source_spec.actuators:
        target = source_actuator.target
        target_name = target if isinstance(target, str) else getattr(target, "name", "")
        if not target_name:
            raise ValueError(
                f"Entity {entity_name!r} has actuator {source_actuator.name!r} without a target"
            )
        target_name = target_name.removeprefix(prefix)
        target_name = f"{prefix}{target_name}"
        source_class = getattr(source_actuator, "classname", None)
        class_name = getattr(source_class, "name", "") if source_class is not None else ""
        class_name = class_name.removeprefix(prefix)
        default = scene_spec.find_default(f"{entity_name}/{class_name}") if class_name else None
        actuator_name = (
            f"{prefix}{source_actuator.name.removeprefix(prefix)}" if source_actuator.name else None
        )
        # Recent MuJoCo versions transfer top-level actuators when the source
        # keyframes are removed; older versions do not.  Make the fallback
        # idempotent so both behaviors produce one global actuator per source.
        if actuator_name is not None and actuator_name in existing_names:
            continue
        actuator = scene_spec.add_actuator(
            default=default,
            name=actuator_name,
            trntype=int(source_actuator.trntype),
            target=target_name,
        )
        if actuator_name is not None:
            existing_names.add(actuator_name)
        for field_name in array_fields:
            value = getattr(source_actuator, field_name, None)
            if value is not None:
                setattr(actuator, field_name, value)
        for field_name in scalar_fields:
            value = getattr(source_actuator, field_name, None)
            if value is not None:
                setattr(actuator, field_name, value)
        for field_name in reference_fields:
            value = getattr(source_actuator, field_name, "")
            if value:
                setattr(actuator, field_name, f"{prefix}{value.removeprefix(prefix)}")


def _spec_object_type_and_names(
    spec: Any, selector: SemanticSelector
) -> tuple[Any, tuple[str, ...]]:
    """Resolve a local selector for a contact sensor before scene compilation."""

    import mujoco

    def named_objects(kind: str) -> tuple[str, ...]:
        """Return recursively nested MjSpec object names.

        ``MjSpec.geoms``/``MjSpec.sites`` are not a reliable flat index for
        objects nested below ``worldbody`` on every supported MuJoCo build.
        Upstream resolves selectors against the entity's complete object
        graph, so do the same here and use the flat collection only as a
        compatibility supplement.
        """

        result: list[str] = []
        seen: set[str] = set()

        def add(value: Any) -> None:
            name = getattr(value, "name", None)
            if name and name not in seen:
                seen.add(name)
                result.append(str(name))

        def walk_body(body: Any) -> None:
            add(body)
            for geom in getattr(body, "geoms", ()):
                if kind == "geom":
                    add(geom)
            for site in getattr(body, "sites", ()):
                if kind == "site":
                    add(site)
            for joint in getattr(body, "joints", ()):
                if kind == "joint":
                    add(joint)
            for child in getattr(body, "bodies", ()):
                walk_body(child)

        worldbody = getattr(spec, "worldbody", None)
        if worldbody is not None:
            walk_body(worldbody)
        collection = getattr(spec, f"{kind}s", ())
        for value in collection:
            if kind == "body" and getattr(value, "name", None) == "world":
                continue
            add(value)
        return tuple(name for name in result if name != "world")

    if selector.mode == "body_subtree":
        pattern = __import__("re").compile(selector.pattern or "")
        names = tuple(name for name in named_objects("body") if pattern.search(name))
        if not names:
            raise ValueError(f"Selector {selector!r} matched no entity bodies")
        return mujoco.mjtObj.mjOBJ_XBODY, names
    import re

    pattern = re.compile(selector.pattern) if selector.mode == "regex" else None
    geoms = named_objects("geom")
    names = (
        tuple(name for name in geoms if pattern.search(name))
        if pattern is not None
        else tuple(name for name in selector.names if name in geoms)
    )
    if names:
        return mujoco.mjtObj.mjOBJ_GEOM, names
    bodies = named_objects("body")
    names = (
        tuple(name for name in bodies if pattern.search(name))
        if pattern is not None
        else tuple(name for name in selector.names if name in bodies)
    )
    if names:
        return mujoco.mjtObj.mjOBJ_BODY, names
    raise ValueError(f"Selector {selector!r} matched no entity geoms or bodies")


_CONTACT_FIELD_BITS = {
    "found": 0,
    "force": 1,
    "torque": 2,
    "dist": 3,
    "pos": 4,
    "normal": 5,
    "tangent": 6,
}
_CONTACT_REDUCTIONS = {"none": 0, "mindist": 1, "maxforce": 2, "netforce": 3, "any": 2}


def _add_configured_sensors(
    spec: Any,
    config: SceneCfg,
    source_specs: dict[str, Any],
    *,
    prefix_entities: bool,
) -> None:
    """Add declarative builtin/contact sensors to an already assembled spec."""

    import mujoco

    existing = {sensor.name for sensor in spec.sensors}
    for sensor_cfg in config.sensors.values():
        # An existing XML sensor is wrapped by SensorManager and must not be
        # duplicated.  A builtin declaration with a sensor_type is authored at
        # this scene boundary, just like mjlab's BuiltinSensorCfg.edit_spec.
        if sensor_cfg.kind == "mujoco" and sensor_cfg.sensor_type is not None:
            sensor_name = sensor_cfg.name if not prefix_entities else sensor_cfg.prefixed_name
            if sensor_name in existing:
                raise ValueError(f"Sensor {sensor_name!r} is defined twice")
            kwargs: dict[str, Any] = {
                "name": sensor_name,
                "type": getattr(mujoco.mjtSensor, f"mjSENS_{sensor_cfg.sensor_type.upper()}"),
            }
            if sensor_cfg.object_type is not None:
                kwargs["objtype"] = getattr(
                    mujoco.mjtObj, f"mjOBJ_{sensor_cfg.object_type.upper()}"
                )
                object_name = sensor_cfg.object_name
                if prefix_entities and sensor_cfg.entity and object_name is not None:
                    object_name = f"{sensor_cfg.entity}/{object_name}"
                kwargs["objname"] = object_name
            if sensor_cfg.reference_type is not None:
                kwargs["reftype"] = getattr(
                    mujoco.mjtObj, f"mjOBJ_{sensor_cfg.reference_type.upper()}"
                )
                reference_name = sensor_cfg.reference_name
                if prefix_entities and sensor_cfg.secondary_entity and reference_name is not None:
                    reference_name = f"{sensor_cfg.secondary_entity}/{reference_name}"
                kwargs["refname"] = reference_name
            if sensor_cfg.cutoff > 0:
                kwargs["cutoff"] = sensor_cfg.cutoff
            spec.add_sensor(**kwargs)
            existing.add(sensor_name)
            continue
        if sensor_cfg.kind != "contact":
            continue
        entity_name = sensor_cfg.primary_entity or sensor_cfg.entity
        if entity_name is None:
            # A sensor attached to a single-entity scene can use the same
            # implicit primary convention as the runtime SensorManager.  In a
            # composed scene, silently guessing would make a contact sensor
            # depend on insertion order, so require the author to scope it.
            if len(source_specs) == 1:
                entity_name = next(iter(source_specs))
            else:
                raise ValueError(
                    f"Contact sensor {sensor_cfg.name!r} requires primary_entity "
                    "when the scene contains multiple entities"
                )
        source_spec = source_specs.get(entity_name)
        if source_spec is None:
            raise KeyError(f"No source spec for contact entity {entity_name!r}")
        primary_type, primary_names = _spec_object_type_and_names(
            source_spec,
            sensor_cfg.primary,  # type: ignore[arg-type]
        )
        secondary_type = secondary_name = None
        secondary_names: tuple[str, ...] = ()
        secondary_entity_name: str | None = None
        if sensor_cfg.secondary is not None:
            secondary_entity = sensor_cfg.secondary_entity
            if secondary_entity is not None:
                secondary_entity_name = secondary_entity
                secondary_spec = source_specs[secondary_entity]
                secondary_type, secondary_names = _spec_object_type_and_names(
                    secondary_spec, sensor_cfg.secondary
                )
                # MuJoCo's native contact sensor accepts one reference object.
                # For a selector that intentionally matches several objects,
                # leave the reference unset and let SensorManager apply the
                # complete semantic filter over the contact graph.
                if len(secondary_names) == 1:
                    secondary_name = secondary_names[0]
            else:
                # World selectors are resolved against the assembled scene.
                secondary_type, secondary_names = _spec_object_type_and_names(
                    spec, sensor_cfg.secondary
                )
                if len(secondary_names) == 1:
                    secondary_name = secondary_names[0]
        for primary_index, primary_name in enumerate(primary_names):
            for field_name in sensor_cfg.fields:
                try:
                    bit = _CONTACT_FIELD_BITS[field_name]
                    reduce = _CONTACT_REDUCTIONS[sensor_cfg.reduce]
                except KeyError as exc:
                    raise ValueError(
                        f"Unsupported contact field/reduction in {sensor_cfg.name!r}"
                    ) from exc
                internal_name = f"__contact__{sensor_cfg.name}__{primary_index}__{field_name}"
                kwargs = {
                    "name": internal_name,
                    "type": mujoco.mjtSensor.mjSENS_CONTACT,
                    "objtype": primary_type,
                    "objname": (
                        f"{entity_name}/{primary_name}"
                        if prefix_entities and entity_name
                        else primary_name
                    ),
                    "intprm": [1 << bit, reduce, sensor_cfg.num_slots],
                }
                if secondary_name is not None:
                    kwargs["reftype"] = secondary_type
                    kwargs["refname"] = (
                        f"{secondary_entity_name}/{secondary_name}"
                        if prefix_entities and secondary_entity_name is not None
                        else secondary_name
                    )
                spec.add_sensor(**kwargs)


def _edit_custom_sensor_specs(spec: Any, config: SceneCfg) -> None:
    """Run first-class custom sensor spec hooks before compilation."""

    for sensor_cfg in config.sensors.values():
        if (
            sensor_cfg.kind not in {"custom", "raycast", "terrain_height"}
            and sensor_cfg.class_type is None
        ):
            continue
        sensor = sensor_cfg.build()
        edit_spec = getattr(sensor, "edit_spec", None)
        if callable(edit_spec):
            edit_spec(spec, config.entities)


def _compose_entity_scene(config: SceneCfg) -> tuple[Path, tuple[Path, ...]]:
    """Materialize a deterministic multi-entity MuJoCo include scene.

    mjlab composes entity spawns into its scene importer.  The Torch backend
    has one compiled XML boundary, so the equivalent is a generated wrapper
    containing one include per entity plus copied world/terrain declarations.
    This is a real composition operation: no task is required to maintain a
    hand-written combined XML merely because it has two entities.
    """

    entity_items = tuple(config.entities.items())
    if not entity_items:
        raise ValueError("A scene must contain at least one entity")

    def root_name(entity_cfg: EntityCfg) -> str:
        configured = entity_cfg.root_body_name or entity_cfg.trunk_body_name
        if configured is not None:
            return configured
        root = _expand_includes(entity_cfg.xml_path.resolve()).find("worldbody/body")
        if root is None or root.get("name") is None:
            raise ValueError(
                f"Entity {entity_cfg.name!r} has no discoverable world root; "
                "set EntityCfg.root_body_name"
            )
        return str(root.get("name"))

    entity_root_names = {name: root_name(entity_cfg) for name, entity_cfg in entity_items}
    entity_paths = tuple(entity.xml_path.resolve() for _, entity in entity_items)
    template = config.scene_xml
    if template is None:
        template = config.terrain.scene_xml
    if template is None:
        template = next(
            (entity.scene_xml_path for _, entity in entity_items if entity.scene_xml_path),
            None,
        )
    template = template.resolve() if template is not None else None
    manifests: list[tuple[str, str]] = []
    for path in entity_paths:
        manifests.extend(_source_manifest(path))
    if template is not None:
        manifests.extend(_source_manifest(template))
    config_digest = {
        "entities": [
            {
                "name": name,
                "xml": str(path),
                "root_body": entity_root_names[name],
                "keyframe": entity_cfg.keyframe_name,
                "spawn_pos": entity_cfg.spawn_pos or entity_cfg.init_state.pos,
                "spawn_quat": entity_cfg.spawn_quat or entity_cfg.init_state.quat,
                "init_state": repr(entity_cfg.init_state),
                "spec_fn": _hook_identity(entity_cfg.spec_fn),
            }
            for (name, entity_cfg), path in zip(entity_items, entity_paths, strict=True)
        ],
        "template": str(template) if template else None,
        "terrain": {
            "repr": repr(config.terrain),
            "spec_fn": _hook_identity(config.terrain.spec_fn),
            "generator": _hook_identity(config.terrain.generator),
        },
        "scene_spec_fn": _hook_identity(config.spec_fn),
        "contact_options": repr(config.contact_options),
        "sensors": repr(tuple(config.sensors.items())),
        "manifest": sorted(set(manifests)),
    }
    digest = hashlib.sha256(
        b"microduck-composed-scene-v9-generic-terrain\0"
        + json.dumps(config_digest, sort_keys=True).encode()
    ).hexdigest()[:16]
    output_dir = Path(tempfile.gettempdir()) / "microduck_rl_torch" / "scenes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"composed_{digest}.xml"
    if output_path.is_file() and not (
        config.terrain.kind == "generator"
        and config.terrain.generator is not None
        and not _is_scene_path_generator(config.terrain.generator)
    ):
        if config.terrain.kind == "generator":
            origins, type_names = _rough_patch_metadata(config)
            config.terrain.generated_origins = tuple(tuple(item) for item in origins)
            config.terrain.generated_types = tuple(tuple(item) for item in type_names)
        elif config.terrain.kind == "ramp":
            params = config.terrain.params
            angle = float(params.get("angle", params.get("max_angle", 0.25)))
            flat_length = float(params.get("flat_length", 2.0))
            spawn_on_ramp = float(params.get("spawn_on_ramp", 0.3))
            config.terrain.generated_origins = (
                ((flat_length + spawn_on_ramp, 0.0, -spawn_on_ramp * math.tan(angle)),),
            )
            config.terrain.generated_types = (("ramp",),)
        return output_path, entity_paths

    # MjSpec.attach is the important part of this implementation.  It applies
    # the same prefix to every entity-local name (bodies, joints, geoms,
    # sites, sensors, actuators, materials, defaults, and references), so two
    # copies of the same asset remain independent.
    import mujoco

    world_root = _copy_world_template(
        template,
        set(entity_root_names.values()),
    )
    scene_spec = mujoco.MjSpec.from_string(ET.tostring(world_root, encoding="unicode"))
    source_specs: dict[str, Any] = {}
    for entity_name, entity_cfg in entity_items:
        entity_spec = mujoco.MjSpec.from_file(str(entity_cfg.xml_path.resolve()))
        _normalize_spec_resources(entity_spec, entity_cfg.xml_path.resolve())
        produced_spec = _run_spec_hook(entity_cfg.spec_fn, entity_spec, entity_cfg)
        if produced_spec is not None:
            if not isinstance(produced_spec, mujoco.MjSpec):
                raise TypeError(
                    f"Entity {entity_name!r} spec_fn returned {type(produced_spec).__name__}; "
                    "expected mujoco.MjSpec or None"
                )
            entity_spec = produced_spec
            _normalize_spec_resources(entity_spec, entity_cfg.xml_path.resolve())
        root_body_name = entity_root_names[entity_name]
        spawn_pos = entity_cfg.spawn_pos or entity_cfg.init_state.pos
        spawn_quat = entity_cfg.spawn_quat or entity_cfg.init_state.quat
        if spawn_pos is not None or spawn_quat is not None:
            try:
                root_body = entity_spec.body(root_body_name)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Entity {entity_name!r} spawn transform targets missing root body "
                    f"{root_body_name!r}"
                ) from exc
            if spawn_pos is not None:
                root_body.pos = spawn_pos
            if spawn_quat is not None:
                root_body.quat = spawn_quat
        # Keyframes are merged by the model loader using local joint names;
        # retaining them during attach would create duplicate/incompatible
        # global keyframes when entities have different qpos widths.
        # Keep an intact local source index for semantic sensor selectors.
        # The attach path deletes source keyframes/defaults below; using that
        # mutated spec for selector resolution loses nested object names on
        # MuJoCo builds that expose them only through body-local collections.
        source_specs[entity_name] = mujoco.MjSpec.from_string(entity_spec.to_xml())
        for key in list(entity_spec.keys):
            entity_spec.delete(key)
        frame = scene_spec.worldbody.add_frame()
        scene_spec.attach(entity_spec, prefix=f"{entity_name}/", frame=frame)
        _copy_entity_actuators(scene_spec, entity_spec, entity_name)

    if (
        config.terrain.kind == "plane"
        and not any(geom.name == "floor" for geom in scene_spec.geoms)
        and not (template is not None and config.terrain.params.get("add_plane", False) is False)
    ):
        floor = scene_spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
        floor.pos = [0.0, 0.0, 0.0]
        floor.friction = [1.0, 0.005, 0.0001]
        floor.contype = 1
        floor.conaffinity = 1
    for option_name, value in config.contact_options.items():
        if not hasattr(scene_spec.option, option_name):
            raise ValueError(f"Unknown MuJoCo option {option_name!r}")
        setattr(scene_spec.option, option_name, value)
    if config.terrain.kind == "generator":
        for geom in list(scene_spec.geoms):
            if geom.name == "floor":
                geom.delete()
        if config.terrain.generator is not None and not _is_scene_path_generator(
            config.terrain.generator
        ):
            scene_spec = _apply_terrain_generator(scene_spec, config)
        elif (
            config.terrain.generator is None
            or config.terrain.generator is make_microduck_rough_scene
        ):
            _add_rough_terrain_to_spec(scene_spec, config)
        else:
            raise TypeError(
                "A legacy terrain generator(config) returns a complete scene path and cannot "
                "be applied after multi-entity composition; implement compile(spec) or "
                "function(difficulty, spec, rng) for composed scenes"
            )
    elif config.terrain.kind == "ramp":
        for geom in list(scene_spec.geoms):
            if geom.name == "floor":
                geom.delete()
        _add_ramp_terrain_to_spec(scene_spec, config)
    scene_spec = _apply_spec_hook(scene_spec, config.terrain.spec_fn, config.terrain, "Terrain")
    scene_spec = _apply_spec_hook(scene_spec, config.spec_fn, config, "Scene")
    _add_configured_sensors(scene_spec, config, source_specs, prefix_entities=True)
    _edit_custom_sensor_specs(scene_spec, config)
    output_path.write_text(scene_spec.to_xml(), encoding="utf-8")
    return output_path, entity_paths


def _materialize_single_scene(scene_path: Path, config: SceneCfg) -> Path:
    """Add configured sensors to one legacy, unprefixed scene wrapper."""

    import mujoco

    scene_path = scene_path.resolve()
    source_paths = tuple(entity.xml_path.resolve() for entity in config.entities.values())
    digest = hashlib.sha256(
        b"microduck-single-scene-v3\0"
        + json.dumps(
            {
                "scene": _source_manifest(scene_path),
                "entities": [_source_manifest(path) for path in source_paths],
                "sensors": repr(tuple(config.sensors.items())),
                "contact_options": repr(config.contact_options),
                "entity_spec_fns": {
                    name: _hook_identity(entity.spec_fn) for name, entity in config.entities.items()
                },
                "terrain_spec_fn": _hook_identity(config.terrain.spec_fn),
                "terrain_generator": _hook_identity(config.terrain.generator),
                "terrain_params": repr(config.terrain.params),
                "scene_spec_fn": _hook_identity(config.spec_fn),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output_dir = Path(tempfile.gettempdir()) / "microduck_rl_torch" / "scenes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"single_{digest}.xml"
    if output_path.is_file() and not (
        config.terrain.kind == "generator"
        and config.terrain.generator is not None
        and not _is_scene_path_generator(config.terrain.generator)
    ):
        return output_path
    spec = mujoco.MjSpec.from_file(str(scene_path))
    source_specs = {
        name: mujoco.MjSpec.from_file(str(entity.xml_path.resolve()))
        for name, entity in config.entities.items()
    }
    _normalize_scene_resources(spec, source_paths)
    for entity_cfg in config.entities.values():
        _run_spec_hook(entity_cfg.spec_fn, spec, entity_cfg)
    if (
        config.terrain.kind == "generator"
        and config.terrain.generator is not None
        and not _is_scene_path_generator(config.terrain.generator)
    ):
        spec = _apply_terrain_generator(spec, config)
    spec = _apply_spec_hook(spec, config.terrain.spec_fn, config.terrain, "Terrain")
    spec = _apply_spec_hook(spec, config.spec_fn, config, "Scene")
    _add_configured_sensors(spec, config, source_specs, prefix_entities=False)
    _edit_custom_sensor_specs(spec, config)
    for option_name, value in config.contact_options.items():
        if not hasattr(spec.option, option_name):
            raise ValueError(f"Unknown MuJoCo option {option_name!r}")
        setattr(spec.option, option_name, value)
    output_path.write_text(spec.to_xml(), encoding="utf-8")
    return output_path


class SceneBuilder:
    """Resolve scene wrappers without leaking XML policy into task code.

    An explicit scene wrapper is a world/template layer. Entity assets are
    attached through one deterministic composition boundary, so task code only
    mutates named entities and never owns XML merging. A single entity's
    compatibility wrapper remains a valid legacy input.
    """

    def __init__(self, config: SceneCfg | None = None) -> None:
        self.config = config

    def build(self, config: SceneCfg | None = None) -> SceneBuild:
        config = config or self.config
        if config is None:
            raise ValueError("SceneBuilder requires a SceneCfg")
        if not config.entities:
            raise ValueError("A scene must contain at least one entity")
        if config.num_envs < 1:
            raise ValueError("SceneCfg.num_envs must be positive")
        entity_paths = {
            name: entity.load_path.resolve() for name, entity in config.entities.items()
        }
        composed = False
        source_paths = tuple(entity_paths.values())
        terrain = config.terrain
        # A scene XML is only a world/template layer once there is more than
        # one configured entity.  Never silently accept a hand-written XML
        # that contains a different subset of the declared entities.
        single_wrapper_mismatch = (
            len(entity_paths) == 1
            and config.scene_xml is not None
            and next(iter(config.entities.values())).scene_xml_path is not None
            and config.scene_xml.resolve()
            != next(iter(config.entities.values())).scene_xml_path.resolve()
        ) or any(entity.spec_fn is not None for entity in config.entities.values())
        if len(entity_paths) > 1 or single_wrapper_mismatch:
            xml_path, source_paths = _compose_entity_scene(config)
            composed = True
        elif callable(terrain.generator) and _is_scene_path_generator(terrain.generator):
            xml_path = Path(terrain.generator(config)).resolve()
        elif terrain.generator is not None:
            # Generic compile/function generators run against the same scene
            # wrapper for one and many entities; the latter is handled by the
            # composed path above and the former by this materializer.
            base_scene = config.scene_xml or next(iter(entity_paths.values()))
            xml_path = _materialize_single_scene(base_scene, config).resolve()
        elif terrain.scene_xml is not None:
            xml_path = terrain.scene_xml.resolve()
        elif terrain.kind == "generator":
            xml_path = make_microduck_rough_scene(config).resolve()
        elif terrain.kind == "ramp":
            xml_path = make_microduck_ramp_scene(config).resolve()
        elif config.scene_xml is not None:
            xml_path = config.scene_xml.resolve()
        elif len(entity_paths) == 1:
            xml_path = next(iter(entity_paths.values()))
        if (
            terrain.kind != "plane"
            and not composed
            and config.scene_xml is None
            and terrain.scene_xml is None
            and terrain.generator is None
        ):
            raise ValueError(
                f"Terrain kind {terrain.kind!r} requires a scene_xml, generator, or scene template"
            )
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        for name, path in entity_paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Entity {name!r} XML source does not exist: {path}")
        # For a single legacy scene wrapper, materialize newly declared
        # builtin/contact sensors without changing the source asset.  This
        # preserves the current unprefixed policy model while providing the
        # same pre-compilation sensor hook used by composed scenes.
        needs_materialization = (
            bool(
                config.sensors
                and any(
                    sensor.kind == "contact"
                    or sensor.sensor_type is not None
                    or sensor.kind == "custom"
                    and callable(getattr(sensor.reader, "edit_spec", None))
                    for sensor in config.sensors.values()
                )
            )
            or config.terrain.spec_fn is not None
            or config.spec_fn is not None
            or any(entity.spec_fn is not None for entity in config.entities.values())
        )
        if not composed and needs_materialization:
            xml_path = _materialize_single_scene(xml_path, config)
        return SceneBuild(
            xml_path=xml_path,
            entity_names=tuple(config.entities),
            terrain_kind=config.terrain.kind,
            composed=composed,
            source_paths=source_paths,
            provenance=hashlib.sha256(
                json.dumps(
                    {
                        "xml": str(xml_path),
                        "entities": [
                            (name, str(cfg.xml_path.resolve()))
                            for name, cfg in config.entities.items()
                        ],
                        "terrain": repr(config.terrain),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        )


def _terrain_parameters(config: SceneCfg) -> dict[str, Any]:
    terrain_params = config.terrain.params
    params = {
        "size": tuple(terrain_params.get("size", (8.0, 8.0))),
        "rows": int(terrain_params.get("rows", 10)),
        "cols": int(terrain_params.get("cols", 20)),
        "max_height": float(terrain_params.get("max_height", 0.015)),
        "seed": int(terrain_params.get("seed", 0)),
        "types": tuple(
            terrain_params.get("types", ("flat", "pyramid_stairs", "random_grid", "pyramid_slope"))
        ),
        "proportions": tuple(terrain_params.get("proportions", (0.25, 0.25, 0.25, 0.25))),
    }
    if len(params["size"]) != 2 or any(float(value) <= 0 for value in params["size"]):
        raise ValueError("Rough terrain size must contain two positive values")
    if params["rows"] < 1 or params["cols"] < 1:
        raise ValueError("Rough terrain rows and cols must be positive")
    if params["max_height"] < 0:
        raise ValueError("Rough terrain max_height must be non-negative")
    if not params["types"] or len(params["types"]) != len(params["proportions"]):
        raise ValueError("Terrain types and proportions must have equal non-zero length")
    if any(float(value) < 0 for value in params["proportions"]):
        raise ValueError("Terrain proportions must be non-negative")
    if sum(float(value) for value in params["proportions"]) <= 0:
        raise ValueError("Terrain proportions must contain a positive value")
    return params


def _rough_patch_metadata(
    config: SceneCfg,
) -> tuple[list[list[tuple[float, float, float]]], list[list[str]]]:
    """Compute spawn origins using the same deterministic draws as generation."""

    params = _terrain_parameters(config)
    width_x, width_y = (float(value) for value in params["size"])
    patch_x, patch_y = width_x / params["cols"], width_y / params["rows"]
    rng = random.Random(params["seed"])
    types = tuple(str(value) for value in params["types"])
    proportions = tuple(float(value) for value in params["proportions"])
    cumulative: list[tuple[float, str]] = []
    running = 0.0
    total = sum(proportions)
    for terrain_type, proportion in zip(types, proportions, strict=True):
        running += proportion / total
        cumulative.append((running, terrain_type))

    def choose_type(row: int, col: int) -> str:
        if config.terrain.curriculum:
            fraction = (col + 0.5) / params["cols"]
            return next((item for limit, item in cumulative if fraction <= limit), types[-1])
        sample = rng.random()
        return next((item for limit, item in cumulative if sample <= limit), types[-1])

    origins: list[list[tuple[float, float, float]]] = []
    type_names: list[list[str]] = []
    for row in range(params["rows"]):
        difficulty = row / max(params["rows"] - 1, 1)
        origin_row: list[tuple[float, float, float]] = []
        type_row: list[str] = []
        for col in range(params["cols"]):
            center_x = (col + 0.5) * patch_x - width_x / 2.0
            center_y = (row + 0.5) * patch_y - width_y / 2.0
            terrain_type = choose_type(row, col)
            base_height = difficulty * params["max_height"] * 0.25
            top_height = base_height
            if terrain_type == "random_grid":
                heights = [
                    rng.uniform(0.0, params["max_height"] * (0.5 + difficulty)) for _ in range(3)
                ]
                top_height += max(heights)
            elif terrain_type == "pyramid_stairs":
                top_height += params["max_height"] * difficulty
            origin_row.append((center_x, center_y, top_height))
            type_row.append(terrain_type)
        origins.append(origin_row)
        type_names.append(type_row)
    return origins, type_names


def _add_rough_terrain_to_spec(spec: Any, config: SceneCfg) -> None:
    """Attach a scalable heightfield realization to a composed scene.

    A box per patch is expensive for ``mujoco_torch`` collision
    precomputation.  A single heightfield preserves the generated patch grid,
    origins, types, and curriculum semantics while keeping composition
    bounded to one support geom.
    """

    import mujoco

    params = _terrain_parameters(config)
    origins, type_names = _rough_patch_metadata(config)
    asset_path = _rough_heightfield_asset(config, params, origins, type_names)
    hfield = spec.add_hfield()
    hfield.name = "terrain_heightfield"
    hfield.file = str(asset_path)
    hfield.size = _rough_heightfield_size(params)
    body = spec.worldbody.add_body(name="terrain")
    geom = body.add_geom(
        name="terrain_heightfield",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        contype=1,
        conaffinity=1,
    )
    geom.hfieldname = hfield.name
    if any(getattr(material, "name", None) == "groundplane" for material in spec.materials):
        geom.material = "groundplane"
    config.terrain.generated_origins = tuple(tuple(item) for item in origins)
    config.terrain.generated_types = tuple(tuple(item) for item in type_names)


def _rough_heightfield_size(params: dict[str, Any]) -> tuple[float, float, float, float]:
    width_x, width_y = (float(value) for value in params["size"])
    max_height = float(params["max_height"])
    # MuJoCo requires all four hfield size values to be positive.
    vertical_scale = max(max_height * 1.5, 1.0e-3)
    return width_x / 2.0, width_y / 2.0, vertical_scale, 1.0e-4


def _rough_heightfield_asset(
    config: SceneCfg,
    params: dict[str, Any],
    origins: list[list[tuple[float, float, float]]],
    type_names: list[list[str]],
) -> Path:
    """Write and cache a compact grayscale heightfield for rough terrain."""

    cells_x = 8
    cells_y = 8
    ncol = params["cols"] * cells_x + 1
    nrow = params["rows"] * cells_y + 1
    key = hashlib.sha256(
        b"microduck-rough-heightfield-v1\0"
        + json.dumps(
            {
                "params": params,
                "curriculum": config.terrain.curriculum,
                "origins": origins,
                "types": type_names,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output_dir = Path(tempfile.gettempdir()) / "microduck_rl_torch" / "scenes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"rough_{key}.png"
    if output.is_file():
        return output

    max_height = float(params["max_height"])
    scale = max(max_height * 1.5, 1.0e-3)
    rng = random.Random(params["seed"])
    types = tuple(str(value) for value in params["types"])
    proportions = tuple(float(value) for value in params["proportions"])
    cumulative: list[tuple[float, str]] = []
    running = 0.0
    total = sum(proportions)
    for terrain_type, proportion in zip(types, proportions, strict=True):
        running += proportion / total
        cumulative.append((running, terrain_type))

    pixels = [[0.0 for _ in range(ncol)] for _ in range(nrow)]
    for row in range(params["rows"]):
        difficulty = row / max(params["rows"] - 1, 1)
        base_height = difficulty * max_height * 0.25
        for col in range(params["cols"]):
            if config.terrain.curriculum:
                fraction = (col + 0.5) / params["cols"]
                terrain_type = next(
                    (item for limit, item in cumulative if fraction <= limit), types[-1]
                )
            else:
                sample = rng.random()
                terrain_type = next(
                    (item for limit, item in cumulative if sample <= limit), types[-1]
                )
            seed_heights = (
                [rng.uniform(0.0, max_height * (0.5 + difficulty)) for _ in range(3)]
                if terrain_type == "random_grid"
                else []
            )
            for local_y in range(cells_y + 1):
                for local_x in range(cells_x + 1):
                    fraction_x = local_x / cells_x
                    fraction_y = local_y / cells_y
                    height = base_height
                    if terrain_type == "random_grid":
                        left = seed_heights[0] * (1.0 - fraction_x) + seed_heights[1] * fraction_x
                        right = seed_heights[1] * (1.0 - fraction_x) + seed_heights[2] * fraction_x
                        height += left * (1.0 - fraction_y) + right * fraction_y
                    elif terrain_type == "pyramid_stairs":
                        height += max_height * difficulty * min(fraction_x * 3.0, 3.0) / 3.0
                    elif terrain_type == "pyramid_slope":
                        height += max_height * difficulty * fraction_x
                    pixels[row * cells_y + local_y][col * cells_x + local_x] = min(
                        255.0, max(0.0, height / scale * 255.0)
                    )
    _write_grayscale_png(output, pixels)
    return output


def _write_grayscale_png(path: Path, pixels: list[list[float]]) -> None:
    """Write an 8-bit PNG without adding an image-library dependency."""

    height = len(pixels)
    width = len(pixels[0]) if height else 0
    if width < 1 or height < 1 or any(len(row) != width for row in pixels):
        raise ValueError("Heightfield image must be a non-empty rectangular grid")
    rows = b"".join(b"\x00" + bytes(int(round(value)) for value in row) for row in pixels)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, level=9))
        + chunk(b"IEND", b"")
    )


def _add_ramp_terrain_to_spec(spec: Any, config: SceneCfg) -> None:
    """Attach an upstream-shaped flat-entry/descending-ramp terrain."""

    import mujoco

    params = config.terrain.params
    length = float(params.get("length", params.get("ramp_length", 4.0)))
    angle = float(params.get("angle", params.get("max_angle", 0.25)))
    width = float(params.get("width", 2.0))
    runout = float(params.get("runout", params.get("runout_length", 2.0)))
    thickness = float(params.get("thickness", 0.5))
    flat_length = float(params.get("flat_length", 2.0))
    spawn_on_ramp = float(params.get("spawn_on_ramp", 0.3))
    if length <= 0 or width <= 0 or runout < 0 or thickness <= 0:
        raise ValueError("Ramp dimensions must be positive and runout non-negative")
    drop = length * math.tan(angle)
    body = spec.worldbody.add_body(name="terrain")
    common = {"type": mujoco.mjtGeom.mjGEOM_BOX, "contype": 1, "conaffinity": 1}
    body.add_geom(
        name="terrain_entry",
        **common,
        size=(flat_length / 2, width / 2, thickness / 2),
        pos=(flat_length / 2, 0, -thickness / 2),
    )
    half = angle / 2
    body.add_geom(
        name="terrain_ramp",
        **common,
        size=(length / (2 * math.cos(angle)), width / 2, thickness / 2),
        pos=(
            flat_length + length / 2 - thickness * math.sin(angle) / 2,
            0,
            -drop / 2 - thickness * math.cos(angle) / 2,
        ),
        quat=(math.cos(half), 0, math.sin(half), 0),
    )
    body.add_geom(
        name="terrain_runout",
        **common,
        size=(runout / 2, width / 2, thickness / 2),
        pos=(flat_length + length + runout / 2, 0, -drop - thickness / 2),
    )
    config.terrain.generated_origins = (
        ((flat_length + spawn_on_ramp, 0.0, -spawn_on_ramp * math.tan(angle)),),
    )
    config.terrain.generated_types = (("ramp",),)


def make_microduck_rough_scene(config: SceneCfg) -> Path:
    """Build a deterministic multi-patch rough-terrain MJCF scene.

    The generated XML is intentionally a scalar-friendly realization of the
    same concepts as mjlab's ``TerrainGeneratorCfg``: every configured patch
    exists in the scene, each patch has a stable origin/type, and difficulty
    is derived from the row.  Runtime terrain selection is provided by
    :class:`TerrainManager`; this is no longer a single representative box.
    """

    if not config.entities:
        raise ValueError("A rough scene needs at least one configured entity")
    base_path = config.scene_xml
    if base_path is None:
        if len(config.entities) > 1:
            base_path, _ = _compose_entity_scene(config)
        else:
            robot = config.entities.get("robot") or next(iter(config.entities.values()))
            base_path = robot.load_path
    base_path = base_path.resolve()
    terrain_params = config.terrain.params
    params = {
        "size": tuple(terrain_params.get("size", (8.0, 8.0))),
        "rows": int(terrain_params.get("rows", 10)),
        "cols": int(terrain_params.get("cols", 20)),
        "max_height": float(terrain_params.get("max_height", 0.015)),
        "seed": int(terrain_params.get("seed", 0)),
        "types": tuple(
            terrain_params.get("types", ("flat", "pyramid_stairs", "random_grid", "pyramid_slope"))
        ),
        "proportions": tuple(terrain_params.get("proportions", (0.25, 0.25, 0.25, 0.25))),
    }
    if len(params["size"]) != 2 or any(float(value) <= 0 for value in params["size"]):
        raise ValueError("Rough terrain size must contain two positive values")
    if params["rows"] < 1 or params["cols"] < 1:
        raise ValueError("Rough terrain rows and cols must be positive")
    if params["max_height"] < 0:
        raise ValueError("Rough terrain max_height must be non-negative")
    if not params["types"] or len(params["types"]) != len(params["proportions"]):
        raise ValueError("Terrain types and proportions must have equal non-zero length")
    if any(float(value) < 0 for value in params["proportions"]):
        raise ValueError("Terrain proportions must be non-negative")
    cache_key = hashlib.sha256(
        b"microduck-rough-scene-v12-heightfield\0"
        + json.dumps(params, sort_keys=True).encode()
        + json.dumps({"curriculum": config.terrain.curriculum}, sort_keys=True).encode()
        + json.dumps(_source_manifest(base_path), sort_keys=True).encode()
    ).hexdigest()[:16]
    output_dir = Path(tempfile.gettempdir()) / "microduck_rl_torch" / "scenes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rough_{cache_key}.xml"
    origins, type_names = _rough_patch_metadata(config)
    if output_path.is_file():
        # Metadata is runtime state as well as generation bookkeeping.  The
        # XML cache must not cause TerrainManager to fall back to fabricated
        # z=0 origins on the second build in a process.
        config.terrain.generated_origins = tuple(tuple(item) for item in origins)
        config.terrain.generated_types = tuple(tuple(item) for item in type_names)
        return output_path

    root = _expand_includes(base_path)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Terrain base scene has no worldbody: {base_path}")
    # A rough-terrain scene owns the support surface.  Remove the flat floor
    # from a legacy walk scene so it cannot mask the generated height field or
    # leave the robot floating on a coplanar second collider.
    for element in list(worldbody):
        if element.tag == "geom" and element.get("name") == "floor":
            worldbody.remove(element)
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        compiler = root.find("compiler")
        insert_at = list(root).index(compiler) + 1 if compiler is not None else 0
        root.insert(insert_at, asset)
    asset_path = _rough_heightfield_asset(config, params, origins, type_names)
    size = _rough_heightfield_size(params)
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": "terrain_heightfield",
            "file": str(asset_path),
            "size": " ".join(f"{value:.9g}" for value in size),
        },
    )
    terrain_body = ET.SubElement(worldbody, "body", {"name": "terrain"})
    ET.SubElement(
        terrain_body,
        "geom",
        {
            "name": "terrain_heightfield",
            "type": "hfield",
            "hfield": "terrain_heightfield",
            "material": "groundplane",
            "contype": "1",
            "conaffinity": "1",
            "solref": "0.04 1",
            "solimp": "0.85 0.95 0.001 0.5 2",
        },
    )
    config.terrain.generated_origins = tuple(tuple(item) for item in origins)
    config.terrain.generated_types = tuple(tuple(item) for item in type_names)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path

    # The legacy per-patch realization below is retained only as historical
    # context; generated scenes return through the scalable hfield path above.
    width_x, width_y = (float(value) for value in params["size"])
    patch_x, patch_y = width_x / params["cols"], width_y / params["rows"]
    rng = random.Random(params["seed"])
    terrain_body = ET.SubElement(worldbody, "body", {"name": "terrain"})
    terrain_types = tuple(str(value) for value in params["types"])
    proportions = tuple(float(value) for value in params["proportions"])
    total = sum(proportions)
    cumulative: list[tuple[float, str]] = []
    running = 0.0
    for terrain_type, proportion in zip(terrain_types, proportions, strict=True):
        running += proportion / total if total else 0.0
        cumulative.append((running, terrain_type))

    def choose_type(row: int, col: int) -> str:
        if config.terrain.curriculum:
            fraction = (col + 0.5) / params["cols"]
            return next(
                (item for limit, item in cumulative if fraction <= limit), terrain_types[-1]
            )
        sample = rng.random()
        return next(
            (terrain_type for limit, terrain_type in cumulative if sample <= limit),
            terrain_types[-1],
        )

    for row in range(params["rows"]):
        difficulty = row / max(params["rows"] - 1, 1)
        for col in range(params["cols"]):
            center_x = (col + 0.5) * patch_x - width_x / 2.0
            center_y = (row + 0.5) * patch_y - width_y / 2.0
            terrain_type = choose_type(row, col)
            base_height = difficulty * params["max_height"] * 0.25
            common = {
                "type": "box",
                "size": f"{patch_x / 2:.6f} {patch_y / 2:.6f} 0.005000",
                "pos": f"{center_x:.6f} {center_y:.6f} {base_height - 0.005:.6f}",
                "material": "groundplane",
                "contype": "1",
                "conaffinity": "1",
                "solref": "0.04 1",
                "solimp": "0.85 0.95 0.001 0.5 2",
            }
            ET.SubElement(terrain_body, "geom", {**common, "name": f"terrain_{row}_{col}"})
            if terrain_type == "random_grid":
                for index in range(3):
                    height = rng.uniform(0.0, params["max_height"] * (0.5 + difficulty))
                    x = center_x - patch_x * 0.3 + index * patch_x * 0.3
                    ET.SubElement(
                        terrain_body,
                        "geom",
                        {
                            **common,
                            "name": f"terrain_{row}_{col}_grid_{index}",
                            "pos": (
                                f"{x:.6f} {center_y:.6f} {base_height + height / 2 - 0.005:.6f}"
                            ),
                            "size": (
                                f"{patch_x / 8:.6f} {patch_y / 5:.6f} {max(height / 2, 0.0005):.6f}"
                            ),
                        },
                    )
            elif terrain_type == "pyramid_stairs":
                for index in range(1, 4):
                    height = params["max_height"] * difficulty * index / 3.0
                    ET.SubElement(
                        terrain_body,
                        "geom",
                        {
                            **common,
                            "name": f"terrain_{row}_{col}_step_{index}",
                            "pos": (
                                f"{center_x - patch_x / 2 + patch_x * index / 8:.6f} "
                                f"{center_y:.6f} {base_height + height / 2 - 0.005:.6f}"
                            ),
                            "size": (
                                f"{patch_x * index / 8:.6f} {patch_y / 3:.6f} "
                                f"{max(height / 2, 0.0005):.6f}"
                            ),
                        },
                    )
            elif terrain_type == "pyramid_slope":
                angle = 0.22 * difficulty
                common["type"] = "box"
                common["size"] = f"{patch_x / 2:.6f} {patch_y / 2:.6f} 0.005000"
                common["quat"] = f"{math.cos(angle / 2):.8f} 0 {math.sin(angle / 2):.8f} 0"
                ET.SubElement(
                    terrain_body, "geom", {**common, "name": f"terrain_{row}_{col}_slope"}
                )
    # A single-entity generated terrain retains the entity's compatible
    # keyframes.  Composed scenes remove keyframes at their attach boundary
    # and overlay them by entity joint name in the model loader.
    config.terrain.generated_origins = tuple(tuple(item) for item in origins)
    config.terrain.generated_types = tuple(tuple(item) for item in type_names)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def make_microduck_ramp_scene(config: SceneCfg) -> Path:
    """Generate a flat-entry/ramp/runout terrain scene for roller tasks."""

    terrain = config.terrain
    params = terrain.params
    base = config.scene_xml
    if base is None:
        entity = config.entities.get("robot") or next(iter(config.entities.values()))
        base = entity.load_path
    base = base.resolve()
    length = float(params.get("length", 4.0))
    angle = float(params.get("angle", params.get("max_angle", 0.25)))
    width = float(params.get("width", 2.0))
    runout = float(params.get("runout", 2.0))
    thickness = float(params.get("thickness", 0.5))
    flat_length = float(params.get("flat_length", 2.0))
    spawn_on_ramp = float(params.get("spawn_on_ramp", 0.3))
    if length <= 0 or width <= 0 or runout < 0:
        raise ValueError("Ramp length/width must be positive and runout non-negative")
    key = hashlib.sha256(
        b"microduck-ramp-scene-v3\0"
        + json.dumps(
            {
                "base": _source_manifest(base),
                "length": length,
                "angle": angle,
                "width": width,
                "runout": runout,
                "thickness": thickness,
                "flat_length": flat_length,
                "spawn_on_ramp": spawn_on_ramp,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output_dir = Path(tempfile.gettempdir()) / "microduck_rl_torch" / "scenes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"ramp_{key}.xml"
    generated_origins = (((flat_length + spawn_on_ramp, 0.0, -spawn_on_ramp * math.tan(angle)),),)
    if output.is_file():
        config.terrain.generated_origins = generated_origins
        config.terrain.generated_types = (("ramp",),)
        return output
    root = _expand_includes(base)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Ramp base scene has no worldbody: {base}")
    body = ET.SubElement(worldbody, "body", {"name": "terrain"})
    drop = length * math.tan(angle)
    surface = {
        "type": "box",
        "size": f"{length / (2 * math.cos(angle)):.6f} {width / 2:.6f} {thickness / 2:.6f}",
        "material": "groundplane",
        "contype": "1",
        "conaffinity": "1",
    }
    ET.SubElement(
        body,
        "geom",
        {
            **surface,
            "name": "terrain_ramp",
            "pos": (
                f"{flat_length + length / 2 - thickness * math.sin(angle) / 2:.6f} "
                f"0 {-drop / 2 - thickness * math.cos(angle) / 2:.6f}"
            ),
            "quat": f"{math.cos(angle / 2):.8f} 0 {math.sin(angle / 2):.8f} 0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            **surface,
            "name": "terrain_entry",
            "size": f"{flat_length / 2:.6f} {width / 2:.6f} {thickness / 2:.6f}",
            "pos": f"{flat_length / 2:.6f} 0 {-thickness / 2:.6f}",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            **surface,
            "name": "terrain_runout",
            "size": f"{runout / 2:.6f} {width / 2:.6f} {thickness / 2:.6f}",
            "pos": f"{flat_length + length + runout / 2:.6f} 0 {-drop - thickness / 2:.6f}",
        },
    )
    # Keep a support origin on the generated ramp, matching the upstream
    # TerrainOutput contract.  The manager can select this origin without
    # knowing anything about the ramp geometry.
    config.terrain.generated_origins = generated_origins
    config.terrain.generated_types = (("ramp",),)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


@dataclass
class TerrainManager:
    """Runtime terrain origins, types, and difficulty levels.

    Geometry is compiled once; reset/curriculum changes select a different
    pre-generated origin instead of mutating global terrain state.
    """

    config: TerrainCfg
    num_envs: int = 1
    device: torch.device | str = "cpu"
    env_spacing: float = 2.0

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        output = self.config.generated_output
        if output is not None and output.origins is None and output.origin is not None:
            origin = np.asarray(output.origin, dtype=np.float64)
            if origin.shape != (3,):
                raise ValueError("TerrainOutput.origin must contain three coordinates")
            self.config.generated_origins = ((tuple(float(value) for value in origin),),)
        elif output is not None and output.origins is not None:
            origins = np.asarray(output.origins, dtype=np.float64)
            if origins.shape == (3,):
                origins = origins.reshape(1, 1, 3)
            elif origins.ndim == 2 and origins.shape[-1] == 3:
                origins = origins.reshape(1, *origins.shape)
            if origins.ndim != 3 or origins.shape[-1] != 3:
                raise ValueError("TerrainOutput.origins must have shape (rows, cols, 3)")
            self.config.generated_origins = tuple(
                tuple(tuple(float(value) for value in origin) for origin in row) for row in origins
            )
        if output is not None and output.types is not None:
            types = output.types
            if isinstance(types, str):
                self.config.generated_types = ((types,),)
            else:
                type_array = np.asarray(types, dtype=object)
                if type_array.ndim == 0:
                    self.config.generated_types = ((str(type_array.item()),),)
                elif type_array.ndim == 1:
                    self.config.generated_types = (tuple(str(value) for value in type_array),)
                else:
                    self.config.generated_types = tuple(
                        tuple(str(value) for value in row) for row in type_array.tolist()
                    )
        rows = int(self.config.params.get("rows", 1))
        cols = int(self.config.params.get("cols", 1))
        if output is not None and self.config.generated_origins is not None:
            generated_shape = torch.as_tensor(self.config.generated_origins).shape
            if len(generated_shape) != 3 or generated_shape[-1] != 3:
                raise ValueError(
                    "TerrainOutput.origins must have shape (rows, cols, 3), "
                    f"got {tuple(generated_shape)}"
                )
            rows, cols = int(generated_shape[0]), int(generated_shape[1])
        if rows < 1 or cols < 1:
            raise ValueError("Terrain rows and cols must be positive")
        width_x, width_y = tuple(
            float(value) for value in self.config.params.get("size", (8.0, 8.0))
        )
        if width_x <= 0 or width_y <= 0:
            raise ValueError("Terrain size must contain two positive values")
        if self.env_spacing <= 0:
            raise ValueError("Scene env_spacing must be positive")
        xs = torch.linspace(
            -width_x / 2 + width_x / (2 * cols),
            width_x / 2 - width_x / (2 * cols),
            cols,
            device=self.device,
        )
        ys = torch.linspace(
            -width_y / 2 + width_y / (2 * rows),
            width_y / 2 - width_y / (2 * rows),
            rows,
            device=self.device,
        )
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.origins = torch.stack((grid_x, grid_y, torch.zeros_like(grid_x)), dim=-1)
        generated = self.config.generated_origins
        if generated is not None:
            generated_tensor = torch.as_tensor(generated, dtype=torch.float32, device=self.device)
            if generated_tensor.shape != (rows, cols, 3):
                raise ValueError(
                    "Terrain generated_origins must have shape "
                    f"({rows}, {cols}, 3), got {tuple(generated_tensor.shape)}"
                )
            self.origins = generated_tensor.to(dtype=torch.float32)
        self.difficulties = torch.linspace(0.0, 1.0, rows, device=self.device)
        if output is not None and output.difficulties is not None:
            difficulties = torch.as_tensor(
                output.difficulties, dtype=torch.float32, device=self.device
            )
            if difficulties.ndim == 1 and difficulties.shape[0] == rows:
                self.difficulties = difficulties
        self.type_names = self.config.generated_types
        if self.config.kind == "plane" and self.num_envs > 1:
            # A shared flat floor still needs distinct per-environment spawn
            # origins.  Keep this separate from terrain levels/types: plane
            # placement is deterministic by environment index, not a
            # curriculum choice.
            placement_cols = max(1, int(ceil(self.num_envs**0.5)))
            placement_rows = int(ceil(self.num_envs / placement_cols))
            x = torch.arange(placement_cols, device=self.device, dtype=torch.float32)
            y = torch.arange(placement_rows, device=self.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)[: self.num_envs]
            grid -= grid.mean(dim=0, keepdim=True)
            plane_origins = torch.zeros((self.num_envs, 3), device=self.device)
            plane_origins[:, :2] = grid * float(self.env_spacing)
            self.origins = plane_origins.reshape(1, self.num_envs, 3)
        self.levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.types = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device)
        self._generator: torch.Generator | None = None
        self._generators: list[torch.Generator | None] = [None] * self.num_envs

    def reset(
        self, env_ids: torch.Tensor | slice | None = None, *, seed: int | None = None
    ) -> None:
        if seed is not None:
            self.set_seed(seed)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else self._ids(env_ids)
        )
        max_level = self.origins.shape[0] - 1
        initial = self.config.max_init_level
        if initial is None:
            initial = max_level
        if initial < 0:
            raise ValueError("Terrain max_init_level must be non-negative")
        level_high = min(initial, max_level) + 1
        for index in ids.tolist():
            generator = self._generators[index] or self._generator
            if self.config.kind == "plane" and self.origins.shape[1] == self.num_envs:
                self.levels[index] = 0
                self.types[index] = index
            else:
                self.levels[index] = torch.randint(
                    0, level_high, (), generator=generator, device=self.device
                )
                self.types[index] = torch.randint(
                    0, self.origins.shape[1], (), generator=generator, device=self.device
                )
        self.env_origins[ids] = self.origins[self.levels[ids], self.types[ids]]

    def set_seed(self, seed: int | None, env_ids: torch.Tensor | slice | None = None) -> None:
        if seed is not None:
            ids = list(range(self.num_envs)) if env_ids is None else self._ids(env_ids).tolist()
            if len(self._generators) != self.num_envs:
                self._generators = [None] * self.num_envs
            for index in ids:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed + index)
                self._generators[index] = generator
            self._generator = next((item for item in self._generators if item is not None), None)

    def set_generator(self, generator: torch.Generator | None) -> None:
        """Use the environment-owned RNG when no terrain-specific seed is set."""

        self._generator = generator
        if self.num_envs == 1:
            self._generators = [generator]

    def set_generators(
        self, generators: tuple[torch.Generator, ...] | list[torch.Generator]
    ) -> None:
        """Bind independent environment RNG streams from a batched backend."""

        if len(generators) != self.num_envs:
            raise ValueError("Terrain generator count must equal num_envs")
        self._generators = list(generators)
        self._generator = self._generators[0] if self._generators else None

    def advance(self, env_ids: torch.Tensor | slice, *, delta: int = 1) -> None:
        ids = self._ids(env_ids)
        max_level = self.origins.shape[0] - 1
        levels = self.levels[ids].clamp(0, max_level) + delta
        if delta > 0:
            # The final generated row is a valid difficulty level.  Wrap only
            # after passing it; using >= would make the highest terrain row
            # unreachable through curriculum progression.
            wrapped = levels > max_level
            if bool(wrapped.any()) and max_level > 0:
                random_levels = torch.stack(
                    [
                        torch.randint(
                            0,
                            max_level + 1,
                            (),
                            generator=self._generators[int(index)] or self._generator,
                            device=self.device,
                        )
                        for index in ids[wrapped].tolist()
                    ]
                )
                levels = torch.where(wrapped, random_levels, levels)
            else:
                levels = levels.clamp_max(max_level)
        else:
            levels = levels.clamp(0, max_level)
        self.levels[ids] = levels
        self.env_origins[ids] = self.origins[self.levels[ids], self.types[ids]]

    @property
    def terrain_levels(self) -> torch.Tensor:
        return self.levels

    @property
    def terrain_types(self) -> torch.Tensor:
        return self.types

    def update_env_origins(
        self,
        env_ids: torch.Tensor,
        move_up: torch.Tensor,
        move_down: torch.Tensor,
    ) -> None:
        """Apply upstream-compatible per-environment terrain progression."""

        ids = self._ids(env_ids)
        up = torch.as_tensor(move_up, dtype=torch.bool, device=self.device).reshape(-1)
        down = torch.as_tensor(move_down, dtype=torch.bool, device=self.device).reshape(-1)
        if up.shape != ids.shape or down.shape != ids.shape:
            raise ValueError("Terrain progression masks must match env_ids")
        if bool((up & down).any()):
            raise ValueError("Terrain progression cannot move an environment up and down together")
        self.levels[ids] = self.levels[ids].clamp(0, self.origins.shape[0] - 1)
        self.levels[ids] += up.to(torch.long) - down.to(torch.long)
        max_level = self.origins.shape[0] - 1
        over = self.levels[ids] > max_level
        if bool(over.any()) and max_level > 0:
            replacement = torch.stack(
                [
                    torch.randint(
                        0,
                        max_level + 1,
                        (),
                        generator=self._generators[int(index)] or self._generator,
                        device=self.device,
                    )
                    for index in ids[over].tolist()
                ]
            )
            levels = self.levels[ids].clone()
            levels[over] = replacement
            self.levels[ids] = levels.clamp_min(0)
        else:
            self.levels[ids].clamp_(0, max_level)
        self.env_origins[ids] = self.origins[self.levels[ids], self.types[ids]]

    def _ids(self, env_ids: torch.Tensor | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
