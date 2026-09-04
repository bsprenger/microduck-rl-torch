"""Declarative scene and entity specifications.

Task configuration is separate from the MuJoCo entities it uses. This module
provides the small, dependency-free specification layer used by the Torch
environment. The specifications are intentionally immutable: task factories
clone them and mutate the containing scene configuration instead of mutating a
shared robot constant.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from scipy import ndimage

from .actuation import ActuatorDelayCfg

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
    """Per-entity reset state."""

    pos: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None
    joint_pos: dict[str, float] = field(default_factory=dict)
    joint_vel: dict[str, float] = field(default_factory=dict)
    linear_velocity: tuple[float, float, float] | None = None
    angular_velocity: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class AssetSwapTransform:
    """Composable pre-attachment transform that replaces an entity asset."""

    xml_path: Path
    name: str = "asset_swap"
    version: str = "1"
    topology_changing: bool = True

    @property
    def cache_key(self) -> tuple[str, str, str]:
        path = self.xml_path.resolve()
        manifest = _source_manifest(path) if path.is_file() else [(str(path), "missing")]
        digest = hashlib.sha256(repr(manifest).encode()).hexdigest()
        return (self.name, self.version, f"{path}:{digest}")

    def apply(self, spec: Any, *, entity_cfg: EntityCfg | None = None, context: Any = None) -> Any:
        del spec, entity_cfg, context
        import mujoco

        replacement = mujoco.MjSpec.from_file(str(self.xml_path.resolve()))
        _normalize_spec_resources(replacement, self.xml_path.resolve())
        return replacement


@dataclass(frozen=True)
class BacklashTransform:
    """Inject serial gearbox-play joints into an articulated entity.

    This is a topology transform, not a special model path. It operates on
    the already selected base ``MjSpec`` and can therefore compose with an
    asset swap, roller geometry, collision transform, or other entity
    transform in a deterministic order.
    """

    total_degrees: float = 2.0
    damping: float = 0.01
    armature: float = 0.001
    frictionloss: float = 0.0
    joint_class: str = "chosen_actuator"
    exclude_pattern: str | None = None
    name: str = "backlash"
    version: str = "1"

    @property
    def cache_key(self) -> tuple[str, str, float, float, float, float, str, str | None]:
        return (
            self.name,
            self.version,
            self.total_degrees,
            self.damping,
            self.armature,
            self.frictionloss,
            self.joint_class,
            self.exclude_pattern,
        )

    def apply(self, spec: Any, *, entity_cfg: EntityCfg | None = None, context: Any = None) -> Any:
        del entity_cfg, context
        import mujoco

        if self.total_degrees <= 0:
            raise ValueError("Backlash total_degrees must be positive")
        exclude = re.compile(self.exclude_pattern) if self.exclude_pattern else None
        joints = tuple(spec.joints)
        targets = [
            joint
            for joint in joints
            if str(getattr(joint, "classname", "")) == self.joint_class
            and int(joint.type) == int(mujoco.mjtJoint.mjJNT_HINGE)
            and joint.name
            and (exclude is None or exclude.search(str(joint.name)) is None)
            and not str(joint.name).startswith("passive_")
        ]
        if not targets:
            raise ValueError(
                f"BacklashTransform found no hinge joints with class {self.joint_class!r}"
            )
        half_range = math.radians(self.total_degrees) / 2.0
        for joint in targets:
            passive_name = f"passive_{joint.name}_backlash"
            if any(candidate.name == passive_name for candidate in spec.joints):
                raise ValueError(f"Backlash joint {passive_name!r} already exists")
            passive = joint.parent.add_joint()
            passive.name = passive_name
            passive.type = mujoco.mjtJoint.mjJNT_HINGE
            passive.axis = joint.axis
            passive.pos = joint.pos
            passive.damping = self.damping
            passive.frictionloss = self.frictionloss
            passive.armature = self.armature
            passive.limited = True
            passive.range = [-half_range, half_range]
            passive.solref_limit = [0.01, 1.0]
            passive.solimp_limit = [0.95, 0.999, 0.0001, 0.5, 2.0]
        return spec


@dataclass(frozen=True)
class EntityCfg:
    """One scene entity and the semantic handles required by task terms."""

    name: str
    xml_path: Path
    # Optional source for entity-authored initialization keyframes. World
    # composition is owned by SceneCfg.scene_xml.
    keyframe_source: Path | None = None
    kind: str = "robot"
    keyframe_name: str | None = None
    root_body_name: str | None = None
    head_body_names: tuple[str, ...] = ()
    foot_site_selector: SemanticSelector | None = None
    foot_contact_selectors: tuple[SemanticSelector, SemanticSelector] | None = None
    collision_name_suffix: str | None = None
    actuator_mode: Literal["bam", "xml"] = "xml"
    actuator_delay: ActuatorDelayCfg = field(default_factory=ActuatorDelayCfg)
    actuator_joint_names: tuple[str, ...] = ()
    # Spawn transforms are applied by the scene composer to the entity root
    # body.  Task reset terms may still override dynamic qpos/qvel afterward.
    spawn_pos: tuple[float, float, float] | None = None
    spawn_quat: tuple[float, float, float, float] | None = None
    init_state: EntityInitStateCfg = field(default_factory=EntityInitStateCfg)
    # This pre-attach MjSpec mutation hook is deliberately typed as a callable
    # rather than a Microduck-specific editor so articulated robots, props, and
    # fixed obstacles share it.
    spec_factory: Callable[..., Any] | None = None
    spec_fn: Callable[..., Any] | None = None
    # Ordered topology/model transforms run before this entity is attached to
    # the scene. Runtime scalar changes belong to ModelMutationManager.
    transforms: tuple[Any, ...] = ()


@dataclass
class TerrainGeometry:
    """One MuJoCo geometry returned by a sub-terrain generator."""

    geom: Any | None = None
    hfield: Any | None = None
    color: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FlatPatchSamplingCfg:
    """Configuration for sampling circular, approximately flat spawn patches."""

    num_patches: int = 10
    patch_radius: float = 0.5
    max_height_diff: float = 0.05
    x_range: tuple[float, float] = (-1.0e6, 1.0e6)
    y_range: tuple[float, float] = (-1.0e6, 1.0e6)
    z_range: tuple[float, float] = (-1.0e6, 1.0e6)
    grid_resolution: float | None = None

    def __post_init__(self) -> None:
        if self.num_patches < 1 or self.patch_radius <= 0.0:
            raise ValueError("Flat patch count and radius must be positive")
        if self.max_height_diff < 0.0:
            raise ValueError("Flat patch max_height_diff must be non-negative")


@dataclass
class TerrainOutput:
    """Typed result of a terrain generation pass.

    Terrain generators return one ``TerrainOutput`` per patch, while the
    generator assembles those patches into a grid. The scene boundary accepts
    both forms: ``origin``/``geometries`` describe one
    generated patch and ``origins``/``types`` describe the complete runtime
    assignment table.  This keeps geometry generation and reset placement
    coupled without making ``TerrainManager`` know how terrain was built.
    """

    origin: Any | None = None
    geometries: list[Any] = field(default_factory=list)
    origins: Any | None = None
    types: Any | None = None
    difficulties: Any | None = None
    flat_patches: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerrainGeneratorCfg:
    """Declarative terrain grid configuration."""

    size: tuple[float, float] = (8.0, 8.0)
    border_width: float = 0.0
    border_height: float = 1.0
    num_rows: int = 1
    num_cols: int = 1
    curriculum: bool = False
    difficulty_range: tuple[float, float] = (0.0, 1.0)
    seed: int | None = None
    sub_terrains: dict[str, Any] = field(default_factory=dict)
    color_scheme: Literal["height", "random", "none"] = "height"
    add_lights: bool = False

    def __post_init__(self) -> None:
        if len(self.size) != 2 or any(float(value) <= 0 for value in self.size):
            raise ValueError("Terrain generator size must contain two positive values")
        if self.num_rows < 1 or self.num_cols < 1:
            raise ValueError("Terrain generator rows and columns must be positive")
        if len(self.difficulty_range) != 2:
            raise ValueError("Terrain difficulty_range must contain two values")


@dataclass
class SubTerrainCfg:
    """Base contract for one procedural terrain patch.

    This is intentionally the same boundary as mjlab's ``SubTerrainCfg``:
    patch functions receive the difficulty, the scene spec, and the generator
    RNG, and return local geometry plus the local spawn origin.  The Torch
    backend owns the concrete MuJoCo realization, not the task hierarchy.
    """

    proportion: float = 1.0
    size: tuple[float, float] = (10.0, 10.0)
    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None

    def function(self, difficulty: float, spec: Any, rng: np.random.Generator) -> TerrainOutput:
        del difficulty, spec, rng
        raise NotImplementedError


def _find_flat_patches_from_heightfield(
    heights: np.ndarray,
    horizontal_scale: float,
    z_offset: float,
    cfg: FlatPatchSamplingCfg,
    rng: np.random.Generator,
) -> np.ndarray:
    """Find and sample valid circular footprints using SciPy morphology."""

    if cfg.grid_resolution is not None and cfg.grid_resolution < horizontal_scale:
        zoom_factor = horizontal_scale / cfg.grid_resolution
        heights = np.asarray(ndimage.zoom(heights, zoom_factor, order=1))
        horizontal_scale = cfg.grid_resolution
    rows, cols = heights.shape
    radius = int(np.ceil(cfg.patch_radius / horizontal_scale))
    y_grid, x_grid = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    footprint = (x_grid**2 + y_grid**2) <= radius**2
    max_height = ndimage.maximum_filter(heights, footprint=footprint, mode="constant", cval=-np.inf)
    min_height = ndimage.minimum_filter(heights, footprint=footprint, mode="constant", cval=np.inf)
    valid_mask = (max_height - min_height) <= cfg.max_height_diff
    if radius:
        valid_mask[:radius, :] = False
        valid_mask[-radius:, :] = False
        valid_mask[:, :radius] = False
        valid_mask[:, -radius:] = False
    x_coords = np.arange(cols) * horizontal_scale
    y_coords = np.arange(rows) * horizontal_scale
    valid_mask &= (
        (y_coords >= cfg.y_range[0])[:, None]
        & (y_coords <= cfg.y_range[1])[:, None]
        & (x_coords >= cfg.x_range[0])[None, :]
        & (x_coords <= cfg.x_range[1])[None, :]
    )
    z_values = heights + z_offset
    valid_mask &= (z_values >= cfg.z_range[0]) & (z_values <= cfg.z_range[1])
    valid_indices = np.argwhere(valid_mask)
    if len(valid_indices) == 0:
        center_row, center_col = min(rows // 2, rows - 1), min(cols // 2, cols - 1)
        return np.tile(
            (
                cols * horizontal_scale / 2.0,
                rows * horizontal_scale / 2.0,
                heights[center_row, center_col] + z_offset,
            ),
            (cfg.num_patches, 1),
        )
    choices = rng.choice(
        len(valid_indices), size=cfg.num_patches, replace=len(valid_indices) < cfg.num_patches
    )
    selected = valid_indices[np.asarray(choices).reshape(-1)]
    return np.stack(
        (
            selected[:, 1] * horizontal_scale,
            selected[:, 0] * horizontal_scale,
            heights[selected[:, 0], selected[:, 1]] + z_offset,
        ),
        axis=-1,
    )


def _terrain_body(spec: Any) -> Any:
    try:
        body = spec.body("terrain")
    except (KeyError, ValueError):
        body = None
    return body if body is not None else spec.worldbody.add_body(name="terrain")


def _safe_box_size(size: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        max(1.0e-6, float(size[0]) / 2.0),
        max(1.0e-6, float(size[1]) / 2.0),
        max(1.0e-6, float(size[2]) / 2.0),
    )


@dataclass(kw_only=True)
class BoxFlatTerrainCfg(SubTerrainCfg):
    """Finite flat box patch matching mjlab's ``BoxFlatTerrainCfg``."""

    def function(self, difficulty: float, spec: Any, rng: np.random.Generator) -> TerrainOutput:
        del difficulty, rng
        body = _terrain_body(spec)
        geom = body.add_geom(
            type=_mujoco_geom_box(),
            size=_safe_box_size((self.size[0], self.size[1], 1.0)),
            pos=(self.size[0] / 2.0, self.size[1] / 2.0, -0.5),
        )
        return TerrainOutput(
            origin=np.asarray((self.size[0] / 2.0, self.size[1] / 2.0, 0.0)),
            geometries=[TerrainGeometry(geom=geom, color=(0.5, 0.5, 0.5, 1.0))],
        )


def _mujoco_geom_box() -> Any:
    import mujoco

    return mujoco.mjtGeom.mjGEOM_BOX


@dataclass(kw_only=True)
class BoxPyramidStairsTerrainCfg(SubTerrainCfg):
    """Four-sided stepped pyramid matching mjlab's primitive terrain."""

    border_width: float = 0.0
    step_height_range: tuple[float, float] = (0.0, 0.0)
    step_width: float = 0.15
    platform_width: float = 1.0
    holes: bool = False

    def function(self, difficulty: float, spec: Any, rng: np.random.Generator) -> TerrainOutput:
        del rng
        body = _terrain_body(spec)
        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )
        num_steps_x = int(
            (self.size[0] - 2.0 * self.border_width - self.platform_width) / (2.0 * self.step_width)
        )
        num_steps_y = int(
            (self.size[1] - 2.0 * self.border_width - self.platform_width) / (2.0 * self.step_width)
        )
        num_steps = max(0, min(num_steps_x, num_steps_y))
        geometries: list[TerrainGeometry] = []

        if self.border_width > 0.0 and not self.holes:
            thickness = self.border_width
            for size, pos in (
                (
                    (self.size[0], thickness, step_height),
                    (self.size[0] / 2.0, self.size[1] - thickness / 2.0, -step_height / 2.0),
                ),
                (
                    (self.size[0], thickness, step_height),
                    (self.size[0] / 2.0, thickness / 2.0, -step_height / 2.0),
                ),
                (
                    (thickness, self.size[1] - 2.0 * thickness, step_height),
                    (
                        thickness / 2.0,
                        self.size[1] / 2.0,
                        -step_height / 2.0,
                    ),
                ),
                (
                    (thickness, self.size[1] - 2.0 * thickness, step_height),
                    (
                        self.size[0] - thickness / 2.0,
                        self.size[1] / 2.0,
                        -step_height / 2.0,
                    ),
                ),
            ):
                geometries.append(
                    TerrainGeometry(
                        geom=body.add_geom(
                            type=_mujoco_geom_box(), size=_safe_box_size(size), pos=pos
                        ),
                        color=(0.12, 0.25, 0.65, 1.0),
                    )
                )

        terrain_size = (
            self.size[0] - 2.0 * self.border_width,
            self.size[1] - 2.0 * self.border_width,
        )
        center = (self.size[0] / 2.0, self.size[1] / 2.0)
        for index in range(num_steps):
            box_offset = (index + 0.5) * self.step_width
            box_height = (index + 2) * step_height
            if self.holes:
                outer_x, outer_y = self.platform_width, self.platform_width
            else:
                outer_x = terrain_size[0] - 2.0 * index * self.step_width
                outer_y = terrain_size[1] - 2.0 * index * self.step_width
            for size, pos in (
                (
                    (outer_x, self.step_width, box_height),
                    (
                        center[0],
                        center[1] + terrain_size[1] / 2.0 - box_offset,
                        index * step_height / 2.0,
                    ),
                ),
                (
                    (outer_x, self.step_width, box_height),
                    (
                        center[0],
                        center[1] - terrain_size[1] / 2.0 + box_offset,
                        index * step_height / 2.0,
                    ),
                ),
            ):
                geometries.append(
                    TerrainGeometry(
                        geom=body.add_geom(
                            type=_mujoco_geom_box(), size=_safe_box_size(size), pos=pos
                        ),
                        color=(0.20 + 0.04 * index, 0.45, 0.95, 1.0),
                    )
                )
            inner_y = self.platform_width if self.holes else outer_y - 2.0 * self.step_width
            for size, pos in (
                (
                    (self.step_width, inner_y, box_height),
                    (
                        center[0] + terrain_size[0] / 2.0 - box_offset,
                        center[1],
                        index * step_height / 2.0,
                    ),
                ),
                (
                    (self.step_width, inner_y, box_height),
                    (
                        center[0] - terrain_size[0] / 2.0 + box_offset,
                        center[1],
                        index * step_height / 2.0,
                    ),
                ),
            ):
                geometries.append(
                    TerrainGeometry(
                        geom=body.add_geom(
                            type=_mujoco_geom_box(), size=_safe_box_size(size), pos=pos
                        ),
                        color=(0.20 + 0.04 * index, 0.45, 0.95, 1.0),
                    )
                )

        middle = (
            terrain_size[0] - 2.0 * num_steps * self.step_width,
            terrain_size[1] - 2.0 * num_steps * self.step_width,
            (num_steps + 2) * step_height,
        )
        geometries.append(
            TerrainGeometry(
                geom=body.add_geom(
                    type=_mujoco_geom_box(),
                    size=_safe_box_size(middle),
                    pos=(center[0], center[1], num_steps * step_height / 2.0),
                ),
                color=(0.35, 0.55, 0.95, 1.0),
            )
        )
        return TerrainOutput(
            origin=np.asarray((center[0], center[1], (num_steps + 1) * step_height)),
            geometries=geometries,
        )


@dataclass(kw_only=True)
class BoxRandomGridTerrainCfg(SubTerrainCfg):
    """Random-height box grid matching mjlab's rough cobblestone terrain."""

    grid_width: float = 0.45
    grid_height_range: tuple[float, float] = (0.0, 0.0)
    platform_width: float = 1.0
    holes: bool = False
    merge_similar_heights: bool = False
    height_merge_threshold: float = 0.05
    max_merge_distance: int = 3
    border_width: float = 0.25

    def function(self, difficulty: float, spec: Any, rng: np.random.Generator) -> TerrainOutput:
        if self.size[0] != self.size[1]:
            raise ValueError(f"The terrain must be square. Received size: {self.size}.")
        body = _terrain_body(spec)
        grid_height = self.grid_height_range[0] + difficulty * (
            self.grid_height_range[1] - self.grid_height_range[0]
        )
        num_x = int((self.size[0] - 2.0 * self.border_width) / self.grid_width)
        num_y = int((self.size[1] - 2.0 * self.border_width) / self.grid_width)
        remainder = self.size[0] - min(num_x, num_y) * self.grid_width
        if remainder <= 0.0:
            raise RuntimeError("Random-grid border width must be greater than zero")
        terrain_height = 1.0
        geometries: list[TerrainGeometry] = []
        # mjlab splits the unused remainder around the grid: the generated
        # edge boxes are half the remainder thick, while the grid starts half
        # the remainder from the patch edge.  Treating the remainder itself
        # as the box thickness leaves the same terrain usable but changes the
        # support surface and every raycast near the edge.
        border_thickness = remainder / 2.0
        border_half = border_thickness / 2.0
        # ``border_thickness`` is the full width of the generated edge box
        # (the MuJoCo size is converted to a half-extent by
        # ``_safe_box_size``).  The first cell starts at that full width so
        # its near face meets the border's inner face, matching the intended
        # ``border_width / 2`` geometry.
        grid_start = border_thickness
        for size, pos in (
            (
                (self.size[0], border_thickness, terrain_height),
                (self.size[0] / 2.0, self.size[1] - border_half, -0.5),
            ),
            (
                (self.size[0], border_thickness, terrain_height),
                (self.size[0] / 2.0, border_half, -0.5),
            ),
            (
                (border_thickness, self.size[1] - 2.0 * border_thickness, terrain_height),
                (border_half, self.size[1] / 2.0, -0.5),
            ),
            (
                (border_thickness, self.size[1] - 2.0 * border_thickness, terrain_height),
                (self.size[0] - border_half, self.size[1] / 2.0, -0.5),
            ),
        ):
            geometries.append(
                TerrainGeometry(
                    geom=body.add_geom(type=_mujoco_geom_box(), size=_safe_box_size(size), pos=pos),
                    color=(0.10, 0.35, 0.18, 1.0),
                )
            )
        height_map = rng.uniform(-grid_height, grid_height, (num_x, num_y))
        platform_min = self.size[0] / 2.0 - self.platform_width / 2.0
        platform_max = self.size[0] / 2.0 + self.platform_width / 2.0
        if self.merge_similar_heights and not self.holes:
            boxes = self._create_merged_boxes(
                body, height_map, num_x, num_y, grid_height, terrain_height, border_thickness
            )
            geometries.extend(boxes)
        else:
            for ix in range(num_x):
                for iy in range(num_y):
                    center_x = grid_start + (ix + 0.5) * self.grid_width
                    center_y = grid_start + (iy + 0.5) * self.grid_width
                    if self.holes and not (
                        platform_min <= center_x <= platform_max
                        or platform_min <= center_y <= platform_max
                    ):
                        continue
                    height = float(height_map[ix, iy])
                    box_height = terrain_height + height
                    geometries.append(
                        TerrainGeometry(
                            geom=body.add_geom(
                                type=_mujoco_geom_box(),
                                size=_safe_box_size((self.grid_width, self.grid_width, box_height)),
                                pos=(center_x, center_y, -0.5 + height / 2.0),
                            ),
                            color=(0.25, 0.65, 0.30, 1.0),
                        )
                    )
        platform_height = terrain_height + grid_height
        geometries.append(
            TerrainGeometry(
                geom=body.add_geom(
                    type=_mujoco_geom_box(),
                    size=_safe_box_size(
                        (self.platform_width, self.platform_width, platform_height)
                    ),
                    pos=(self.size[0] / 2.0, self.size[1] / 2.0, -0.5 + grid_height / 2.0),
                ),
                color=(0.45, 0.75, 0.55, 1.0),
            )
        )
        return TerrainOutput(
            origin=np.asarray((self.size[0] / 2.0, self.size[1] / 2.0, grid_height)),
            geometries=geometries,
        )

    def _create_merged_boxes(
        self,
        body: Any,
        height_map: np.ndarray,
        num_x: int,
        num_y: int,
        grid_height: float,
        terrain_height: float,
        border_thickness: float,
    ) -> list[TerrainGeometry]:
        """Greedily merge compatible rectangular cell runs."""

        threshold = max(float(self.height_merge_threshold), 1.0e-12)
        quantized = np.round(height_map / threshold) * threshold
        visited = np.zeros((num_x, num_y), dtype=bool)
        result: list[TerrainGeometry] = []
        for ix in range(num_x):
            for iy in range(num_y):
                if visited[ix, iy]:
                    continue
                height = float(quantized[ix, iy])
                max_x, max_y = ix + 1, iy + 1
                while max_x < min(ix + self.max_merge_distance, num_x):
                    if visited[max_x, iy] or abs(quantized[max_x, iy] - height) > 1.0e-6:
                        break
                    max_x += 1
                can_expand = True
                while max_y < min(iy + self.max_merge_distance, num_y) and can_expand:
                    for x in range(ix, max_x):
                        if visited[x, max_y] or abs(quantized[x, max_y] - height) > 1.0e-6:
                            can_expand = False
                            break
                    if can_expand:
                        max_y += 1
                visited[ix:max_x, iy:max_y] = True
                width_x = (max_x - ix) * self.grid_width
                width_y = (max_y - iy) * self.grid_width
                center_x = border_thickness + (ix + (max_x - ix) / 2.0) * self.grid_width
                center_y = border_thickness + (iy + (max_y - iy) / 2.0) * self.grid_width
                normalized = (height + grid_height) / (2.0 * grid_height) if grid_height else 0.5
                shade = float(np.clip(normalized, 0.0, 1.0))
                result.append(
                    TerrainGeometry(
                        geom=body.add_geom(
                            type=_mujoco_geom_box(),
                            size=_safe_box_size((width_x, width_y, terrain_height + height)),
                            pos=(center_x, center_y, -terrain_height / 2.0 + height / 2.0),
                        ),
                        color=(0.20 + 0.20 * shade, 0.45 + 0.20 * shade, 0.30, 1.0),
                    )
                )
        return result


@dataclass(kw_only=True)
class HfPyramidSlopedTerrainCfg(SubTerrainCfg):
    """Smooth heightfield pyramid matching mjlab's slope terrain."""

    slope_range: tuple[float, float] = (0.0, 0.0)
    platform_width: float = 1.0
    inverted: bool = False
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 1.0

    def function(self, difficulty: float, spec: Any, rng: np.random.Generator) -> TerrainOutput:
        body = _terrain_body(spec)
        slope = self.slope_range[0] + difficulty * (self.slope_range[1] - self.slope_range[0])
        if self.inverted:
            slope = -slope
        if self.border_width > 0.0 and self.border_width < self.horizontal_scale:
            raise ValueError("Slope border_width must be at least horizontal_scale")
        width = int(self.size[0] / self.horizontal_scale)
        length = int(self.size[1] / self.horizontal_scale)
        border = int(self.border_width / self.horizontal_scale)
        inner_w = width - 2 * border
        inner_l = length - 2 * border
        if width < 2 or length < 2 or inner_w < 2 or inner_l < 2:
            raise ValueError("Slope terrain is too small for its resolution and border")
        noise = np.zeros((width, length), dtype=np.int16)
        cx, cy = int(inner_w / 2), int(inner_l / 2)
        x = np.arange(inner_w)
        y = np.arange(inner_l)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        xx = (cx - np.abs(cx - xx)) / max(cx, 1)
        yy = (cy - np.abs(cy - yy)) / max(cy, 1)
        height_max = int(slope * (inner_w * self.horizontal_scale) / 2.0 / self.vertical_scale)
        raw = height_max * xx * yy
        platform = int(self.platform_width / self.horizontal_scale / 2.0)
        px, py = inner_w // 2 - platform, inner_l // 2 - platform
        platform_height = raw[max(px, 0), max(py, 0)]
        raw = np.clip(raw, min(0, platform_height), max(0, platform_height))
        if border:
            noise[border:-border, border:-border] = np.rint(raw).astype(np.int16)
        else:
            noise = np.rint(raw).astype(np.int16)
        elevation_min = int(noise.min())
        elevation_max = int(noise.max())
        elevation_range = max(1, elevation_max - elevation_min)
        max_physical_height = elevation_range * self.vertical_scale
        normalized = (noise - elevation_min) / elevation_range
        field = spec.add_hfield(
            name=f"terrain_slope_{uuid.uuid4().hex}",
            size=(
                self.size[0] / 2.0,
                self.size[1] / 2.0,
                max_physical_height,
                max_physical_height * self.base_thickness_ratio,
            ),
            nrow=width,
            ncol=length,
            userdata=normalized.astype(np.float32).flatten().tolist(),
        )
        z_offset = -max_physical_height if self.inverted else 0.0
        # Keep physical heights exact; use the RGBA fallback because the
        # current backend path does not transfer buffer textures.
        geom = body.add_geom(
            type=_mujoco_geom_hfield(),
            hfieldname=field.name,
            pos=(self.size[0] / 2.0, self.size[1] / 2.0, z_offset),
        )
        flat_patches = {
            name: _find_flat_patches_from_heightfield(
                (noise.astype(np.float64) - noise.min()) * self.vertical_scale,
                self.horizontal_scale,
                z_offset,
                patch_cfg,
                rng,
            )
            for name, patch_cfg in (self.flat_patch_sampling or {}).items()
        }
        return TerrainOutput(
            origin=np.asarray(
                (
                    self.size[0] / 2.0,
                    self.size[1] / 2.0,
                    z_offset if self.inverted else max_physical_height,
                )
            ),
            geometries=[TerrainGeometry(geom=geom, hfield=field)],
            flat_patches=flat_patches,
        )


def _mujoco_geom_hfield() -> Any:
    import mujoco

    return mujoco.mjtGeom.mjGEOM_HFIELD


class TerrainGenerator:
    """Compile a typed terrain grid."""

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu") -> None:
        if not cfg.sub_terrains:
            raise ValueError("At least one sub-terrain must be specified")
        self.cfg = cfg
        self.device = device
        self._num_cols = len(cfg.sub_terrains) if cfg.curriculum else cfg.num_cols
        if self._num_cols < 1:
            raise ValueError("Terrain generator columns must be positive")
        # A generator owns a compiled copy.  Mutating caller-owned sub-terrain
        # configs here makes a second scene silently inherit the first scene's
        # patch size.
        self.sub_terrains = {name: deepcopy(sub_cfg) for name, sub_cfg in cfg.sub_terrains.items()}
        for sub_cfg in self.sub_terrains.values():
            sub_cfg.size = cfg.size
        seed = cfg.seed if cfg.seed is not None else int(np.random.randint(0, 10000))
        self.np_rng = np.random.default_rng(seed)
        self.terrain_origins = np.zeros((cfg.num_rows, self._num_cols, 3), dtype=np.float64)
        self.terrain_types = [["" for _ in range(self._num_cols)] for _ in range(cfg.num_rows)]
        self.terrain_difficulties = np.zeros((cfg.num_rows, self._num_cols), dtype=np.float64)
        self.flat_patches: dict[str, np.ndarray] = {}
        self.flat_patch_radii: dict[str, float] = {}
        patch_counts: dict[str, int] = {}
        for sub_cfg in self.sub_terrains.values():
            for name, patch_cfg in (sub_cfg.flat_patch_sampling or {}).items():
                patch_counts[name] = max(patch_counts.get(name, 0), patch_cfg.num_patches)
                self.flat_patch_radii[name] = max(
                    self.flat_patch_radii.get(name, 0.0), patch_cfg.patch_radius
                )
        for name, count in patch_counts.items():
            self.flat_patches[name] = np.zeros(
                (cfg.num_rows, self._num_cols, count, 3), dtype=np.float64
            )

    def compile(self, spec: Any) -> None:
        # Scene templates may already contain a terrain body.  Reuse it so a
        # generator composes with a world template without creating duplicate
        # MuJoCo names; a bare robot scene gets the body here.
        body = _terrain_body(spec)
        if tuple(body.geoms):
            raise ValueError(
                "Terrain body must be empty before TerrainGenerator.compile(); "
                "reusing stale terrain geometry would change the generated task"
            )
        if self.cfg.add_lights:
            light = body.add_light()
            light.name = "terrain_light"
            light.pos = (0.0, 0.0, max(self.cfg.size) * 0.75)
            light.dir = (0.0, 0.0, -1.0)
            light.directional = True
            light.castshadow = True
        sub_items = tuple(self.sub_terrains.items())
        proportions = np.asarray(
            [float(item.proportion) for _, item in sub_items], dtype=np.float64
        )
        if (proportions < 0).any() or float(proportions.sum()) <= 0:
            raise ValueError("Terrain proportions must be non-negative and non-zero")
        probabilities = proportions / proportions.sum()
        lower, upper = self.cfg.difficulty_range
        positions = (
            ((row, col) for col in range(self._num_cols) for row in range(self.cfg.num_rows))
            if self.cfg.curriculum
            else ((row, col) for row in range(self.cfg.num_rows) for col in range(self._num_cols))
        )
        for row, col in positions:
            if self.cfg.curriculum:
                difficulty = lower + (upper - lower) * (
                    (row + self.np_rng.uniform()) / self.cfg.num_rows
                )
                name, sub_cfg = sub_items[col]
            else:
                # Select terrain type first, then sample its independent
                # difficulty to keep the random stream deterministic.
                index = int(self.np_rng.choice(len(sub_items), p=probabilities))
                difficulty = float(self.np_rng.uniform(lower, upper))
                name, sub_cfg = sub_items[index]
            output = _coerce_terrain_output(sub_cfg.function(difficulty, spec, self.np_rng))
            if output.origin is None:
                raise ValueError(f"Terrain {name!r} did not return a spawn origin")
            corner = np.asarray(
                (
                    -self.cfg.num_rows * self.cfg.size[0] / 2.0 + row * self.cfg.size[0],
                    -self._num_cols * self.cfg.size[1] / 2.0 + col * self.cfg.size[1],
                    0.0,
                ),
                dtype=np.float64,
            )
            for terrain_geometry in output.geometries:
                geom = getattr(terrain_geometry, "geom", terrain_geometry)
                if geom is not None and hasattr(geom, "pos"):
                    geom.pos = np.asarray(geom.pos, dtype=np.float64) + corner
                if geom is not None:
                    color = getattr(terrain_geometry, "color", None)
                    if self.cfg.color_scheme == "height" and color is not None:
                        geom.rgba = color
                    elif self.cfg.color_scheme == "random":
                        geom.rgba = (*self.np_rng.uniform(0.3, 0.8, 3), 1.0)
                    elif self.cfg.color_scheme == "none":
                        geom.rgba = (0.5, 0.5, 0.5, 1.0)
            spawn_origin = np.asarray(output.origin, dtype=np.float64) + corner
            for patch_name, patches in self.flat_patches.items():
                local_patches = (output.flat_patches or {}).get(patch_name)
                if local_patches is None:
                    patches[row, col] = spawn_origin
                else:
                    local = np.asarray(local_patches, dtype=np.float64).reshape(-1, 3)
                    if len(local) > patches.shape[2]:
                        raise ValueError(
                            f"Terrain flat-patch set {patch_name!r} returned {len(local)} "
                            f"patches, maximum is {patches.shape[2]}"
                        )
                    patches[row, col, : len(local)] = local + corner
                    patches[row, col, len(local) :] = spawn_origin
            self.terrain_origins[row, col] = spawn_origin
            self.terrain_types[row][col] = str(name)
            self.terrain_difficulties[row, col] = difficulty
        total_size = (
            self.cfg.num_rows * self.cfg.size[0] + 2.0 * self.cfg.border_width,
            self._num_cols * self.cfg.size[1] + 2.0 * self.cfg.border_width,
        )
        if self.cfg.border_width > 0.0:
            thickness = self.cfg.border_width
            border_z = -abs(self.cfg.border_height) / 2.0
            for size, pos in (
                (
                    (total_size[0], thickness, abs(self.cfg.border_height)),
                    (0.0, total_size[1] / 2.0 - thickness / 2.0, border_z),
                ),
                (
                    (total_size[0], thickness, abs(self.cfg.border_height)),
                    (0.0, -total_size[1] / 2.0 + thickness / 2.0, border_z),
                ),
                (
                    (thickness, self._num_cols * self.cfg.size[1], abs(self.cfg.border_height)),
                    (-self.cfg.num_rows * self.cfg.size[0] / 2.0 - thickness / 2.0, 0.0, border_z),
                ),
                (
                    (thickness, self._num_cols * self.cfg.size[1], abs(self.cfg.border_height)),
                    (self.cfg.num_rows * self.cfg.size[0] / 2.0 + thickness / 2.0, 0.0, border_z),
                ),
            ):
                geom = body.add_geom(type=_mujoco_geom_box(), size=_safe_box_size(size), pos=pos)
                geom.rgba = (
                    (0.2, 0.2, 0.2, 1.0)
                    if self.cfg.color_scheme != "random"
                    else (*self.np_rng.uniform(0.3, 0.8, 3), 1.0)
                )
        for index, geom in enumerate(body.geoms):
            geom.name = f"terrain_{index}"
            geom.mass = 0.0


@dataclass
class TerrainCfg:
    """Terrain declaration; generators are deliberately opaque to the core."""

    kind: Literal["plane", "generator"] = "plane"
    generator: Any | None = None
    # A terrain generator implements the canonical ``compile(spec)`` boundary.
    # Scene mutations belong in ``spec_fn`` and runtime placement consumes the
    # generator's typed output metadata.
    spec_fn: Callable[..., Any] | None = None
    max_init_level: int | None = None
    # Filled by the deterministic generator at scene materialization time.
    # Keeping the generated spawn metadata on the terrain config makes the
    # runtime origin table describe the actual support geometry rather than a
    # fabricated z=0 grid.
    generated_origins: Any | None = field(default=None, repr=False, compare=False)
    generated_types: tuple[tuple[str, ...], ...] | None = field(
        default=None, repr=False, compare=False
    )
    generated_flat_patches: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    generated_output: TerrainOutput | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class SensorCfg:
    """Declarative first-class sensor contract.

    ``kind="mujoco"`` reads a named sensor declared in the compiled XML and
    preserves the original two-argument constructor.  The other kinds are
    resolved by ``SensorManager`` from semantic entity selectors, so task
    terms never need to know MuJoCo addresses.  ``reader`` is the explicit
    escape hatch for backend-native sensors while
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
    secondary_policy: Literal["first", "any", "error"] = "first"
    global_frame: bool = False
    update_period: int = 1
    debug_visualization: bool = False
    fields: tuple[str, ...] = ("found",)
    reduce: Literal["none", "mindist", "maxforce", "netforce"] = "maxforce"
    num_slots: int = 1
    track_air_time: bool = False
    reader: Any | None = None
    params: dict[str, Any] = field(default_factory=dict)
    history_length: int = 0

    @property
    def prefixed_name(self) -> str:
        """Name used by a newly authored MuJoCo sensor declaration.

        Existing XML sensors are intentionally looked up by their logical
        name. Only sensors authored through ``sensor_type`` receive an entity
        namespace.
        """

        if self.sensor_type is not None and self.entity:
            return f"{self.entity}/{self.name}"
        return self.name

    def build(self) -> Any:
        """Build the runtime sensor object without importing it at config time."""

        from .sensors import Sensor

        return Sensor(self)


@dataclass
class SceneCfg:
    """Composable scene containing named entities, terrain, and sensors."""

    entities: dict[str, EntityCfg] = field(default_factory=dict)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    sensors: dict[str, SensorCfg] = field(default_factory=dict)
    scene_xml: Path | None = None
    contact_options: dict[str, Any] = field(default_factory=dict)
    # Scene state is explicitly batched. A scalar scene is simply the B=1
    # specialization and keeps the existing public behavior.
    num_envs: int = 1
    env_spacing: float = 2.0
    spec_fn: Callable[..., Any] | None = None
    # Used only to scope scene-level sensors whose selectors are local to the
    # task's primary entity. It is explicit so multi-entity scenes never
    # depend on mapping insertion order.
    primary_entity: str | None = None


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
    retained. This makes a configured scene path a world template for a
    multi-entity scene while keeping world and entity ownership separate.
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
    world_bodies: set[str] = (
        {name for element in worldbody.iter("body") if (name := element.get("name")) is not None}
        if worldbody is not None
        else set()
    )
    world_geoms: set[str] = (
        {name for element in worldbody.iter("geom") if (name := element.get("name")) is not None}
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
    """Run an entity/scene mutation hook through ``hook(spec, cfg)``."""

    if hook is None:
        return None
    return hook(spec, cfg)


def _load_entity_spec(entity_cfg: EntityCfg) -> Any:
    """Load an entity source through a distinct factory/transform pipeline."""

    import mujoco

    if entity_cfg.spec_factory is not None:
        result = entity_cfg.spec_factory()
        if not isinstance(result, mujoco.MjSpec):
            raise TypeError(
                f"Entity {entity_cfg.name!r} spec_factory returned "
                f"{type(result).__name__}; expected mujoco.MjSpec"
            )
        spec = result
    else:
        spec = mujoco.MjSpec.from_file(str(entity_cfg.xml_path.resolve()))
    for index, transform in enumerate(entity_cfg.transforms):
        apply = getattr(transform, "apply", None)
        if not callable(apply):
            raise TypeError(
                f"Entity {entity_cfg.name!r} transform {index} must implement apply(spec, ...)"
            )
        result = apply(spec, entity_cfg=entity_cfg, context=None)
        if result is not None:
            if not isinstance(result, mujoco.MjSpec):
                raise TypeError(
                    f"Entity {entity_cfg.name!r} transform {index} returned "
                    f"{type(result).__name__}; expected mujoco.MjSpec or None"
                )
            spec = result
    return spec


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


def _apply_terrain_generator(spec: Any, config: SceneCfg) -> Any:
    """Apply one generic terrain generator to an assembled scene.

    This is the single terrain extension boundary used by both one-entity and
    multi-entity scenes. Terrain generators have one canonical ``compile``
    method; scene mutations belong in ``TerrainCfg.spec_fn``.
    """

    generator = config.terrain.generator
    if generator is None:
        return spec
    compile = getattr(generator, "compile", None)
    if not callable(compile):
        raise TypeError("Terrain generator must implement the canonical compile(spec) method")
    result = compile(spec)
    import mujoco

    if isinstance(result, mujoco.MjSpec):
        spec = result
    elif result is not None and _is_terrain_output(result):
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
                flat_patches=getattr(generator, "flat_patches", {}) or {},
            )
            config.terrain.generated_output = output
            config.terrain.generated_origins = origins
            config.terrain.generated_flat_patches = dict(output.flat_patches)
            if output.types is not None:
                config.terrain.generated_types = tuple(tuple(row) for row in output.types)
    return spec


def _is_terrain_output(output: Any) -> bool:
    """Accept terrain-output objects at one boundary."""

    return any(
        hasattr(output, name) for name in ("origin", "origins", "geometries", "flat_patches")
    )


def _coerce_terrain_output(output: Any) -> TerrainOutput:
    if isinstance(output, TerrainOutput):
        return output
    if not _is_terrain_output(output):
        raise TypeError(
            f"Terrain generator returned {type(output).__name__}; expected "
            "TerrainOutput-like output"
        )
    return TerrainOutput(
        origin=getattr(output, "origin", None),
        geometries=list(getattr(output, "geometries", ()) or ()),
        origins=getattr(output, "origins", None),
        types=getattr(output, "types", None),
        difficulties=getattr(output, "difficulties", None),
        flat_patches=dict(getattr(output, "flat_patches", {}) or {}),
        metadata=dict(getattr(output, "metadata", {}) or {}),
    )


def _record_terrain_output(config: SceneCfg, output: Any) -> None:
    """Normalize one patch or a complete grid into the runtime terrain table."""

    output = _coerce_terrain_output(output)
    config.terrain.generated_output = output
    config.terrain.generated_flat_patches = dict(output.flat_patches)
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
    """Normalize resources in an expanded single-scene template."""

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
        # Avoid duplicating an actuator that was already transferred into the
        # scene.
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
        Resolve selectors against the entity's complete object graph and use
        the flat collection only as a supplement for scene implementations
        that expose it.
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
        pattern = re.compile(selector.pattern or "")
        names = tuple(name for name in named_objects("body") if pattern.search(name))
        if not names:
            raise ValueError(f"Selector {selector!r} matched no entity bodies")
        return mujoco.mjtObj.mjOBJ_XBODY, names
    pattern = re.compile(selector.pattern or "") if selector.mode == "regex" else None
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
_CONTACT_REDUCTIONS = {"none": 0, "mindist": 1, "maxforce": 2, "netforce": 3}


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
        entity_name = sensor_cfg.primary_entity or sensor_cfg.entity or config.primary_entity
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
        primary = sensor_cfg.primary
        if primary is None:
            raise ValueError(f"Contact sensor {sensor_cfg.name!r} requires a primary selector")
        primary_type, primary_names = _spec_object_type_and_names(source_spec, primary)
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
        if sensor_cfg.kind not in {"custom", "raycast", "terrain_height"}:
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

    # Resolve topology-changing transforms and entity hooks before deriving
    # root names.  The transformed artifact is then reused for attachment,
    # semantic source handles, and actuator fallback copying.
    import mujoco

    prepared_specs: dict[str, Any] = {}
    for entity_name, entity_cfg in entity_items:
        entity_spec = _load_entity_spec(entity_cfg)
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
        prepared_specs[entity_name] = entity_spec

    def root_name(entity_name: str, entity_cfg: EntityCfg) -> str:
        configured = entity_cfg.root_body_name
        if configured is not None:
            try:
                prepared_specs[entity_name].body(configured)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Entity {entity_name!r} transform removed configured root body {configured!r}"
                ) from exc
            return configured
        root = ET.fromstring(prepared_specs[entity_name].to_xml()).find("worldbody/body")
        if root is None or root.get("name") is None:
            raise ValueError(
                f"Entity {entity_cfg.name!r} has no discoverable world root; "
                "set EntityCfg.root_body_name"
            )
        return str(root.get("name"))

    entity_root_names = {name: root_name(name, entity_cfg) for name, entity_cfg in entity_items}
    entity_paths = tuple(entity.xml_path.resolve() for _, entity in entity_items)
    template = config.scene_xml
    if template is None:
        template = next(
            (entity.keyframe_source for _, entity in entity_items if entity.keyframe_source),
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
                "spec_factory": _hook_identity(entity_cfg.spec_factory),
                "spec_fn": _hook_identity(entity_cfg.spec_fn),
                "transforms": tuple(
                    repr(getattr(transform, "cache_key", transform))
                    for transform in entity_cfg.transforms
                ),
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
    if (
        output_path.is_file()
        and config.terrain.generator is None
        and config.terrain.kind != "generator"
    ):
        return output_path, entity_paths

    # MjSpec.attach is the important part of this implementation.  It applies
    # the same prefix to every entity-local name (bodies, joints, geoms,
    # sites, sensors, actuators, materials, defaults, and references), so two
    # copies of the same asset remain independent.
    world_root = _copy_world_template(
        template,
        set(entity_root_names.values()),
    )
    scene_spec = mujoco.MjSpec.from_string(ET.tostring(world_root, encoding="unicode"))
    source_specs: dict[str, Any] = {}
    for entity_name, entity_cfg in entity_items:
        entity_spec = prepared_specs[entity_name]
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
        actuator_source = mujoco.MjSpec.from_string(entity_spec.to_xml())
        actuator_count_before = len(scene_spec.actuators)
        for key in list(entity_spec.keys):
            entity_spec.delete(key)
        frame = scene_spec.worldbody.add_frame()
        scene_spec.attach(entity_spec, prefix=f"{entity_name}/", frame=frame)
        # Current MuJoCo transfers actuators through attach; keep a defensive
        # fallback for versions/backends that only attach body-local objects.
        if len(scene_spec.actuators) == actuator_count_before:
            _copy_entity_actuators(scene_spec, actuator_source, entity_name)

    if config.terrain.kind == "plane" and not any(
        geom.name == "floor" for geom in scene_spec.geoms
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
                scene_spec.delete(geom)
        if config.terrain.generator is not None:
            scene_spec = _apply_terrain_generator(scene_spec, config)
        else:
            raise TypeError("TerrainCfg(kind='generator') requires a typed terrain generator")
    scene_spec = _apply_spec_hook(scene_spec, config.terrain.spec_fn, config.terrain, "Terrain")
    scene_spec = _apply_spec_hook(scene_spec, config.spec_fn, config, "Scene")
    _add_configured_sensors(scene_spec, config, source_specs, prefix_entities=True)
    _edit_custom_sensor_specs(scene_spec, config)
    output_path.write_text(scene_spec.to_xml(), encoding="utf-8")
    return output_path, entity_paths


def _materialize_single_scene(scene_path: Path, config: SceneCfg) -> Path:
    """Add configured sensors to one explicit, unprefixed scene template."""

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
                "entity_transforms": {
                    name: tuple(
                        repr(getattr(transform, "cache_key", transform))
                        for transform in entity.transforms
                    )
                    for name, entity in config.entities.items()
                },
                "terrain_spec_fn": _hook_identity(config.terrain.spec_fn),
                "terrain_generator": _hook_identity(config.terrain.generator),
                "scene_spec_fn": _hook_identity(config.spec_fn),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output_dir = Path(tempfile.gettempdir()) / "microduck_rl_torch" / "scenes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"single_{digest}.xml"
    if output_path.is_file() and config.terrain.kind != "generator":
        return output_path
    spec = mujoco.MjSpec.from_file(str(scene_path))
    source_specs = {
        name: mujoco.MjSpec.from_file(str(entity.xml_path.resolve()))
        for name, entity in config.entities.items()
    }
    _normalize_scene_resources(spec, source_paths)
    for entity_cfg in config.entities.values():
        _run_spec_hook(entity_cfg.spec_fn, spec, entity_cfg)
    if config.terrain.kind == "generator" and config.terrain.generator is not None:
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
    explicit unprefixed template remains a valid scene input.
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
        # Entity XMLs are the canonical assets. Keyframe sources are only used
        # for entity-authored initialization data; SceneCfg.scene_xml owns the
        # world/template layer.
        entity_paths = {name: entity.xml_path.resolve() for name, entity in config.entities.items()}
        composed = False
        source_paths = tuple(entity_paths.values())
        terrain = config.terrain
        # A scene XML is only a world/template layer once there is more than
        # one configured entity.  Never silently accept a hand-written XML
        # that contains a different subset of the declared entities.
        primary_entity = next(iter(config.entities.values()))
        keyframe_source = primary_entity.keyframe_source
        single_wrapper_mismatch = (
            len(entity_paths) == 1
            and config.scene_xml is not None
            and keyframe_source is not None
            and config.scene_xml.resolve() != keyframe_source.resolve()
        ) or any(
            entity.spec_fn is not None or entity.spec_factory is not None or bool(entity.transforms)
            for entity in config.entities.values()
        )
        canonical_single_entity_scene = (
            len(entity_paths) == 1
            and config.scene_xml is None
            and terrain.generator is None
            and terrain.kind == "plane"
        )
        if (
            len(entity_paths) > 1
            or single_wrapper_mismatch
            or canonical_single_entity_scene
            or terrain.kind == "generator"
        ):
            xml_path, source_paths = _compose_entity_scene(config)
            composed = True
        elif config.scene_xml is not None:
            xml_path = config.scene_xml.resolve()
        elif len(entity_paths) == 1:
            xml_path = next(iter(entity_paths.values()))
        if (
            terrain.kind != "plane"
            and not composed
            and config.scene_xml is None
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
        # For a single unprefixed scene template, materialize newly declared
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
        # A typed terrain generator already materializes the single-entity
        # source above, including configured sensors. Do not feed that output
        # back through the sensor authoring pass a second time.
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
        # A typed TerrainGeneratorCfg is the runtime declaration.  Runtime
        # placement consumes the same shape/scale used by scene compilation.
        generator_cfg = self.config.generator
        if generator_cfg is not None and hasattr(generator_cfg, "cfg"):
            generator_cfg = generator_cfg.cfg

        def generator_value(name: str, default: Any) -> Any:
            if isinstance(generator_cfg, Mapping):
                return generator_cfg.get(name, default)
            return getattr(generator_cfg, name, default) if generator_cfg is not None else default

        proportions = None
        if generator_cfg is not None:
            sub_terrains = getattr(generator_cfg, "sub_terrains", None)
            if isinstance(sub_terrains, Mapping) and sub_terrains:
                values = np.asarray(
                    [
                        float(getattr(sub_cfg, "proportion", 1.0))
                        for sub_cfg in sub_terrains.values()
                    ],
                    dtype=np.float64,
                )
                if (values < 0).any() or float(values.sum()) <= 0:
                    raise ValueError("Terrain proportions must be non-negative and non-zero")
                proportions = values / values.sum()
        self._proportions = proportions

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
        rows = int(generator_value("num_rows", 1))
        cols = int(generator_value("num_cols", 1))
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
        width_x, width_y = tuple(float(value) for value in generator_value("size", (8.0, 8.0)))
        if width_x <= 0 or width_y <= 0:
            raise ValueError("Terrain size must contain two positive values")
        if self.env_spacing <= 0:
            raise ValueError("Scene env_spacing must be positive")
        # TerrainOutput follows mjlab's indexing: rows are the x/difficulty
        # axis and columns are the y/type axis.  Origins are patch centers for
        # the synthetic fallback table; generated outputs override them with
        # their actual support points below.
        xs = torch.linspace(
            -width_x / 2 + width_x / (2 * rows),
            width_x / 2 - width_x / (2 * rows),
            rows,
            device=self.device,
        )
        ys = torch.linspace(
            -width_y / 2 + width_y / (2 * cols),
            width_y / 2 - width_y / (2 * cols),
            cols,
            device=self.device,
        )
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
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
        self.difficulty_table = self.difficulties[:, None].expand(rows, cols).clone()
        if output is not None and output.difficulties is not None:
            difficulties = torch.as_tensor(
                output.difficulties, dtype=torch.float32, device=self.device
            )
            if difficulties.ndim == 1 and difficulties.shape[0] == rows:
                self.difficulties = difficulties
                self.difficulty_table = difficulties[:, None].expand(rows, cols).clone()
            elif difficulties.ndim == 2 and difficulties.shape == (rows, cols):
                self.difficulty_table = difficulties
                self.difficulties = difficulties[:, 0]
            else:
                raise ValueError(
                    "TerrainOutput.difficulties must have shape (rows,) or (rows, cols), "
                    f"got {tuple(difficulties.shape)}"
                )
        self.type_names = self.config.generated_types
        self.flat_patches = dict(self.config.generated_flat_patches)
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
        self._origin_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
        # Assign terrain origins once when an environment is initialized. A
        # normal reset must reset task state, not silently move an environment
        # to a different terrain. Explicit curriculum progression and
        # ``randomize_env_origins`` are the only reassignment paths afterward.
        uninitialized = ids[~self._origin_initialized[ids]]
        if uninitialized.numel() == 0:
            self.env_origins[ids] = self.origins[self.levels[ids], self.types[ids]]
            return

        max_level = self.origins.shape[0] - 1
        initial = self.config.max_init_level
        if initial is None:
            initial = max_level
        if initial < 0:
            raise ValueError("Terrain max_init_level must be non-negative")
        level_high = min(initial, max_level) + 1
        for index in uninitialized.tolist():
            generator = self._generators[index] or self._generator
            if self.config.kind == "plane" and self.origins.shape[1] == self.num_envs:
                self.levels[index] = 0
                self.types[index] = index
            else:
                self.levels[index] = torch.randint(
                    0, level_high, (), generator=generator, device=self.device
                )

        if self.config.kind != "plane" or self.origins.shape[1] != self.num_envs:
            if env_ids is None and uninitialized.numel() == self.num_envs:
                if (
                    self._proportions is not None
                    and len(self._proportions) == self.origins.shape[1]
                ):
                    counts = np.ones(len(self._proportions), dtype=np.int64)
                    remaining = self.num_envs - len(counts)
                    if remaining < 0:
                        counts = np.zeros(len(self._proportions), dtype=np.int64)
                        remaining = self.num_envs
                    if remaining > 0:
                        ideal = self._proportions * remaining
                        floor = np.floor(ideal).astype(np.int64)
                        counts += floor
                        leftover = remaining - int(floor.sum())
                        if leftover:
                            counts[np.argsort(-(ideal - floor))[:leftover]] += 1
                    assigned_types = torch.repeat_interleave(
                        torch.arange(self.origins.shape[1], device=self.device),
                        torch.as_tensor(counts, dtype=torch.long, device=self.device),
                    )
                else:
                    # Use even column allocation when proportional curriculum
                    # allocation is not applicable.
                    assigned_types = torch.div(
                        torch.arange(self.num_envs, device=self.device),
                        self.num_envs / self.origins.shape[1],
                        rounding_mode="floor",
                    ).to(torch.long)
                self.types[uninitialized] = assigned_types
            else:
                for index in uninitialized.tolist():
                    generator = self._generators[index] or self._generator
                    self.types[index] = torch.randint(
                        0, self.origins.shape[1], (), generator=generator, device=self.device
                    )
        self._origin_initialized[uninitialized] = True
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

    def randomize_env_origins(
        self, env_ids: torch.Tensor | slice, *, seed: int | None = None
    ) -> None:
        """Move selected environments to random generated terrain cells.

        This is intentionally separate from ``reset``. Plane placement is
        deterministic and has
        no terrain cell assignment to randomize.
        """

        if self.config.kind == "plane" and self.origins.shape[1] == self.num_envs:
            return
        if seed is not None:
            self.set_seed(seed, env_ids)
        ids = self._ids(env_ids)
        rows, cols = self.origins.shape[:2]
        if ids.numel() == 0:
            return
        levels = torch.stack(
            [
                torch.randint(
                    0,
                    rows,
                    (),
                    generator=self._generators[int(index)] or self._generator,
                    device=self.device,
                )
                for index in ids.tolist()
            ]
        )
        types = torch.stack(
            [
                torch.randint(
                    0,
                    cols,
                    (),
                    generator=self._generators[int(index)] or self._generator,
                    device=self.device,
                )
                for index in ids.tolist()
            ]
        )
        self.levels[ids] = levels
        self.types[ids] = types
        self._origin_initialized[ids] = True
        self.env_origins[ids] = self.origins[levels, types]

    @property
    def terrain_levels(self) -> torch.Tensor:
        return self.levels

    @property
    def terrain_types(self) -> torch.Tensor:
        return self.types

    def flat_patch(self, name: str) -> Any:
        """Return a named generated flat-patch table or raise clearly."""

        try:
            return self.flat_patches[name]
        except KeyError as exc:
            raise KeyError(f"Terrain flat-patch set {name!r} is not available") from exc

    def update_env_origins(
        self,
        env_ids: torch.Tensor,
        move_up: torch.Tensor,
        move_down: torch.Tensor,
    ) -> None:
        """Apply per-environment terrain progression."""

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
