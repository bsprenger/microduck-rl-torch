"""First-class, semantic sensor management.

The upstream scene owns sensors and terms consume resolved sensor handles.  The
Torch backend has no mjlab scene graph, so this module provides the equivalent
contract over a compiled MuJoCo model: task configuration names a sensor,
``SensorManager`` resolves all model addresses once, and task terms read the
named value without touching MuJoCo indices.
"""

from __future__ import annotations

import copy
import inspect
import re
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch

from .dispatch import construct, invoke_compatible
from .model import (
    EntityView,
    ModelBundle,
    _body_descendants,
    _joint_qpos_width,
    _joint_qvel_width,
    _resolve_selector,
    _scoped_name,
)
from .scene import EntityCfg, SemanticSelector, SensorCfg

mujoco_api: Any = mujoco


@dataclass
class ContactData:
    """Structured contact result matching mjlab's public contact fields."""

    found: torch.Tensor | None = None
    force: torch.Tensor | None = None
    torque: torch.Tensor | None = None
    dist: torch.Tensor | None = None
    pos: torch.Tensor | None = None
    normal: torch.Tensor | None = None
    tangent: torch.Tensor | None = None
    current_air_time: torch.Tensor | None = None
    last_air_time: torch.Tensor | None = None
    current_contact_time: torch.Tensor | None = None
    last_contact_time: torch.Tensor | None = None
    force_history: torch.Tensor | None = None
    torque_history: torch.Tensor | None = None
    dist_history: torch.Tensor | None = None


class Sensor:
    """First-class sensor lifecycle object.

    The manager resolves addresses and owns the backend buffers, while this
    object is the task-facing typed handle.  ``edit_spec`` is intentionally a
    hook: custom sensors can add their MuJoCo declarations before compilation
    and still use the same initialize/reset/update/data contract.
    """

    requires_sensor_context = False

    def __init__(self, cfg: SensorCfg) -> None:
        self.cfg = cfg
        self._manager: SensorManager | None = None
        self._name = cfg.name
        self._reader: Any | None = None

    def edit_spec(self, scene_spec: Any, entities: dict[str, EntityCfg]) -> None:
        reader = self.cfg.reader
        if inspect.isclass(reader) and callable(getattr(reader, "edit_spec", None)):
            reader = construct(reader, self.cfg)
        edit_spec = getattr(reader, "edit_spec", None)
        if callable(edit_spec):
            invoke_compatible(
                edit_spec,
                (((scene_spec, entities), {}), ((scene_spec,), {}), ((), {})),
            )

    def initialize(self, *_args: Any, **_kwargs: Any) -> None:
        initialize = getattr(self._reader, "initialize", None)
        if callable(initialize):
            invoke_compatible(
                initialize,
                ((_args, _kwargs), ((), {})),
            )

    def _bind(self, manager: SensorManager) -> None:
        self._manager = manager

    def _set_reader(self, reader: Any | None) -> None:
        self._reader = reader

    @property
    def data(self) -> Any:
        if self._manager is None:
            raise RuntimeError("Sensor is not attached to a SensorManager")
        return self._manager.data(self._name)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        reset = getattr(self._reader, "reset", None)
        if callable(reset):
            invoke_compatible(
                reset,
                (((env_ids,), {}), ((), {"env_ids": env_ids}), ((), {})),
            )

    def update(self, dt: float) -> None:
        update = getattr(self._reader, "update", None)
        if callable(update):
            invoke_compatible(update, (((dt,), {}), ((), {})))

    def debug_vis(self, visualizer: Any) -> None:
        debug_vis = getattr(self._reader, "debug_vis", None)
        if callable(debug_vis):
            invoke_compatible(debug_vis, (((visualizer,), {}), ((), {})))

    def compute_first_contact(self, dt: float) -> torch.Tensor:
        if self._manager is None:
            raise RuntimeError("Sensor is not attached to a SensorManager")
        return self._manager.compute_first_contact(self._name, dt)

    def compute_first_air(self, dt: float) -> torch.Tensor:
        if self._manager is None:
            raise RuntimeError("Sensor is not attached to a SensorManager")
        return self._manager.compute_first_air(self._name, dt)


@dataclass(frozen=True)
class ObjRef:
    """Reference to one named MuJoCo frame in an entity namespace."""

    type: str
    name: str
    entity: str | None = None


@dataclass(frozen=True)
class GridPatternCfg:
    """Parallel local rays arranged on a rectangular grid."""

    size: tuple[float, float] = (1.0, 1.0)
    resolution: float = 0.1
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def generate_rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.resolution <= 0:
            raise ValueError("Ray grid resolution must be positive")
        x = torch.arange(
            -self.size[0] / 2,
            self.size[0] / 2 + self.resolution / 2,
            self.resolution,
        )
        y = torch.arange(
            -self.size[1] / 2,
            self.size[1] / 2 + self.resolution / 2,
            self.resolution,
        )
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        offsets = torch.zeros((grid_x.numel(), 3), dtype=torch.float32)
        offsets[:, 0] = grid_x.reshape(-1)
        offsets[:, 1] = grid_y.reshape(-1)
        direction = torch.tensor(self.direction, dtype=torch.float32)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-12)
        return offsets, direction.expand_as(offsets).clone()


@dataclass(frozen=True)
class RingPatternCfg:
    """Concentric local ray offsets with a shared direction."""

    @dataclass(frozen=True)
    class Ring:
        radius: float
        num_samples: int

    rings: tuple[Ring, ...] = ()
    include_center: bool = True
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)

    @classmethod
    def single_ring(
        cls,
        radius: float = 0.04,
        num_samples: int = 2,
        *,
        include_center: bool = True,
        direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
    ) -> RingPatternCfg:
        return cls(
            rings=(cls.Ring(radius=radius, num_samples=num_samples),),
            include_center=include_center,
            direction=direction,
        )

    def generate_rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        offsets: list[tuple[float, float, float]] = []
        if self.include_center:
            offsets.append((0.0, 0.0, 0.0))
        for ring in self.rings:
            if ring.radius < 0 or ring.num_samples < 1:
                raise ValueError("Ray rings need a non-negative radius and positive sample count")
            for index in range(ring.num_samples):
                angle = 2.0 * np.pi * index / ring.num_samples
                offsets.append((ring.radius * np.cos(angle), ring.radius * np.sin(angle), 0.0))
        if not offsets:
            raise ValueError("Ray pattern must contain at least one ray")
        offset_tensor = torch.tensor(offsets, dtype=torch.float32)
        direction = torch.tensor(self.direction, dtype=torch.float32)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-12)
        return offset_tensor, direction.expand_as(offset_tensor).clone()


@dataclass(frozen=True)
class RayCastSensorCfg(SensorCfg):
    """Upstream-shaped raycast sensor configuration."""

    kind: str = "raycast"
    frame: ObjRef | tuple[ObjRef, ...] = ()
    pattern: GridPatternCfg | RingPatternCfg = GridPatternCfg()
    ray_alignment: str = "base"
    max_distance: float = 10.0
    exclude_parent_body: bool = True
    include_geom_groups: tuple[int, ...] | None = (0, 1, 2)
    debug_vis: bool = False

    def build(self) -> Sensor:
        return RayCastSensor(self)


@dataclass(frozen=True)
class TerrainHeightSensorCfg(RayCastSensorCfg):
    """Raycast sensor that additionally exposes frame-to-terrain heights."""

    kind: str = "terrain_height"
    reduction: str = "min"

    def build(self) -> Sensor:
        return TerrainHeightSensor(self)


@dataclass
class RayCastData:
    """Resolved ray distances and world-frame hit geometry."""

    distances: torch.Tensor
    normals_w: torch.Tensor
    hit_pos_w: torch.Tensor
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    frame_pos_w: torch.Tensor
    frame_quat_w: torch.Tensor
    heights: torch.Tensor | None = None


class RayCastSensor(Sensor):
    requires_sensor_context = True


class TerrainHeightSensor(RayCastSensor):
    pass


def _object_name(model: Any, object_type: Any, object_id: int) -> str:
    return mujoco_api.mj_id2name(model, object_type, object_id) or f"<{object_id}>"


def _resolve_joint_indices(
    model: Any,
    entity: EntityView,
    names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = set(names)
    joint_ids = (
        entity.joint_ids
        if not names
        else tuple(
            joint_id
            for joint_id in entity.joint_ids
            if _object_name(model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_id).removeprefix(
                f"{entity.name}/"
            )
            in selected
            or any(
                re.search(
                    pattern,
                    _object_name(model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_id).removeprefix(
                        f"{entity.name}/"
                    ),
                )
                for pattern in names
            )
        )
    )
    if names and not joint_ids:
        raise ValueError(f"Joint sensor selector {names!r} matched no joints in {entity.name!r}")
    qpos: list[int] = []
    qvel: list[int] = []
    for joint_id in joint_ids:
        joint_type = int(model.jnt_type[joint_id])
        qpos_start = int(model.jnt_qposadr[joint_id])
        qvel_start = int(model.jnt_dofadr[joint_id])
        qpos.extend(range(qpos_start, qpos_start + _joint_qpos_width(joint_type)))
        qvel.extend(range(qvel_start, qvel_start + _joint_qvel_width(joint_type)))
    device = entity.qpos_indices.device
    return (
        torch.tensor(qpos, dtype=torch.long, device=device),
        torch.tensor(qvel, dtype=torch.long, device=device),
    )


@dataclass(frozen=True)
class SensorHandle:
    """Resolved immutable sensor metadata exposed for introspection."""

    name: str
    kind: str
    dimension: int
    entity: str | None


@dataclass
class _ResolvedSensor:
    cfg: SensorCfg
    handle: SensorHandle
    entity: EntityView | None = None
    ids: tuple[int, ...] = ()
    qpos_indices: torch.Tensor | None = None
    qvel_indices: torch.Tensor | None = None
    primary_ids: tuple[int, ...] = ()
    secondary_ids: tuple[int, ...] | None = None
    primary_geom_ids: tuple[int, ...] = ()
    primary_geom_groups: tuple[tuple[int, ...], ...] = ()
    secondary_geom_ids: tuple[int, ...] | None = None
    exclude_geom_ids: tuple[int, ...] = ()
    reader_instance: Any | None = None
    reader_instances: dict[int, Any] = field(default_factory=dict)
    contact_slices: tuple[dict[str, slice], ...] = ()
    contact_primary_count: int = 0
    contact_data: ContactData | None = None
    ray_frames: tuple[tuple[str, int, int], ...] = ()
    ray_offsets: torch.Tensor | None = None
    ray_directions: torch.Tensor | None = None
    ray_data: RayCastData | None = None
    source_name: str | None = None


class _SensorRowEnv:
    """Scalar view used to evaluate exact native readers for one batch row."""

    def __init__(self, env: Any, index: int) -> None:
        self._env = env
        self.index = index
        self.num_envs = 1
        self.data = env.data[index]
        if hasattr(env.physics, "instances"):
            self.physics = env.physics.instances[index]
            self.bundle = self.physics.bundle
        else:
            self.physics = env.physics
            self.bundle = env.bundle

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class SensorManager:
    """Resolve, update, and expose named sensors for one environment.

    Supported built-ins cover the generic data contracts used by upstream
    tasks: named MuJoCo sensors, body pose/velocity, site position/velocity,
    joint position/velocity, and pairwise contact sensors.  A ``custom``
    sensor may provide a callable or stateful reader for backend-native
    sensors such as a raycaster; it still receives the same reset/update
    lifecycle and can be history-tracked.
    """

    def __init__(self, config: dict[str, SensorCfg], bundle: ModelBundle) -> None:
        self.config = config
        self.bundle = bundle
        self._sensors: dict[str, Sensor] = {}
        self._resolved: dict[str, _ResolvedSensor] = {}
        self._values: dict[str, torch.Tensor] = {}
        self._history: dict[str, list[torch.Tensor]] = {}
        self._air_time: dict[str, torch.Tensor] = {}
        self._contact_timing: dict[str, dict[str, torch.Tensor]] = {}
        self._contact_history: dict[str, list[ContactData]] = {}
        self._ticks: dict[str, int] = {}
        self._native_data = mujoco_api.MjData(bundle.native_model)
        self._native_data_by_model: dict[int, Any] = {id(bundle.native_model): self._native_data}
        self._initialized = False
        for name, sensor_cfg in config.items():
            if name != sensor_cfg.name:
                raise ValueError(
                    f"Sensor mapping key {name!r} != configured name {sensor_cfg.name!r}"
                )
            sensor = sensor_cfg.build()
            sensor._bind(self)
            self._sensors[name] = sensor
            self._resolved[name] = self._resolve(sensor_cfg)

    @property
    def active_sensors(self) -> tuple[str, ...]:
        return tuple(self._resolved)

    def get_handle(self, name: str) -> SensorHandle:
        try:
            return self._resolved[name].handle
        except KeyError as exc:
            raise KeyError(f"Sensor {name!r} is not configured") from exc

    def get_sensor(self, name: str) -> Sensor:
        try:
            return self._sensors[name]
        except KeyError as exc:
            raise KeyError(f"Sensor {name!r} is not configured") from exc

    def _entity(self, name: str | None) -> EntityView | None:
        if name is None:
            return None
        return self.bundle.entity(name)

    def _resolve_selector(
        self,
        selector: SemanticSelector,
        entity_cfg: EntityCfg | None,
        object_type: Any,
    ) -> tuple[int, ...]:
        # A body-subtree selector needs the compiled model and is already
        # implemented by the model resolver.  Entity scoping is enforced after
        # resolution to prevent a selector from accidentally crossing entity
        # boundaries in a composed scene.
        result = _resolve_selector(
            self.bundle.native_model,
            object_type,
            selector,
            entity_cfg.name if entity_cfg is not None else None,
        )
        if entity_cfg is None:
            return result
        entity = self.bundle.entity(entity_cfg.name)
        allowed = {
            mujoco_api.mjtObj.mjOBJ_BODY: set(entity.body_ids),
            mujoco_api.mjtObj.mjOBJ_GEOM: set(entity.geom_ids),
            mujoco_api.mjtObj.mjOBJ_SITE: set(entity.site_ids),
        }.get(object_type)
        if allowed is None:
            return result
        filtered = tuple(value for value in result if value in allowed)
        if not filtered:
            raise ValueError(
                f"Selector {selector!r} matched no {object_type!r} in entity {entity_cfg.name!r}"
            )
        return filtered

    def _resolve(self, cfg: SensorCfg) -> _ResolvedSensor:
        entity_name = cfg.entity or self.bundle.primary_entity_name
        entity = self._entity(entity_name)
        entity_cfg = self._entity_cfg(entity_name)
        kind = cfg.kind
        ids: tuple[int, ...] = ()
        qpos_indices = qvel_indices = None
        primary_ids: tuple[int, ...] = ()
        secondary_ids: tuple[int, ...] | None = None
        primary_geom_ids: tuple[int, ...] = ()
        primary_geom_groups: tuple[tuple[int, ...], ...] = ()
        secondary_geom_ids: tuple[int, ...] | None = None
        exclude_geom_ids: tuple[int, ...] = ()
        primary_object_type: Any | None = None
        secondary_object_type: Any | None = None
        ray_frames: tuple[tuple[str, int, int], ...] = ()
        ray_offsets: torch.Tensor | None = None
        ray_directions: torch.Tensor | None = None
        source_name: str | None = None
        if kind == "mujoco":
            source_candidates = [cfg.source or cfg.name]
            if entity_name is not None:
                scoped_source = f"{entity_name}/{cfg.source or cfg.name}"
                if scoped_source not in source_candidates:
                    source_candidates.append(scoped_source)
            if cfg.source is None and cfg.prefixed_name not in source_candidates:
                source_candidates.append(cfg.prefixed_name)
            source = source_candidates[0]
            sensor_id = -1
            for candidate in source_candidates:
                candidate_id = mujoco_api.mj_name2id(
                    self.bundle.native_model, mujoco_api.mjtObj.mjOBJ_SENSOR, candidate
                )
                if candidate_id >= 0 and candidate in self.bundle.sensor_slices:
                    source, sensor_id = candidate, candidate_id
                    break
            source_name = source
            if sensor_id < 0 or source not in self.bundle.sensor_slices:
                if cfg.required:
                    raise ValueError(f"Required MuJoCo sensor {source!r} was not found")
                dimension = cfg.expected_dim or 0
            else:
                dimension = int(self.bundle.native_model.sensor_dim[sensor_id])
        elif kind in {"site_position", "site_velocity"}:
            if entity is None:
                raise ValueError(f"Sensor {cfg.name!r} needs an entity")
            if cfg.selector is None and cfg.source is None:
                raise ValueError(f"Sensor {cfg.name!r} needs a site selector or source name")
            selector = cfg.selector or SemanticSelector(names=(cfg.source or "",))
            ids = self._resolve_selector(selector, entity_cfg, mujoco_api.mjtObj.mjOBJ_SITE)
            dimension = 3 * len(ids)
        elif kind in {"body_pose", "body_velocity"}:
            if entity is None:
                raise ValueError(f"Sensor {cfg.name!r} needs an entity")
            if cfg.selector is None:
                ids = (entity.root_body_id,)
            else:
                ids = self._resolve_selector(cfg.selector, entity_cfg, mujoco_api.mjtObj.mjOBJ_BODY)
            dimension = (7 if kind == "body_pose" else 6) * len(ids)
        elif kind in {"joint_position", "joint_velocity"}:
            if entity is None:
                raise ValueError(f"Sensor {cfg.name!r} needs an entity")
            qpos_indices, qvel_indices = _resolve_joint_indices(
                self.bundle.native_model, entity, cfg.joint_names
            )
            dimension = int(
                qpos_indices.numel() if kind == "joint_position" else qvel_indices.numel()
            )
        elif kind in {"raycast", "terrain_height"}:
            if not isinstance(cfg, RayCastSensorCfg):
                raise TypeError(f"Sensor {cfg.name!r} must use RayCastSensorCfg for {kind!r}")
            frames = cfg.frame if isinstance(cfg.frame, tuple) else (cfg.frame,)
            if not frames:
                raise ValueError(f"Ray sensor {cfg.name!r} needs at least one frame")
            resolved_frames: list[tuple[str, int, int]] = []
            object_types = {
                "body": mujoco_api.mjtObj.mjOBJ_BODY,
                "xbody": mujoco_api.mjtObj.mjOBJ_BODY,
                "geom": mujoco_api.mjtObj.mjOBJ_GEOM,
                "site": mujoco_api.mjtObj.mjOBJ_SITE,
            }
            for frame in frames:
                if isinstance(frame, str):
                    frame = ObjRef("site", frame, entity_name)
                if not isinstance(frame, ObjRef) or frame.type not in object_types:
                    raise ValueError(
                        f"Ray sensor {cfg.name!r} frame must be an ObjRef of type "
                        "body, xbody, geom, or site"
                    )
                frame_entity = frame.entity or entity_name
                full_name = _scoped_name(
                    self.bundle.native_model,
                    object_types[frame.type],
                    frame_entity,
                    frame.name,
                )
                object_id = mujoco_api.mj_name2id(
                    self.bundle.native_model, object_types[frame.type], full_name
                )
                if frame.type in {"body", "xbody"}:
                    body_id = int(object_id)
                elif frame.type == "geom":
                    body_id = int(self.bundle.native_model.geom_bodyid[object_id])
                else:
                    body_id = int(self.bundle.native_model.site_bodyid[object_id])
                resolved_frames.append((frame.type, int(object_id), body_id))
            ray_frames = tuple(resolved_frames)
            ray_offsets, ray_directions = cfg.pattern.generate_rays()
            if cfg.max_distance <= 0:
                raise ValueError(f"Ray sensor {cfg.name!r} max_distance must be positive")
            if cfg.ray_alignment not in {"base", "yaw", "world"}:
                raise ValueError(f"Unsupported ray alignment {cfg.ray_alignment!r}")
            dimension = len(ray_frames) * int(ray_offsets.shape[0])
            if kind == "terrain_height" and getattr(cfg, "reduction", "min") != "none":
                dimension = len(ray_frames)
        elif kind == "contact":
            if cfg.num_slots < 1:
                raise ValueError(f"Contact sensor {cfg.name!r} num_slots must be positive")
            unknown_fields = set(cfg.fields) - {
                "found",
                "force",
                "torque",
                "dist",
                "pos",
                "normal",
                "tangent",
            }
            if unknown_fields:
                raise ValueError(
                    f"Contact sensor {cfg.name!r} has unsupported fields {sorted(unknown_fields)!r}"
                )
            if cfg.reduce not in {"none", "mindist", "maxforce", "netforce", "any"}:
                raise ValueError(f"Unsupported contact reduction {cfg.reduce!r}")
            if cfg.primary is None:
                raise ValueError(f"Contact sensor {cfg.name!r} needs a primary selector")
            primary_cfg = self._entity_cfg(cfg.primary_entity or entity_name)
            primary_object_type = (
                mujoco_api.mjtObj.mjOBJ_BODY
                if cfg.primary.mode == "body_subtree"
                else mujoco_api.mjtObj.mjOBJ_GEOM
            )
            try:
                primary_ids = self._resolve_selector(cfg.primary, primary_cfg, primary_object_type)
            except ValueError:
                if cfg.primary.mode == "body_subtree":
                    raise
                primary_object_type = mujoco_api.mjtObj.mjOBJ_BODY
                primary_ids = self._resolve_selector(cfg.primary, primary_cfg, primary_object_type)
            if cfg.secondary is not None:
                secondary_cfg = (
                    self._entity_cfg(cfg.secondary_entity)
                    if cfg.secondary_entity is not None
                    else None
                )
                secondary_object_type = (
                    mujoco_api.mjtObj.mjOBJ_BODY
                    if cfg.secondary.mode == "body_subtree"
                    else mujoco_api.mjtObj.mjOBJ_GEOM
                )
                try:
                    secondary_ids = self._resolve_selector(
                        cfg.secondary, secondary_cfg, secondary_object_type
                    )
                except ValueError:
                    secondary_object_type = mujoco_api.mjtObj.mjOBJ_BODY
                    secondary_ids = self._resolve_selector(
                        cfg.secondary, secondary_cfg, secondary_object_type
                    )
                if len(secondary_ids) > 1 and cfg.secondary_policy == "error":
                    raise ValueError(
                        f"Contact sensor {cfg.name!r} secondary selector matched "
                        "multiple objects with secondary_policy='error'"
                    )
                if len(secondary_ids) > 1 and cfg.secondary_policy == "first":
                    secondary_ids = secondary_ids[:1]
            exclude_ids: list[int] = []
            for selector in cfg.exclude:
                exclude_cfg = self._entity_cfg(entity_name) if entity_name else None
                try:
                    exclude_type = (
                        mujoco_api.mjtObj.mjOBJ_BODY
                        if selector.mode == "body_subtree"
                        else mujoco_api.mjtObj.mjOBJ_GEOM
                    )
                    resolved_exclude = self._resolve_selector(selector, exclude_cfg, exclude_type)
                except ValueError:
                    exclude_type = mujoco_api.mjtObj.mjOBJ_BODY
                    resolved_exclude = self._resolve_selector(selector, exclude_cfg, exclude_type)
                exclude_ids.extend(self._contact_geom_ids(resolved_exclude, exclude_type))
            exclude_geom_ids = tuple(sorted(set(exclude_ids)))
            dimension = (
                len(primary_ids)
                * (cfg.num_slots if cfg.reduce == "none" else 1)
                * sum(self._contact_field_dim(field) for field in cfg.fields)
            )
            primary_geom_groups = self._contact_geom_groups(primary_ids, primary_object_type)
            primary_geom_ids = self._contact_geom_ids(primary_ids, primary_object_type)
            secondary_geom_ids = (
                None
                if secondary_ids is None
                else self._contact_geom_ids(secondary_ids, secondary_object_type)
            )
        elif kind == "custom":
            if cfg.reader is None:
                raise ValueError(f"Custom sensor {cfg.name!r} needs a reader")
            dimension = cfg.expected_dim or 0
        else:
            raise ValueError(f"Unsupported sensor kind {kind!r}")
        if cfg.expected_dim is not None and dimension != cfg.expected_dim:
            raise ValueError(
                f"Sensor {cfg.name!r} has dimension {dimension}; expected {cfg.expected_dim}"
            )
        if kind == "contact":
            entity = self._entity(cfg.primary_entity or entity_name)
        handle = SensorHandle(cfg.name, kind, dimension, entity_name)
        contact_slices: list[dict[str, slice]] = []
        if kind == "contact":
            for index in range(len(primary_ids)):
                fields: dict[str, slice] = {}
                for field_name in cfg.fields:
                    internal_name = f"__contact__{cfg.name}__{index}__{field_name}"
                    sensor_slice = self.bundle.sensor_slices.get(internal_name)
                    if sensor_slice is not None:
                        fields[field_name] = sensor_slice
                contact_slices.append(fields)
            if secondary_geom_ids is not None and len(secondary_geom_ids) > 1:
                # A MuJoCo contact sensor supports one ref object.  Keep the
                # declaration as a cheap broad-phase aid, but use the native
                # contact graph below to preserve the full multi-object
                # selector semantics.
                contact_slices = []
            if exclude_geom_ids:
                # Native graph evaluation is required to apply arbitrary
                # exclusion selectors exactly; MuJoCo's generated sensor
                # declaration has no equivalent multi-object exclude list.
                contact_slices = []
            if not any(contact_slices):
                # A caller can construct a bundle directly without a
                # SceneBuilder.  Retain the semantic fallback for that case.
                contact_slices = []
        return _ResolvedSensor(
            cfg=cfg,
            handle=handle,
            entity=entity,
            ids=ids,
            qpos_indices=qpos_indices,
            qvel_indices=qvel_indices,
            primary_ids=primary_ids,
            secondary_ids=secondary_ids,
            primary_geom_ids=primary_geom_ids,
            primary_geom_groups=primary_geom_groups,
            secondary_geom_ids=secondary_geom_ids,
            exclude_geom_ids=exclude_geom_ids,
            contact_slices=tuple(contact_slices),
            contact_primary_count=len(primary_ids),
            ray_frames=ray_frames,
            ray_offsets=ray_offsets,
            ray_directions=ray_directions,
            source_name=source_name,
        )

    def _entity_cfg(self, name: str | None) -> EntityCfg:
        if name is None:
            raise ValueError("An entity name is required for this sensor")
        # The bundle retains the configured entity map through the resolved
        # views.  ``entity_cfg`` is the robot; additional configs are attached
        # by the environment before manager construction.
        configs = getattr(self.bundle, "entity_configs", {})
        cfg = configs.get(name)
        if cfg is None and name == self.bundle.entity_cfg.name:
            cfg = self.bundle.entity_cfg
        if cfg is None:
            raise KeyError(f"No configuration for scene entity {name!r}")
        return cfg

    @staticmethod
    def _contact_fields(cfg: SensorCfg, slots: int) -> tuple[str, ...]:
        if cfg.reduce == "none":
            return tuple(field for field in cfg.fields for _ in range(slots))
        return cfg.fields

    def _contact_geom_ids(
        self, object_ids: tuple[int, ...], object_type: Any | None
    ) -> tuple[int, ...]:
        """Expand body/subtree matches to the geoms participating in contacts."""

        return tuple(
            geom_id
            for group in self._contact_geom_groups(object_ids, object_type)
            for geom_id in group
        )

    def _contact_geom_groups(
        self, object_ids: tuple[int, ...], object_type: Any | None
    ) -> tuple[tuple[int, ...], ...]:
        """Return one contact-geometry group for each selected object."""

        if object_type not in {
            mujoco_api.mjtObj.mjOBJ_BODY,
            mujoco_api.mjtObj.mjOBJ_XBODY,
        }:
            return tuple((int(object_id),) for object_id in object_ids)
        return tuple(
            tuple(
                geom_id
                for geom_id in range(int(self.bundle.native_model.ngeom))
                if int(self.bundle.native_model.geom_bodyid[geom_id])
                in _body_descendants(self.bundle.native_model, int(body_id))
            )
            for body_id in object_ids
        )

    @staticmethod
    def _contact_field_dim(field: str) -> int:
        return {
            "found": 1,
            "force": 3,
            "torque": 3,
            "dist": 1,
            "pos": 3,
            "normal": 3,
            "tangent": 3,
        }.get(field, 0)

    def _read_contact_from_sensors(self, env: Any, resolved: _ResolvedSensor) -> ContactData | None:
        """Read exact MuJoCo contact sensors generated by SceneBuilder."""

        if not resolved.contact_slices:
            return None
        cfg = resolved.cfg
        values: dict[str, list[torch.Tensor]] = {field: [] for field in cfg.fields}
        slots = cfg.num_slots if cfg.reduce == "none" else 1
        for fields in resolved.contact_slices:
            for field_name in cfg.fields:
                sensor_slice = fields.get(field_name)
                if sensor_slice is None:
                    return None
                raw = env.data.sensordata[..., sensor_slice]
                dim = self._contact_field_dim(field_name)
                values[field_name].append(raw.reshape(slots, dim))
        data = ContactData()
        for field_name, chunks in values.items():
            stacked = torch.stack(chunks, dim=0)
            if self._contact_field_dim(field_name) == 1:
                stacked = stacked.squeeze(-1)
            setattr(data, field_name, stacked)
        return data

    def _read_raycast(self, env: Any, resolved: _ResolvedSensor) -> RayCastData:
        """Evaluate semantic rays against the current MuJoCo state.

        ``mujoco-torch`` does not currently expose MuJoCo's BVH ray query.
        The native query is exact and is used only for this sensor read; it
        never becomes the simulation state or a second physics step.
        """

        cfg = resolved.cfg
        if not isinstance(cfg, RayCastSensorCfg):  # pragma: no cover - guarded in resolve
            raise TypeError("Raycast resolution requires RayCastSensorCfg")
        if resolved.ray_offsets is None or resolved.ray_directions is None:
            raise RuntimeError(f"Ray sensor {cfg.name!r} has no generated ray pattern")
        data = env.data
        native_model, native_data = self._native_context(env)
        native_data.qpos[:] = data.qpos.detach().cpu().numpy()
        native_data.qvel[:] = data.qvel.detach().cpu().numpy()
        if native_data.ctrl.shape == data.ctrl.shape:
            native_data.ctrl[:] = data.ctrl.detach().cpu().numpy()
        mujoco_api.mj_forward(native_model, native_data)

        group = None
        if cfg.include_geom_groups is not None:
            group = np.zeros(6, dtype=np.uint8)
            for value in cfg.include_geom_groups:
                if value < 0 or value > 5:
                    raise ValueError("Raycast geometry groups must be in [0, 5]")
                group[value] = 1
        offsets = resolved.ray_offsets.detach().cpu().numpy()
        directions = resolved.ray_directions.detach().cpu().numpy()
        distances: list[float] = []
        normals: list[torch.Tensor] = []
        hits: list[torch.Tensor] = []
        frame_positions: list[torch.Tensor] = []
        frame_quaternions: list[torch.Tensor] = []
        ray_count = offsets.shape[0]

        for frame_type, object_id, body_id in resolved.ray_frames:
            if frame_type == "body" or frame_type == "xbody":
                frame_pos = native_data.xpos[object_id]
                frame_mat = native_data.xmat[object_id].reshape(3, 3)
            elif frame_type == "geom":
                frame_pos = native_data.geom_xpos[object_id]
                frame_mat = native_data.geom_xmat[object_id].reshape(3, 3)
            else:
                frame_pos = native_data.site_xpos[object_id]
                frame_mat = native_data.site_xmat[object_id].reshape(3, 3)
            frame_positions.append(
                torch.as_tensor(frame_pos, dtype=self.bundle.dtype, device=self.bundle.device)
            )
            quat = np.zeros(4, dtype=np.float64)
            mujoco_api.mju_mat2Quat(quat, frame_mat.reshape(-1))
            frame_quaternions.append(
                torch.as_tensor(quat, dtype=self.bundle.dtype, device=self.bundle.device)
            )

            if cfg.ray_alignment == "base":
                rotation = frame_mat
            elif cfg.ray_alignment == "world":
                rotation = np.eye(3, dtype=np.float64)
            else:
                yaw = np.arctan2(frame_mat[1, 0], frame_mat[0, 0])
                rotation = np.array(
                    [
                        [np.cos(yaw), -np.sin(yaw), 0.0],
                        [np.sin(yaw), np.cos(yaw), 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
            world_origins = np.asarray(frame_pos, dtype=np.float64) + offsets @ rotation.T
            world_directions = directions @ rotation.T
            for origin, direction in zip(world_origins, world_directions, strict=True):
                geomid = np.zeros(1, dtype=np.int32)
                normal = np.zeros(3, dtype=np.float64)
                distance = float(
                    mujoco_api.mj_ray(
                        native_model,
                        native_data,
                        origin,
                        direction,
                        group,
                        True,
                        body_id if cfg.exclude_parent_body else -1,
                        geomid,
                        normal,
                    )
                )
                if distance < 0.0 or distance > cfg.max_distance:
                    distance = -1.0
                    normal[:] = 0.0
                hit = origin if distance < 0.0 else origin + distance * direction
                distances.append(distance)
                normals.append(
                    torch.as_tensor(normal, dtype=self.bundle.dtype, device=self.bundle.device)
                )
                hits.append(
                    torch.as_tensor(hit, dtype=self.bundle.dtype, device=self.bundle.device)
                )

        distance_tensor = torch.tensor(
            distances, dtype=self.bundle.dtype, device=self.bundle.device
        )
        normal_tensor = torch.stack(normals)
        hit_tensor = torch.stack(hits)
        frame_position_tensor = torch.stack(frame_positions)
        frame_quaternion_tensor = torch.stack(frame_quaternions)
        ray_data = RayCastData(
            distances=distance_tensor,
            normals_w=normal_tensor,
            hit_pos_w=hit_tensor,
            pos_w=frame_position_tensor[0],
            quat_w=frame_quaternion_tensor[0],
            frame_pos_w=frame_position_tensor,
            frame_quat_w=frame_quaternion_tensor,
        )
        if cfg.kind == "terrain_height":
            heights = frame_position_tensor[:, 2].repeat_interleave(ray_count) - hit_tensor[:, 2]
            heights = torch.where(
                distance_tensor < 0,
                torch.full_like(heights, cfg.max_distance),
                heights,
            ).reshape(len(resolved.ray_frames), ray_count)
            reduction = getattr(cfg, "reduction", "min")
            if reduction == "min":
                heights = heights.min(dim=-1).values
            elif reduction == "max":
                heights = heights.max(dim=-1).values
            elif reduction == "mean":
                heights = heights.mean(dim=-1)
            elif reduction != "none":
                raise ValueError(f"Unknown terrain height reduction {reduction!r}")
            ray_data.heights = heights
        resolved.ray_data = ray_data
        return ray_data

    @staticmethod
    def _flatten_contact(data: ContactData, fields: tuple[str, ...]) -> torch.Tensor:
        chunks = [getattr(data, field) for field in fields]
        present = [value.reshape(-1) for value in chunks if value is not None]
        if not present:
            return torch.zeros(0)
        return torch.cat(present)

    def _contact_mask(self, data: Any, resolved: _ResolvedSensor) -> torch.Tensor:
        geom1 = data.contact.geom1
        geom2 = data.contact.geom2
        slots = torch.arange(geom1.shape[-1], device=geom1.device)
        count = torch.as_tensor(data.ncon, dtype=torch.long, device=geom1.device).clamp(
            0, geom1.shape[-1]
        )
        valid_slots = slots < count[..., None] if count.ndim else slots < count
        valid = valid_slots & (data.contact.dist <= data.contact.includemargin)
        if resolved.cfg.reduce == "none":
            valid = valid & (slots < resolved.cfg.num_slots)
        primary = torch.as_tensor(resolved.primary_geom_ids, dtype=geom1.dtype, device=geom1.device)
        match_primary = torch.isin(geom1, primary) | torch.isin(geom2, primary)
        if resolved.secondary_geom_ids is None:
            result = valid & match_primary
        else:
            secondary = torch.as_tensor(
                resolved.secondary_geom_ids, dtype=geom1.dtype, device=geom1.device
            )
            result = valid & (
                (torch.isin(geom1, primary) & torch.isin(geom2, secondary))
                | (torch.isin(geom2, primary) & torch.isin(geom1, secondary))
            )
        if resolved.exclude_geom_ids:
            excluded = torch.as_tensor(
                resolved.exclude_geom_ids, dtype=geom1.dtype, device=geom1.device
            )
            result &= ~(torch.isin(geom1, excluded) | torch.isin(geom2, excluded))
        return result

    def _read_contact_native(self, env: Any, resolved: _ResolvedSensor) -> ContactData:
        """Read semantic contacts directly when a selector spans many objects.

        MuJoCo's XML contact sensor can name only one reference object.  The
        native contact graph is therefore the authoritative implementation for
        multi-object filters and for body/subtree matches.  It preserves the
        same public per-primary/per-slot layout as the generated sensor path.
        """

        data = env.data
        native_model, native_data = self._native_context(env)
        native_data.qpos[:] = data.qpos.detach().cpu().numpy()
        native_data.qvel[:] = data.qvel.detach().cpu().numpy()
        if native_data.ctrl.shape == data.ctrl.shape:
            native_data.ctrl[:] = data.ctrl.detach().cpu().numpy()
        mujoco_api.mj_forward(native_model, native_data)

        primary_groups = resolved.primary_geom_groups
        secondary = (
            None if resolved.secondary_geom_ids is None else set(resolved.secondary_geom_ids)
        )
        contacts_by_primary: list[list[dict[str, Any]]] = []
        primary_sets = [set(group) for group in primary_groups]
        for primary_set in primary_sets:
            records: list[dict[str, Any]] = []
            for contact_id in range(int(native_data.ncon)):
                contact = native_data.contact[contact_id]
                # Keep the native graph path bit-for-bit aligned with the
                # tensor contact-mask path: MuJoCo may retain speculative
                # contacts whose distance is outside the configured margin.
                if float(contact.dist) > float(contact.includemargin):
                    continue
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                if geom1 in primary_set:
                    other = geom2
                elif geom2 in primary_set:
                    other = geom1
                else:
                    continue
                if secondary is not None and other not in secondary:
                    continue
                if resolved.exclude_geom_ids and (
                    geom1 in resolved.exclude_geom_ids or geom2 in resolved.exclude_geom_ids
                ):
                    continue
                force6 = np.zeros(6, dtype=np.float64)
                mujoco_api.mj_contactForce(native_model, native_data, contact_id, force6)
                frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
                records.append(
                    {
                        "found": 1.0,
                        "force": force6[:3].copy(),
                        "torque": force6[3:].copy(),
                        "dist": float(contact.dist),
                        "pos": np.asarray(contact.pos, dtype=np.float64).copy(),
                        "normal": frame[0].copy(),
                        "tangent": frame[1].copy(),
                    }
                )
            contacts_by_primary.append(records)

        def tensor(value: Any, *, size: int | None = None) -> torch.Tensor:
            result = torch.as_tensor(value, dtype=self.bundle.dtype, device=self.bundle.device)
            if size is not None:
                result = result.reshape(size)
            return result

        fields: dict[str, torch.Tensor] = {}
        selected_records: list[list[dict[str, Any]]] = []
        if resolved.cfg.reduce == "none":
            selected_records = [
                records[: resolved.cfg.num_slots] for records in contacts_by_primary
            ]
            for records in selected_records:
                records.extend(
                    {
                        "found": 0.0,
                        "force": np.zeros(3),
                        "torque": np.zeros(3),
                        "dist": 0.0,
                        "pos": np.zeros(3),
                        "normal": np.zeros(3),
                        "tangent": np.zeros(3),
                    }
                    for _ in range(resolved.cfg.num_slots - len(records))
                )
        else:
            for records in contacts_by_primary:
                if not records:
                    selected_records.append([])
                    continue
                if resolved.cfg.reduce == "mindist":
                    selected_records.append([min(records, key=lambda item: item["dist"])])
                elif resolved.cfg.reduce == "maxforce":
                    selected_records.append(
                        [max(records, key=lambda item: float(np.linalg.norm(item["force"])))]
                    )
                else:
                    selected_records.append(records)

        def values_for(field_name: str, primary_index: int) -> Any:
            records = selected_records[primary_index]
            if resolved.cfg.reduce == "none":
                if field_name == "found":
                    return [record["found"] for record in records]
                return [record[field_name] for record in records]
            if field_name == "found":
                return 1.0 if records else 0.0
            if not records:
                dim = self._contact_field_dim(field_name)
                return np.zeros(dim if dim > 1 else 1)
            if resolved.cfg.reduce == "netforce" and field_name in {"force", "torque"}:
                return np.sum([record[field_name] for record in records], axis=0)
            if field_name == "dist":
                return min(record["dist"] for record in records)
            if resolved.cfg.reduce == "any" and field_name in {"force", "torque"}:
                return max(records, key=lambda item: float(np.linalg.norm(item["force"])))[
                    field_name
                ]
            return records[0][field_name]

        for field_name in resolved.cfg.fields:
            values = [values_for(field_name, index) for index in range(len(primary_groups))]
            if self._contact_field_dim(field_name) == 1:
                fields[field_name] = tensor(values).reshape(len(primary_groups), -1)
                if resolved.cfg.reduce != "none":
                    fields[field_name] = fields[field_name].squeeze(-1)
            else:
                fields[field_name] = tensor(values).reshape(
                    len(primary_groups),
                    resolved.cfg.num_slots if resolved.cfg.reduce == "none" else 1,
                    self._contact_field_dim(field_name),
                )
                if resolved.cfg.reduce != "none":
                    fields[field_name] = fields[field_name].squeeze(1)
        return ContactData(**fields)

    def _native_context(self, env: Any) -> tuple[Any, Any]:
        """Return native model/data matching the active scalar environment row."""

        physics = getattr(env, "physics", None)
        bundle = getattr(physics, "bundle", None) or getattr(env, "bundle", self.bundle)
        model = bundle.native_model
        key = id(model)
        native_data = self._native_data_by_model.get(key)
        if native_data is None:
            native_data = mujoco_api.MjData(model)
            self._native_data_by_model[key] = native_data
        return model, native_data

    def _read_resolved(self, env: Any, resolved: _ResolvedSensor) -> torch.Tensor:
        if getattr(env, "num_envs", 1) > 1:
            values: list[torch.Tensor] = []
            contact_rows: list[ContactData] = []
            ray_rows: list[RayCastData] = []
            for index in range(env.num_envs):
                row_resolved = copy.copy(resolved)
                value = self._read_resolved(_SensorRowEnv(env, index), row_resolved)
                values.append(value)
                if row_resolved.contact_data is not None:
                    contact_rows.append(row_resolved.contact_data)
                if row_resolved.ray_data is not None:
                    ray_rows.append(row_resolved.ray_data)
            if contact_rows:
                resolved.contact_data = self._stack_contact_data(contact_rows)
            if ray_rows:
                resolved.ray_data = self._stack_ray_data(ray_rows)
            return torch.stack(values, dim=0)
        data = env.data
        if data is None:
            raise RuntimeError("Call reset() before reading sensors")
        cfg = resolved.cfg
        kind = cfg.kind
        if kind == "mujoco":
            source = resolved.source_name or cfg.source or cfg.name
            sensor_slice = self.bundle.sensor_slices.get(source)
            if sensor_slice is None:
                return torch.zeros(
                    cfg.expected_dim or 0,
                    dtype=self.bundle.dtype,
                    device=self.bundle.device,
                )
            value = data.sensordata[..., sensor_slice]
        elif kind == "body_pose":
            value = torch.cat(
                [
                    torch.cat(
                        (data.xpos[..., body_id, :], data.xquat[..., body_id, :]),
                        dim=-1,
                    )
                    for body_id in (resolved.ids or (resolved.entity.root_body_id,))
                ],
                dim=-1,
            )
        elif kind == "body_velocity":
            body_ids = list(resolved.ids or (resolved.entity.root_body_id,))
            value = data.cvel[..., body_ids, :]
            if cfg.global_frame:
                rotation = data.xmat[..., body_ids, :].reshape(
                    *data.xmat[..., body_ids, :].shape[:-1], 3, 3
                )
                angular = torch.matmul(rotation, value[..., :3].unsqueeze(-1)).squeeze(-1)
                linear = torch.matmul(rotation, value[..., 3:].unsqueeze(-1)).squeeze(-1)
                value = torch.cat((angular, linear), dim=-1)
            value = value.reshape(-1)
        elif kind == "site_position":
            value = data.site_xpos[..., list(resolved.ids), :].reshape(-1)
        elif kind == "site_velocity":
            site_velocity = getattr(data, "site_xvelp", None)
            if site_velocity is None:
                raise RuntimeError("The active physics backend does not expose site velocities")
            site_ids = list(resolved.ids)
            value = site_velocity[..., site_ids, :]
            if not cfg.global_frame:
                rotation = data.site_xmat[..., site_ids, :].reshape(
                    *data.site_xmat[..., site_ids, :].shape[:-1], 3, 3
                )
                value = torch.matmul(rotation.transpose(-1, -2), value.unsqueeze(-1)).squeeze(-1)
            value = value.reshape(-1)
        elif kind == "joint_position":
            value = data.qpos.index_select(-1, resolved.qpos_indices)
        elif kind == "joint_velocity":
            value = data.qvel.index_select(-1, resolved.qvel_indices)
        elif kind in {"raycast", "terrain_height"}:
            ray_data = self._read_raycast(env, resolved)
            value = ray_data.heights if kind == "terrain_height" else ray_data.distances
        elif kind == "contact":
            contact_data = self._read_contact_from_sensors(env, resolved)
            if contact_data is not None:
                resolved.contact_data = contact_data
                value = self._flatten_contact(contact_data, cfg.fields)
            else:
                contact_data = self._read_contact_native(env, resolved)
                resolved.contact_data = contact_data
                value = self._flatten_contact(contact_data, cfg.fields)
        elif kind == "custom":
            reader = resolved.reader_instance or cfg.reader
            index = getattr(env, "index", None)
            if index is not None and resolved.reader_instances:
                reader = resolved.reader_instances[index]
            elif inspect.isclass(reader):
                if getattr(env, "num_envs", 1) > 1:
                    if index is None:
                        raise RuntimeError("Batched custom sensors require a row environment")
                    if index not in resolved.reader_instances:
                        resolved.reader_instances[index] = construct(reader, cfg, env)
                    reader = resolved.reader_instances[index]
                else:
                    if resolved.reader_instance is None:
                        resolved.reader_instance = construct(reader, cfg, env)
                    reader = resolved.reader_instance
            if reader is None:
                raise RuntimeError(f"Custom sensor {cfg.name!r} has no reader")
            value = reader.read(env) if hasattr(reader, "read") else reader(env, **cfg.params)
        else:  # pragma: no cover - guarded during resolution
            raise RuntimeError(f"Unsupported resolved sensor kind {kind!r}")
        value = torch.as_tensor(value, dtype=self.bundle.dtype, device=self.bundle.device)
        value = value.reshape(1) if value.ndim == 0 else value.reshape(-1)
        if resolved.handle.dimension and value.numel() != resolved.handle.dimension:
            raise RuntimeError(
                f"Sensor {cfg.name!r} produced {value.numel()} values; "
                f"expected {resolved.handle.dimension}"
            )
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Sensor {cfg.name!r} produced non-finite values")
        return value

    @staticmethod
    def _stack_contact_data(rows: list[ContactData]) -> ContactData:
        result = ContactData()
        for field_name in ContactData.__dataclass_fields__:
            values = [getattr(row, field_name) for row in rows]
            if all(value is not None for value in values):
                setattr(result, field_name, torch.stack(values, dim=0))
        return result

    @staticmethod
    def _stack_ray_data(rows: list[RayCastData]) -> RayCastData:
        return RayCastData(
            distances=torch.stack([row.distances for row in rows]),
            normals_w=torch.stack([row.normals_w for row in rows]),
            hit_pos_w=torch.stack([row.hit_pos_w for row in rows]),
            pos_w=torch.stack([row.pos_w for row in rows]),
            quat_w=torch.stack([row.quat_w for row in rows]),
            frame_pos_w=torch.stack([row.frame_pos_w for row in rows]),
            frame_quat_w=torch.stack([row.frame_quat_w for row in rows]),
            heights=(
                torch.stack([row.heights for row in rows]) if rows[0].heights is not None else None
            ),
        )

    def _contact_force(
        self, env: Any, resolved: _ResolvedSensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # mujoco-torch exposes body external forces but not mj_contactForce's
        # per-contact API.  Summing the primary entity's external-force norm is
        # the backend-independent equivalent available from the current data
        # contract and is zero when no matching contact is active.
        if not bool(mask.any()):
            return torch.zeros(
                (getattr(env, "num_envs", 1),) if getattr(env, "num_envs", 1) > 1 else (),
                dtype=self.bundle.dtype,
                device=self.bundle.device,
            )
        if resolved.entity is None:
            return torch.zeros(
                (getattr(env, "num_envs", 1),) if getattr(env, "num_envs", 1) > 1 else (),
                dtype=self.bundle.dtype,
                device=self.bundle.device,
            )
        force = env.data.cfrc_ext[..., list(resolved.entity.body_ids), 3:6]
        force = torch.linalg.vector_norm(force, dim=-1).sum(dim=-1)
        if getattr(env, "num_envs", 1) > 1:
            return torch.where(mask.any(dim=-1), force, torch.zeros_like(force))
        return (
            force if bool(mask.any()) else torch.zeros((), dtype=force.dtype, device=force.device)
        )

    def reset(self, env: Any, env_ids: torch.Tensor | slice | None = None) -> None:
        num_envs = getattr(env, "num_envs", 1)
        for resolved in self._resolved.values():
            reader = resolved.reader_instance
            if inspect.isclass(resolved.cfg.reader):
                if num_envs > 1:
                    if not resolved.reader_instances:
                        resolved.reader_instances = {
                            index: construct(
                                resolved.cfg.reader, resolved.cfg, _SensorRowEnv(env, index)
                            )
                            for index in range(num_envs)
                        }
                elif reader is None:
                    reader = construct(resolved.cfg.reader, resolved.cfg, env)
                    resolved.reader_instance = reader
            elif resolved.cfg.reader is not None:
                if num_envs > 1 and not resolved.reader_instances:
                    resolved.reader_instances = {
                        index: copy.deepcopy(resolved.cfg.reader) for index in range(num_envs)
                    }
                elif num_envs == 1 and resolved.reader_instance is None:
                    resolved.reader_instance = resolved.cfg.reader
                    reader = resolved.reader_instance
            self._sensors[resolved.cfg.name]._set_reader(reader)
        if not self._initialized:
            for name, sensor in self._sensors.items():
                resolved = self._resolved[name]
                if num_envs > 1 and resolved.reader_instances:
                    for index, reader in resolved.reader_instances.items():
                        row_env = _SensorRowEnv(env, index)
                        initialize = getattr(reader, "initialize", None)
                        if callable(initialize):
                            invoke_compatible(
                                initialize,
                                (
                                    ((row_env,), {}),
                                    (
                                        (
                                            row_env.bundle.native_model,
                                            row_env.bundle.torch_model,
                                            row_env.data,
                                            row_env.bundle.device,
                                        ),
                                        {},
                                    ),
                                    ((), {}),
                                ),
                            )
                else:
                    sensor.initialize(
                        self.bundle.native_model,
                        self.bundle.torch_model,
                        env.data,
                        self.bundle.device,
                    )
            self._initialized = True
        if env_ids is None or num_envs == 1:
            self._values.clear()
            self._history = {name: [] for name in self._resolved}
            self._air_time.clear()
            self._contact_timing.clear()
            self._contact_history.clear()
            self._ticks.clear()
        else:
            ids = self._ids(env_ids, env.num_envs)
            for _name, value in list(self._values.items()):
                value[ids] = 0
            for history in self._history.values():
                for value in history:
                    if value.ndim > 0 and value.shape[0] == num_envs:
                        value[ids] = 0
            for _name, value in self._air_time.items():
                value[ids] = 0
            for timing in self._contact_timing.values():
                for value in timing.values():
                    value[ids] = 0
            for history in self._contact_history.values():
                for contact in history:
                    for field_name in ContactData.__dataclass_fields__:
                        value = getattr(contact, field_name)
                        if (
                            isinstance(value, torch.Tensor)
                            and value.ndim > 0
                            and value.shape[0] == num_envs
                        ):
                            value[ids] = 0
        for resolved in self._resolved.values():
            resolved.contact_data = None
            resolved.ray_data = None
            if num_envs > 1:
                ticks = self._ticks.get(
                    resolved.cfg.name,
                    torch.zeros(num_envs, dtype=torch.long, device=self.bundle.device),
                )
                if env_ids is None:
                    ticks.zero_()
                else:
                    ticks[self._ids(env_ids, num_envs)] = 0
                self._ticks[resolved.cfg.name] = ticks
            else:
                self._ticks[resolved.cfg.name] = 0
        for name, sensor in self._sensors.items():
            resolved = self._resolved[name]
            if num_envs > 1 and resolved.reader_instances:
                ids = torch.arange(num_envs) if env_ids is None else self._ids(env_ids, num_envs)
                for index in ids.tolist():
                    reset = getattr(resolved.reader_instances[index], "reset", None)
                    if callable(reset):
                        invoke_compatible(
                            reset,
                            (((None,), {}), ((), {"env_ids": None}), ((), {})),
                        )
            else:
                sensor.reset(env_ids)

    @staticmethod
    def _ids(env_ids: torch.Tensor | slice, num_envs: int) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(num_envs)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)

    def update(self, env: Any) -> None:
        dt = self.bundle.timestep
        # Upstream invalidates/updates stateful sensors before their data is
        # consumed.  This matters for raycasts and custom temporal sensors:
        # their ``read`` observes the current substep, not the previous one.
        if getattr(env, "num_envs", 1) > 1:
            for resolved in self._resolved.values():
                for reader in resolved.reader_instances.values():
                    update = getattr(reader, "update", None)
                    if callable(update):
                        invoke_compatible(update, (((dt,), {}), ((), {})))
        else:
            for sensor in self._sensors.values():
                sensor.update(dt)
        for name, resolved in self._resolved.items():
            num_envs = getattr(env, "num_envs", 1)
            tick = self._ticks.get(name, 0)
            period = max(1, int(resolved.cfg.update_period))
            value = self._read_resolved(env, resolved)
            if num_envs > 1:
                if not isinstance(tick, torch.Tensor):
                    tick = torch.full((num_envs,), int(tick), dtype=torch.long, device=value.device)
                due = (tick % period == 0) | (name not in self._values)
                if name in self._values:
                    current = self._values[name].clone()
                    current[due] = value[due]
                    value = current
                self._ticks[name] = tick + 1
            else:
                if tick % period != 0 and name in self._values:
                    continue
                self._ticks[name] = int(tick) + 1
            self._values[name] = value
            history = self._history.setdefault(name, [])
            history.insert(0, value.clone())
            length = max(1, resolved.cfg.history_length + 1)
            del history[length:]
            if resolved.cfg.kind == "contact" and resolved.cfg.track_air_time:
                contact = resolved.contact_data
                found = (
                    contact.found.any()
                    if contact is not None and contact.found is not None
                    else self._contact_mask(env.data, resolved).any()
                )
                previous = self._air_time.get(name)
                if contact is not None and contact.found is not None:
                    contact_found = contact.found.any(dim=-1)
                    if env.num_envs == 1 and contact_found.ndim > 1:
                        contact_found = contact_found.squeeze(0)
                else:
                    contact_found = torch.full(
                        (
                            (env.num_envs, resolved.contact_primary_count)
                            if env.num_envs > 1
                            else (resolved.contact_primary_count,)
                        ),
                        bool(found),
                        dtype=torch.bool,
                        device=self.bundle.device,
                    )
                if previous is None:
                    previous = torch.zeros_like(contact_found, dtype=self.bundle.dtype)
                    self._air_time[name] = previous.clone()
                    self._contact_timing[name] = {
                        "air": previous.clone(),
                        "contact": previous.clone(),
                        "last_air": previous.clone(),
                        "last_contact": previous.clone(),
                    }
                timing = self._contact_timing[name]
                was_air = timing["air"] > 0
                was_contact = timing["contact"] > 0
                timing["last_air"] = torch.where(
                    was_air & contact_found,
                    timing["air"] + dt,
                    timing["last_air"],
                )
                timing["last_contact"] = torch.where(
                    was_contact & ~contact_found,
                    timing["contact"] + dt,
                    timing["last_contact"],
                )
                timing["air"] = torch.where(
                    contact_found,
                    torch.zeros_like(timing["air"]),
                    timing["air"] + dt,
                )
                timing["contact"] = torch.where(
                    contact_found,
                    timing["contact"] + dt,
                    torch.zeros_like(timing["contact"]),
                )
                self._air_time[name] = timing["air"]
                if contact is not None:
                    contact.current_air_time = timing["air"]
                    contact.last_air_time = timing["last_air"]
                    contact.current_contact_time = timing["contact"]
                    contact.last_contact_time = timing["last_contact"]
            contact = resolved.contact_data
            if contact is not None and resolved.cfg.history_length > 0:
                history_data = self._contact_history.setdefault(name, [])
                history_data.insert(0, contact)
                del history_data[resolved.cfg.history_length :]
                for field_name in ("force", "torque", "dist"):
                    if field_name in resolved.cfg.fields:
                        field_values = [
                            getattr(item, field_name)
                            for item in history_data
                            if getattr(item, field_name) is not None
                        ]
                        if field_values:
                            history_value = torch.stack(field_values, dim=2)
                            setattr(contact, f"{field_name}_history", history_value)

    def read(self, name: str, *, history: int = 0) -> torch.Tensor:
        if name not in self._values:
            raise RuntimeError(f"Sensor {name!r} has not been initialized; call reset() first")
        if history <= 0:
            return self._values[name]
        values = self._history[name]
        if not values:
            return self._values[name]
        requested = max(1, history)
        selected = values[:requested]
        if len(selected) < requested:
            zero = torch.zeros_like(values[0])
            selected = selected + [zero] * (requested - len(selected))
        return torch.cat(selected, dim=-1)

    def data(self, name: str) -> Any:
        """Return a typed sensor value; contacts expose ``ContactData``."""

        if name not in self._values:
            raise RuntimeError(f"Sensor {name!r} has not been initialized; call reset() first")
        resolved = self._resolved[name]
        if resolved.cfg.kind == "contact" and resolved.contact_data is not None:
            return resolved.contact_data
        if resolved.cfg.kind in {"raycast", "terrain_height"} and resolved.ray_data is not None:
            return resolved.ray_data
        return self._values[name]

    def __getitem__(self, name: str) -> Sensor:
        return self.get_sensor(name)

    def air_time(self, name: str) -> torch.Tensor:
        try:
            return self._air_time[name]
        except KeyError as exc:
            raise KeyError(f"Sensor {name!r} has no contact air-time state") from exc

    def _first_contact_state(self, name: str, dt: float, *, contact: bool) -> torch.Tensor:
        try:
            timing = self._contact_timing[name]
        except KeyError as exc:
            raise KeyError(f"Sensor {name!r} has no contact air-time state") from exc
        key = "contact" if contact else "air"
        current = timing[key]
        return (current > 0) & (current < dt + 1.0e-8)

    def compute_first_contact(self, name: str, dt: float) -> torch.Tensor:
        return self._first_contact_state(name, dt, contact=True)

    def compute_first_air(self, name: str, dt: float) -> torch.Tensor:
        return self._first_contact_state(name, dt, contact=False)

    def foot_contact_mask(self, bundle: ModelBundle, *, fallback: Any) -> torch.Tensor:
        """Return a two-foot mask through named contact sensors when present.

        The fallback keeps old configurations valid while task configs migrate
        their foot contacts to first-class sensor declarations.
        """

        names = ("left_foot_contact", "right_foot_contact")
        if not all(name in self._values for name in names):
            return fallback()
        del bundle  # retained in the public contract for task-side symmetry
        values = [self._values[name].reshape(-1) > 0 for name in names]
        if values[0].numel() > 1:
            return torch.stack(values, dim=-1)
        return torch.stack([value.squeeze() for value in values])


__all__ = ["ContactData", "Sensor", "SensorHandle", "SensorManager"]
