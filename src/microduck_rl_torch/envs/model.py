"""Load and fingerprint the Microduck MuJoCo model for torch execution."""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import mujoco_torch
import numpy as np
import torch

from ..robot.constants import SERVO_JOINT_NAMES
from .actuation import ActuatorDelayCfg, BamM6Parameters
from .scene import EntityCfg, SemanticSelector

mujoco_api: Any = mujoco

SENSOR_NAMES = ("imu_ang_vel", "imu_accel")


def _required_id(model: Any, obj_type: int, name: str) -> int:
    object_id = mujoco_api.mj_name2id(model, obj_type, name)
    if object_id < 0:
        raise ValueError(f"Required MuJoCo object {name!r} was not found")
    return int(object_id)


def _object_name(model: Any, obj_type: int, object_id: int) -> str:
    name = mujoco_api.mj_id2name(model, obj_type, object_id)
    return name or f"<unnamed:{object_id}>"


def _local_object_name(global_name: str, entity_name: str | None) -> str:
    """Return an entity-local name from an attached MuJoCo namespace."""

    prefix = f"{entity_name}/" if entity_name else ""
    return global_name.removeprefix(prefix)


def _scoped_name(
    model: Any,
    obj_type: int,
    entity_name: str | None,
    local_name: str,
) -> str:
    """Resolve an entity-local name, with attached-scene namespaces preferred."""

    candidates = (f"{entity_name}/{local_name}", local_name) if entity_name else (local_name,)
    for candidate in candidates:
        if mujoco_api.mj_name2id(model, obj_type, candidate) >= 0:
            return candidate
    raise ValueError(
        f"Required {obj_type!r} {local_name!r} was not found for entity {entity_name!r}"
    )


def _body_descendants(model: Any, body_id: int) -> set[int]:
    """Return a body and every descendant body in the MuJoCo tree."""

    descendants: set[int] = set()
    for candidate in range(int(model.nbody)):
        current = candidate
        while current >= 0:
            if current == body_id:
                descendants.add(candidate)
                break
            parent = int(model.body_parentid[current])
            if parent == current:
                break
            current = parent
    return descendants


def _discover_root_body_id(model: Any, entity_name: str) -> int:
    """Find an entity's direct world-body root when no semantic name is given."""

    prefix = f"{entity_name}/"
    candidates = [
        body_id
        for body_id in range(1, int(model.nbody))
        if int(model.body_parentid[body_id]) == 0
        and (
            _object_name(model, mujoco_api.mjtObj.mjOBJ_BODY, body_id).startswith(prefix)
            or not any(
                _object_name(model, mujoco_api.mjtObj.mjOBJ_BODY, candidate).startswith(prefix)
                for candidate in range(1, int(model.nbody))
            )
        )
    ]
    if not candidates:
        raise ValueError(
            f"Could not discover a root body for entity {entity_name!r}; "
            "set EntityCfg.root_body_name explicitly"
        )
    return candidates[0]


def _resolve_selector(
    model: Any,
    obj_type: int,
    selector: SemanticSelector,
    entity_name: str | None = None,
) -> tuple[int, ...]:
    """Resolve a semantic selector against the compiled native model."""

    def in_entity_namespace(object_id: int, object_type_: int) -> bool:
        """Accept local names in direct scenes and scoped names when attached."""

        if entity_name is None:
            return True
        object_name = _object_name(model, object_type_, object_id)
        # A direct one-entity XML intentionally keeps local names.  A
        # composed scene, by contrast, prefixes every entity object.  The
        # distinction is structural, not a special case for the robot task.
        count_ = {
            mujoco_api.mjtObj.mjOBJ_BODY: int(model.nbody),
            mujoco_api.mjtObj.mjOBJ_GEOM: int(model.ngeom),
            mujoco_api.mjtObj.mjOBJ_SITE: int(model.nsite),
        }.get(object_type_)
        if count_ is None:
            return object_name.startswith(f"{entity_name}/")
        has_namespace = any(
            _object_name(model, object_type_, candidate_id).startswith(f"{entity_name}/")
            for candidate_id in range(count_)
        )
        return not has_namespace or object_name.startswith(f"{entity_name}/")

    if selector.mode == "names":
        result = tuple(
            _required_id(model, obj_type, _scoped_name(model, obj_type, entity_name, name))
            for name in selector.names
        )
    elif selector.mode == "regex":
        pattern = re.compile(selector.pattern or "")
        # MuJoCo's Python API does not expose a common object-count accessor;
        # resolve the supported selector types explicitly below.
        count = {
            mujoco_api.mjtObj.mjOBJ_BODY: int(model.nbody),
            mujoco_api.mjtObj.mjOBJ_GEOM: int(model.ngeom),
            mujoco_api.mjtObj.mjOBJ_SITE: int(model.nsite),
        }.get(obj_type)
        if count is None:
            raise ValueError(f"Regex selectors do not support MuJoCo object type {obj_type}")
        result = tuple(
            object_id
            for object_id in range(count)
            if pattern.search(
                _local_object_name(_object_name(model, obj_type, object_id), entity_name)
            )
            and (in_entity_namespace(object_id, obj_type))
        )
        if not result:
            raise ValueError(f"Selector {selector!r} matched no MuJoCo objects")
    elif selector.mode == "body_subtree":
        if obj_type == mujoco_api.mjtObj.mjOBJ_BODY or obj_type == mujoco_api.mjtObj.mjOBJ_XBODY:
            pattern = re.compile(selector.pattern or "")
            result = tuple(
                body_id
                for body_id in range(int(model.nbody))
                if pattern.search(
                    _local_object_name(
                        _object_name(model, mujoco_api.mjtObj.mjOBJ_BODY, body_id), entity_name
                    )
                )
                and (in_entity_namespace(body_id, mujoco_api.mjtObj.mjOBJ_BODY))
            )
            if not result:
                raise ValueError(f"Body selector {selector!r} matched no MuJoCo bodies")
            return result
        if obj_type != mujoco_api.mjtObj.mjOBJ_GEOM:
            raise ValueError("body_subtree selectors resolve bodies or geometry contacts")
        pattern = re.compile(selector.pattern or "")
        body_ids = tuple(
            body_id
            for body_id in range(int(model.nbody))
            if pattern.search(
                _local_object_name(
                    _object_name(model, mujoco_api.mjtObj.mjOBJ_BODY, body_id), entity_name
                )
            )
            and in_entity_namespace(body_id, mujoco_api.mjtObj.mjOBJ_BODY)
        )
        if not body_ids:
            raise ValueError(f"Body selector {selector!r} matched no MuJoCo bodies")
        descendants = set().union(*(_body_descendants(model, body_id) for body_id in body_ids))
        result = tuple(
            geom_id
            for geom_id in range(int(model.ngeom))
            if int(model.geom_bodyid[geom_id]) in descendants
            and in_entity_namespace(geom_id, mujoco_api.mjtObj.mjOBJ_GEOM)
        )
        if not result:
            raise ValueError(f"Body selector {selector!r} matched no geometry")
    else:
        raise ValueError(f"Unsupported selector mode {selector.mode!r}")
    return result


@dataclass(frozen=True)
class CollisionCapabilityReport:
    """Authored/effective collision capability result for one compiled scene."""

    policy: str
    backend: str
    native_supported: tuple[str, ...] = ()
    approximated: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    effective_geom_types: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EntityView:
    """Resolved semantic view of one compiled scene entity."""

    name: str
    kind: str
    root_body_id: int
    body_ids: tuple[int, ...]
    geom_ids: tuple[int, ...]
    site_ids: tuple[int, ...]
    joint_ids: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    actuator_joint_ids: tuple[int, ...]
    tendon_ids: tuple[int, ...]
    camera_ids: tuple[int, ...]
    light_ids: tuple[int, ...]
    material_ids: tuple[int, ...]
    pair_ids: tuple[int, ...]
    non_free_joint_ids: tuple[int, ...]
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    geom_names: tuple[str, ...]
    site_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    tendon_names: tuple[str, ...]
    qpos_indices: torch.Tensor
    qvel_indices: torch.Tensor
    non_free_qpos_indices: torch.Tensor
    non_free_qvel_indices: torch.Tensor
    free_qpos_indices: torch.Tensor
    free_qvel_indices: torch.Tensor

    def _find(
        self, names: tuple[str, ...], query: str | list[str] | tuple[str, ...]
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Resolve exact/regex names in entity-local order."""

        patterns = (query,) if isinstance(query, str) else tuple(query)
        matches: list[int] = []
        matched_names: list[str] = []
        for index, name in enumerate(names):
            if any(name == pattern or re.search(pattern, name) is not None for pattern in patterns):
                matches.append(index)
                matched_names.append(name)
        if not matches:
            raise ValueError(f"No names matched {patterns!r} in entity {self.name!r}")
        return torch.tensor(matches, dtype=torch.long, device=self.qpos_indices.device), tuple(
            matched_names
        )

    def find_joints(
        self, query: str | list[str] | tuple[str, ...]
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Resolve non-free articulation joints like ``Entity.find_joints``."""

        names = tuple(
            name
            for joint_id, name in zip(self.joint_ids, self.joint_names, strict=True)
            if joint_id in self.non_free_joint_ids
        )
        ids, matched = self._find(names, query)
        # Entity data methods use entity-local articulation ordering. Global
        # MuJoCo addresses stay inside the resolved view and are applied only
        # at the write sink.
        return ids, matched

    def find_bodies(
        self, query: str | list[str] | tuple[str, ...]
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        ids, matched = self._find(self.body_names, query)
        return ids, matched

    def find_sites(
        self, query: str | list[str] | tuple[str, ...]
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        ids, matched = self._find(self.site_names, query)
        return ids, matched

    def find_joints_by_actuator_names(
        self, query: str | list[str] | tuple[str, ...]
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Resolve joints through actuator-local names."""

        ids, names = self._find(self.actuator_names, query)
        joint_ids: list[int] = []
        joint_names: list[str] = []
        local_joint_by_global = {
            joint_id: index for index, joint_id in enumerate(self.non_free_joint_ids)
        }
        for actuator_index, name in zip(ids.tolist(), names, strict=True):
            # Non-joint transmissions have no articulation joint to return.
            joint_id = self.actuator_joint_ids[actuator_index]
            local_joint_id = local_joint_by_global.get(joint_id)
            if local_joint_id is not None:
                joint_ids.append(local_joint_id)
                joint_names.append(name)
        if not joint_ids:
            raise ValueError(f"No actuated joints matched {query!r} in entity {self.name!r}")
        return torch.tensor(joint_ids, dtype=torch.long, device=self.qpos_indices.device), tuple(
            joint_names
        )


def _joint_qpos_width(joint_type: int) -> int:
    # MuJoCo joint enum values are stable: free, ball, slide, hinge.
    return (7, 4, 1, 1)[joint_type]


def _joint_qvel_width(joint_type: int) -> int:
    return (6, 3, 1, 1)[joint_type]


def _resolve_entity_view(
    model: Any,
    entity_cfg: EntityCfg,
    *,
    device: torch.device,
) -> EntityView:
    root_name = entity_cfg.root_body_name
    root_body_id = (
        _discover_root_body_id(model, entity_cfg.name)
        if root_name is None
        else _required_id(
            model,
            mujoco_api.mjtObj.mjOBJ_BODY,
            _scoped_name(model, mujoco_api.mjtObj.mjOBJ_BODY, entity_cfg.name, root_name),
        )
    )
    body_ids = tuple(sorted(_body_descendants(model, root_body_id)))
    body_id_set = set(body_ids)
    geom_ids = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in body_id_set
    )

    joint_ids = tuple(
        joint_id
        for joint_id in range(int(model.njnt))
        if int(model.jnt_bodyid[joint_id]) in body_id_set
    )
    site_ids = tuple(
        site_id
        for site_id in range(int(model.nsite))
        if int(model.site_bodyid[site_id]) in body_id_set
    )
    # Tendons/materials do not have a body address.  Entity attachment gives
    # them a namespace, so only prefixed names belong to an attached entity;
    # in a direct one-entity scene all names are local to that entity.
    has_entity_prefix = any(
        _object_name(model, mujoco_api.mjtObj.mjOBJ_BODY, body_id).startswith(f"{entity_cfg.name}/")
        for body_id in body_ids
    )
    tendon_ids = tuple(
        tendon_id
        for tendon_id in range(int(getattr(model, "ntendon", 0)))
        if (
            _object_name(model, mujoco_api.mjtObj.mjOBJ_TENDON, tendon_id).startswith(
                f"{entity_cfg.name}/"
            )
            if has_entity_prefix
            else True
        )
    )
    joint_id_set = set(joint_ids)
    tendon_id_set = set(tendon_ids)
    actuator_ids = tuple(
        actuator_id
        for actuator_id in range(int(model.nu))
        if (
            int(model.actuator_trntype[actuator_id]) == int(mujoco_api.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[actuator_id, 0]) in joint_id_set
        )
        or (
            int(model.actuator_trntype[actuator_id]) == int(mujoco_api.mjtTrn.mjTRN_TENDON)
            and int(model.actuator_trnid[actuator_id, 0]) in tendon_id_set
        )
    )
    camera_ids = tuple(
        camera_id
        for camera_id in range(int(getattr(model, "ncam", 0)))
        if (
            int(model.cam_bodyid[camera_id]) in body_id_set
            if hasattr(model, "cam_bodyid")
            else False
        )
    )
    light_ids = tuple(
        light_id
        for light_id in range(int(getattr(model, "nlight", 0)))
        if (
            int(model.light_bodyid[light_id]) in body_id_set
            if hasattr(model, "light_bodyid")
            else False
        )
    )
    material_ids = tuple(
        material_id
        for material_id in range(int(getattr(model, "nmat", 0)))
        if (
            _object_name(model, mujoco_api.mjtObj.mjOBJ_MATERIAL, material_id).startswith(
                f"{entity_cfg.name}/"
            )
            if has_entity_prefix
            else True
        )
    )
    pair_ids = tuple(
        pair_id
        for pair_id in range(int(getattr(model, "npair", 0)))
        if (
            int(model.pair_geom1[pair_id]) in set(geom_ids)
            or int(model.pair_geom2[pair_id]) in set(geom_ids)
        )
    )
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    non_free_joint_ids: list[int] = []
    non_free_qpos_indices: list[int] = []
    non_free_qvel_indices: list[int] = []
    free_qpos_indices: list[int] = []
    free_qvel_indices: list[int] = []
    for joint_id in joint_ids:
        joint_type = int(model.jnt_type[joint_id])
        qpos_start = int(model.jnt_qposadr[joint_id])
        qvel_start = int(model.jnt_dofadr[joint_id])
        qpos_values = list(range(qpos_start, qpos_start + _joint_qpos_width(joint_type)))
        qvel_values = list(range(qvel_start, qvel_start + _joint_qvel_width(joint_type)))
        qpos_indices.extend(qpos_values)
        qvel_indices.extend(qvel_values)
        if joint_type == 0:
            free_qpos_indices.extend(qpos_values)
            free_qvel_indices.extend(qvel_values)
        else:
            non_free_joint_ids.append(joint_id)
            non_free_qpos_indices.extend(qpos_values)
            non_free_qvel_indices.extend(qvel_values)
    return EntityView(
        name=entity_cfg.name,
        kind=entity_cfg.kind,
        root_body_id=root_body_id,
        body_ids=body_ids,
        geom_ids=geom_ids,
        site_ids=site_ids,
        joint_ids=joint_ids,
        actuator_ids=actuator_ids,
        actuator_joint_ids=tuple(
            int(model.actuator_trnid[actuator_id, 0])
            if int(model.actuator_trntype[actuator_id]) == int(mujoco_api.mjtTrn.mjTRN_JOINT)
            else -1
            for actuator_id in actuator_ids
        ),
        tendon_ids=tendon_ids,
        camera_ids=camera_ids,
        light_ids=light_ids,
        material_ids=material_ids,
        pair_ids=pair_ids,
        non_free_joint_ids=tuple(non_free_joint_ids),
        joint_names=tuple(
            _local_object_name(
                _object_name(model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_id), entity_cfg.name
            )
            for joint_id in joint_ids
        ),
        body_names=tuple(
            _local_object_name(
                _object_name(model, mujoco_api.mjtObj.mjOBJ_BODY, body_id), entity_cfg.name
            )
            for body_id in body_ids
        ),
        geom_names=tuple(
            _local_object_name(
                _object_name(model, mujoco_api.mjtObj.mjOBJ_GEOM, geom_id), entity_cfg.name
            )
            for geom_id in geom_ids
        ),
        site_names=tuple(
            _local_object_name(
                _object_name(model, mujoco_api.mjtObj.mjOBJ_SITE, site_id), entity_cfg.name
            )
            for site_id in site_ids
        ),
        actuator_names=tuple(
            _local_object_name(
                _object_name(model, mujoco_api.mjtObj.mjOBJ_ACTUATOR, actuator_id),
                entity_cfg.name,
            )
            for actuator_id in actuator_ids
        ),
        tendon_names=tuple(
            _local_object_name(
                _object_name(model, mujoco_api.mjtObj.mjOBJ_TENDON, tendon_id), entity_cfg.name
            )
            for tendon_id in tendon_ids
        ),
        qpos_indices=torch.tensor(qpos_indices, dtype=torch.long, device=device),
        qvel_indices=torch.tensor(qvel_indices, dtype=torch.long, device=device),
        non_free_qpos_indices=torch.tensor(non_free_qpos_indices, dtype=torch.long, device=device),
        non_free_qvel_indices=torch.tensor(non_free_qvel_indices, dtype=torch.long, device=device),
        free_qpos_indices=torch.tensor(free_qpos_indices, dtype=torch.long, device=device),
        free_qvel_indices=torch.tensor(free_qvel_indices, dtype=torch.long, device=device),
    )


def _overlay_entity_keyframes(
    native_model: Any,
    default_qpos: np.ndarray,
    default_qvel: np.ndarray,
    default_ctrl: np.ndarray,
    entities: Mapping[str, EntityCfg],
    views: Mapping[str, EntityView],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlay per-entity source keyframes onto composed state vectors.

    A composed XML intentionally has no global keyframe because free props
    change the qpos width. Store initialization on each entity and copy each
    configured entity keyframe by joint name into the compiled scene's qpos
    layout.
    """

    result_qpos = default_qpos.copy()
    result_qvel = default_qvel.copy()
    result_ctrl = default_ctrl.copy()
    for name, cfg in entities.items():
        if cfg.keyframe_name is None:
            continue
        source_path = (cfg.keyframe_source or cfg.xml_path).resolve()
        if not source_path.is_file():
            continue
        source_model = mujoco_api.MjModel.from_xml_path(str(source_path))
        source_key_id = mujoco_api.mj_name2id(
            source_model, mujoco_api.mjtObj.mjOBJ_KEY, cfg.keyframe_name
        )
        if source_key_id < 0:
            continue
        source_qpos = source_model.key_qpos[source_key_id]
        source_qvel = source_model.key_qvel[source_key_id]
        source_ctrl = source_model.key_ctrl[source_key_id]
        view = views[name]
        for joint_id in view.joint_ids:
            joint_name = _local_object_name(
                _object_name(native_model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_id), name
            )
            source_joint_id = mujoco_api.mj_name2id(
                source_model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_name
            )
            if source_joint_id < 0:
                continue
            target_start = int(native_model.jnt_qposadr[joint_id])
            source_start = int(source_model.jnt_qposadr[source_joint_id])
            width = _joint_qpos_width(int(native_model.jnt_type[joint_id]))
            result_qpos[target_start : target_start + width] = source_qpos[
                source_start : source_start + width
            ]
            target_v_start = int(native_model.jnt_dofadr[joint_id])
            source_v_start = int(source_model.jnt_dofadr[source_joint_id])
            v_width = _joint_qvel_width(int(native_model.jnt_type[joint_id]))
            result_qvel[target_v_start : target_v_start + v_width] = source_qvel[
                source_v_start : source_v_start + v_width
            ]
        for actuator_id in view.actuator_ids:
            actuator_name = _local_object_name(
                _object_name(native_model, mujoco_api.mjtObj.mjOBJ_ACTUATOR, actuator_id), name
            )
            source_actuator_id = mujoco_api.mj_name2id(
                source_model, mujoco_api.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if source_actuator_id >= 0 and source_actuator_id < len(source_ctrl):
                result_ctrl[actuator_id] = source_ctrl[source_actuator_id]
    return result_qpos, result_qvel, result_ctrl


def _apply_entity_initial_state(
    native_model: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    entities: Mapping[str, EntityCfg],
    views: Mapping[str, EntityView],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply configured entity spawn/init state to compiled model vectors."""

    result_qpos = qpos.copy()
    result_qvel = qvel.copy()
    for name, cfg in entities.items():
        init = cfg.init_state
        view = views[name]
        root_pos = cfg.spawn_pos or init.pos
        root_quat = cfg.spawn_quat or init.quat
        for joint_id in view.joint_ids:
            joint_type = int(native_model.jnt_type[joint_id])
            qpos_start = int(native_model.jnt_qposadr[joint_id])
            qvel_start = int(native_model.jnt_dofadr[joint_id])
            joint_name = _local_object_name(
                _object_name(native_model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_id), name
            )
            if joint_type == 0:  # mjJNT_FREE
                if root_pos is not None:
                    result_qpos[qpos_start : qpos_start + 3] = root_pos
                if root_quat is not None:
                    result_qpos[qpos_start + 3 : qpos_start + 7] = root_quat
                if init.linear_velocity is not None:
                    result_qvel[qvel_start : qvel_start + 3] = init.linear_velocity
                if init.angular_velocity is not None:
                    # The public EntityInitStateCfg accepts world-frame
                    # angular velocity, while MuJoCo stores free-joint
                    # angular velocity in the root body frame.  Keep the
                    # compiled default consistent with ResetManager and the
                    # entity data writer.
                    quaternion = result_qpos[qpos_start + 3 : qpos_start + 7]
                    inverse = quaternion.copy()
                    inverse[1:] *= -1.0
                    angular = np.asarray(init.angular_velocity, dtype=np.float64)
                    twice = 2.0 * np.cross(inverse[1:], angular)
                    body_angular = angular + inverse[0] * twice + np.cross(inverse[1:], twice)
                    result_qvel[qvel_start + 3 : qvel_start + 6] = body_angular
            # ``joint_pos``/``joint_vel`` are articulation maps.  A free joint
            # is initialized only through its root pose/velocity fields and
            # must never receive a regex assignment such as ``.*``.
            if joint_type != 0:
                for pattern, value in init.joint_pos.items():
                    if joint_name == pattern or re.search(pattern, joint_name) is not None:
                        width = _joint_qpos_width(joint_type)
                        result_qpos[qpos_start : qpos_start + width] = value
                        break
                for pattern, value in init.joint_vel.items():
                    if joint_name == pattern or re.search(pattern, joint_name) is not None:
                        width = _joint_qvel_width(joint_type)
                        result_qvel[qvel_start : qvel_start + width] = value
                        break
    return result_qpos, result_qvel


@dataclass(frozen=True)
class ModelBundle:
    """Native and device-resident representations of one compiled scene.

    The bundle is scene-generic: it owns compiled global indexing and named
    entity views. Microduck-specific selectors are optional metadata populated
    by the active entity configuration, not a separate physics backend.
    """

    xml_path: Path
    primary_entity_cfg: EntityCfg
    primary_entity_name: str
    native_model: Any
    torch_model: Any
    device: torch.device
    dtype: torch.dtype
    timestep: float
    decimation: int
    fixed_iterations: bool
    solver_iterations: int
    line_search_iterations: int
    nconmax: int
    contacts_enabled: bool
    disable_mesh_mesh_contacts: bool
    qpos_indices: torch.Tensor
    qvel_indices: torch.Tensor
    default_qpos: torch.Tensor
    default_qvel: torch.Tensor
    default_ctrl: torch.Tensor
    backlash_qpos_indices: torch.Tensor
    backlash_qvel_indices: torch.Tensor
    backlash_mask: torch.Tensor
    default_pose: torch.Tensor
    actuator_joint_names: tuple[str, ...]
    actuator_joint_mask: torch.Tensor
    sensor_slices: dict[str, slice]
    root_body_id: int
    task_handles: Mapping[str, Any]
    actuator_mode: str
    actuator_delay: ActuatorDelayCfg
    # The current primary bundle owns one actuator-delay buffer; independently
    # delayed groups in multi-entity scenes are not supported at this boundary.
    bam_parameters: BamM6Parameters | None
    friction_dof_count: int
    entities: Mapping[str, EntityView]
    entity_configs: Mapping[str, EntityCfg]
    observation_size: int = 0
    action_size: int = 0
    collision_report: CollisionCapabilityReport = CollisionCapabilityReport(
        policy="error", backend="mujoco-torch"
    )

    @property
    def has_backlash(self) -> bool:
        return bool(self.backlash_mask.any().item())

    def handle(self, name: str, default: Any = ()) -> Any:
        """Return an optional task handle without imposing robot semantics."""

        return self.task_handles.get(name, default)

    @property
    def head_body_ids(self) -> tuple[int, ...]:
        return tuple(self.handle("head_body_ids"))

    @property
    def foot_site_ids(self) -> tuple[int, ...]:
        return tuple(self.handle("foot_site_ids"))

    @property
    def foot_geom_ids(self) -> tuple[int, ...]:
        return tuple(self.handle("foot_geom_ids"))

    @property
    def foot_geom_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(group) for group in self.handle("foot_geom_groups"))

    @property
    def collision_geom_ids(self) -> tuple[int, ...]:
        return tuple(self.handle("collision_geom_ids"))

    def entity(self, name: str) -> EntityView:
        """Return a resolved semantic entity view by task-configured name."""

        try:
            return self.entities[name]
        except KeyError as exc:
            raise KeyError(f"Scene entity {name!r} is not present in the model bundle") from exc

    def new_data(self) -> Any:
        """Create a forward-computed standing state on the target device."""

        # ``mujoco_torch.make_data`` currently allocates floating-point data
        # using its package default (float64) on the host.  Move the complete
        # data tree to the model's requested device and dtype before replacing
        # the initial state; otherwise float32 models fail in contact/Jacobian
        # code and non-CPU models receive mixed-device state.
        # ``make_data`` still constructs several intermediate fields with its
        # package-wide float64 default. MPS rejects float64 tensors entirely,
        # even when they are immediately cast afterward, so allocate those
        # intermediates from a temporary CPU model and only then move the
        # complete data tree to the requested device/dtype.
        data_model = self.torch_model
        if self.device.type != "cpu":
            data_model = mujoco_torch.device_put(self.native_model, dtype=torch.float64)
        data = mujoco_torch.make_data(data_model).to(
            device=self.device,
            dtype=self.dtype,
        )
        qpos = torch.as_tensor(
            self.default_qpos.detach().cpu().numpy(),
            dtype=self.dtype,
            device=self.device,
        )
        qvel = self.default_qvel.to(device=self.device, dtype=self.dtype)
        ctrl = (
            torch.zeros_like(data.ctrl)
            if self.actuator_mode == "bam"
            else self.default_ctrl.clone()
        )
        return mujoco_torch.forward(
            self.torch_model,
            data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl),
            fixed_iterations=self.fixed_iterations,
        )

    def fingerprint(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.xml_path.read_bytes()).hexdigest()
        return {
            "xml_sha256": digest,
            "xml_path": str(self.xml_path),
            "nq": int(self.native_model.nq),
            "nv": int(self.native_model.nv),
            "nu": int(self.native_model.nu),
            "nbody": int(self.native_model.nbody),
            "ngeom": int(self.native_model.ngeom),
            "nsensor": int(self.native_model.nsensor),
            "timestep": self.timestep,
            "decimation": self.decimation,
            "fixed_iterations": self.fixed_iterations,
            "solver_iterations": self.solver_iterations,
            "line_search_iterations": self.line_search_iterations,
            "nconmax": self.nconmax,
            "contacts_enabled": self.contacts_enabled,
            "disable_mesh_mesh_contacts": self.disable_mesh_mesh_contacts,
            "actuator_joint_names": list(self.actuator_joint_names),
            "primary_entity_name": self.primary_entity_cfg.name,
            "entities": {
                name: {
                    "kind": entity.kind,
                    "root_body_id": entity.root_body_id,
                    "body_ids": list(entity.body_ids),
                    "geom_ids": list(entity.geom_ids),
                    "site_ids": list(entity.site_ids),
                    "joint_ids": list(entity.joint_ids),
                    "actuator_ids": list(entity.actuator_ids),
                    "joint_names": list(entity.joint_names),
                }
                for name, entity in self.entities.items()
            },
            "primary_entity_xml_path": str(self.primary_entity_cfg.xml_path),
            "keyframe_source": (
                str(self.primary_entity_cfg.keyframe_source)
                if self.primary_entity_cfg.keyframe_source is not None
                else None
            ),
            "keyframe_name": self.primary_entity_cfg.keyframe_name,
            "foot_geom_groups": [list(group) for group in self.foot_geom_groups],
            "actuator_mode": self.actuator_mode,
            "actuator_delay": {
                "delay_min_lag": self.actuator_delay.delay_min_lag,
                "delay_max_lag": self.actuator_delay.delay_max_lag,
                "delay_hold_prob": self.actuator_delay.delay_hold_prob,
                "delay_update_period": self.actuator_delay.delay_update_period,
                "delay_per_env_phase": self.actuator_delay.delay_per_env_phase,
            },
            "has_backlash": self.has_backlash,
            "collision_report": {
                "policy": self.collision_report.policy,
                "backend": self.collision_report.backend,
                "native_supported": list(self.collision_report.native_supported),
                "approximated": list(self.collision_report.approximated),
                "unsupported": list(self.collision_report.unsupported),
                "effective_geom_types": [
                    list(value) for value in self.collision_report.effective_geom_types
                ],
            },
            "head_body_ids": list(self.head_body_ids),
            "bam_parameters": (
                {
                    "model": "m6",
                    "motor": "xl330",
                    "kp_fw": self.bam_parameters.kp_fw,
                    "vin": self.bam_parameters.vin,
                    "vin_drop_gain": self.bam_parameters.vin_drop_gain,
                    "vin_min": self.bam_parameters.vin_min,
                }
                if self.bam_parameters is not None
                else None
            ),
        }


def _compile_bam_model(
    path: Path,
    *,
    parameters: BamM6Parameters,
    expected_actuator_names: tuple[str, ...] | None = None,
    collision_policy: str = "error",
    disable_mesh_mesh_contacts: bool = False,
    nconmax: int | None = None,
) -> Any:
    """Compile the XML as the motor/friction model used by BAM.

    BAM's controller supplies torque in ``data.ctrl``.  The XML position
    actuators are consequently converted to unit-gear torque motors, while
    MuJoCo's joint-friction fields are reserved for the per-step BAM budget.
    The initial non-zero friction values are intentional: ``mujoco-torch``
    specializes the constraint layout at ``device_put`` time and must see the
    DOF-friction rows before the first controller update.
    """

    spec = mujoco_api.MjSpec.from_file(str(path))
    # Apply the active full-collision settings. The source XML supplies the
    # collision geoms, while the task applies these per-geom solver
    # fields when it builds the robot entity. Omitting the priority changes
    # contact ordering when a sole is simultaneously touching several meshes.
    for geom in spec.geoms:
        local_name = str(geom.name).rsplit("/", 1)[-1]
        if local_name in {"left_foot_collision", "right_foot_collision"}:
            geom.condim = 3
            geom.priority = 1
            geom.friction = [1.0, 0.005, 0.0001]
    force_limit = parameters.force_limit(8.2)
    matched = 0
    for actuator in spec.actuators:
        target = actuator.target
        target_name = target if isinstance(target, str) else target.name
        if target_name.startswith("passive_"):
            continue
        if int(actuator.trntype) != int(mujoco_api.mjtTrn.mjTRN_JOINT):
            raise ValueError(
                "BAM requires joint transmission actuators; "
                f"actuator {actuator.name!r} targets {target_name!r} with "
                f"transmission type {int(actuator.trntype)}. Use actuator_mode='xml' "
                "for tendon/site actuators."
            )
        actuator.set_to_motor()
        actuator.forcerange = (-force_limit, force_limit)
        actuator.forcelimited = True
        actuator.ctrllimited = False
        actuator.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint = spec.joint(target_name)
        joint.damping = np.zeros((3, 1))
        joint.frictionloss = parameters.friction_base
        joint.armature = parameters.armature
        joint.solref_friction = (-5.0e4, -2.0e2)
        joint.solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)
        matched += 1
    if expected_actuator_names is not None and matched != len(expected_actuator_names):
        raise ValueError(f"Expected {len(expected_actuator_names)} BAM actuators, got {matched}")
    _convert_unsupported_collision_primitives(spec, policy=collision_policy)
    if disable_mesh_mesh_contacts:
        _disable_mesh_mesh_contact_candidates(spec)
    if nconmax is not None:
        spec.nconmax = nconmax
    return spec.compile()


def _collision_pair_support(geom_type: int, other_type: int) -> bool:
    """Ask the installed mujoco-torch driver about one collision pair."""

    try:
        from mujoco_torch._src.collision_driver import get_collision_fn
        from mujoco_torch._src.types import GeomType

        first, second = GeomType(geom_type), GeomType(other_type)
        return (
            get_collision_fn((first, second)) is not None
            or get_collision_fn((second, first)) is not None
        )
    except (ImportError, TypeError, ValueError, AttributeError):
        return False


def collision_capability_report(spec: Any, *, policy: str = "error") -> CollisionCapabilityReport:
    """Inspect authored collision geometry before any optional approximation."""

    if policy not in {"approximate", "error"}:
        raise ValueError("collision_policy must be 'approximate' or 'error'")
    import mujoco

    type_names = {
        int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
        int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    }
    named_geoms = {str(geom.name): geom for geom in spec.geoms if geom.name}
    active_geoms = list(
        geom for geom in spec.geoms if int(geom.contype) != 0 or int(geom.conaffinity) != 0
    )
    # Explicit pairs are collision edges even when the ordinary bit masks are
    # zero.  Include their endpoints in the capability graph so an authored
    # pair cannot be silently omitted from validation.
    for pair in spec.pairs:
        for name in (pair.geomname1, pair.geomname2):
            geom = named_geoms.get(str(name))
            if geom is not None and geom not in active_geoms:
                active_geoms.append(geom)
    active_types = tuple(sorted({int(geom.type) for geom in active_geoms}))
    unsupported_types: set[int] = set()
    unsupported_geometry_names: set[str] = set()
    # Capability is a property of the collision pairs the scene can actually
    # produce, not of a Cartesian product of every geometry instance. Keep one
    # aggregate collision mask per (body, type). A rough grid may contain tens
    # of thousands of boxes, but the capability graph then has only one terrain
    # record and a small number of articulated-body records.
    body_type_masks: dict[str, dict[int, tuple[int, int]]] = {}
    for geom_index, geom in enumerate(active_geoms):
        parent = getattr(geom, "parent", None)
        parent_name = getattr(parent, "name", None)
        body_name = str(parent_name) if parent_name else f"<unnamed-body-{geom_index}>"
        type_index = int(geom.type)
        masks = body_type_masks.setdefault(body_name, {})
        old_contype, old_conaffinity = masks.get(type_index, (0, 0))
        masks[type_index] = (
            old_contype | int(geom.contype),
            old_conaffinity | int(geom.conaffinity),
        )
    active_type_values = tuple(active_types)
    for type_index, first_type in enumerate(active_type_values):
        for second_type in active_type_values[type_index:]:
            compatible = False
            for first_body, first_by_type in body_type_masks.items():
                first_masks = first_by_type.get(first_type)
                if first_masks is None:
                    continue
                for second_body, second_by_type in body_type_masks.items():
                    if first_body == second_body:
                        continue
                    second_masks = second_by_type.get(second_type)
                    if second_masks is None:
                        continue
                    if (first_masks[0] & second_masks[1]) or (second_masks[0] & first_masks[1]):
                        compatible = True
                        break
                if compatible:
                    break
            # An explicit pair is an edge regardless of body filtering or bit
            # masks. Its endpoint types still need a backend collision kernel.
            for pair in spec.pairs:
                pair_one = named_geoms.get(str(pair.geomname1))
                pair_two = named_geoms.get(str(pair.geomname2))
                if (
                    pair_one is not None
                    and pair_two is not None
                    and {int(pair_one.type), int(pair_two.type)} == {first_type, second_type}
                ):
                    compatible = True
                    break
            if compatible and not _collision_pair_support(first_type, second_type):
                unsupported_types.update((first_type, second_type))
                # Record every active geometry for which this implementation
                # has an explicit approximation. Keeping every sibling name
                # matters when a generated terrain has many hfields.
                unsupported_geometry_names.update(
                    str(geom.name)
                    for geom in active_geoms
                    if int(geom.type)
                    in {
                        int(mujoco.mjtGeom.mjGEOM_HFIELD),
                        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
                    }
                    and int(geom.type) in {first_type, second_type}
                )
    supported = [
        type_names.get(geom_type, str(geom_type))
        for geom_type in active_types
        if geom_type not in unsupported_types
    ]
    unsupported = [
        type_names.get(geom_type, str(geom_type))
        for geom_type in active_types
        if geom_type in unsupported_types
    ]
    approximated = tuple(
        str(geom.name)
        for geom in spec.geoms
        if (int(geom.contype) != 0 or int(geom.conaffinity) != 0)
        and type_names.get(int(geom.type), str(geom.type)) in {"cylinder", "ellipsoid"}
        and policy == "approximate"
    )
    if policy == "approximate" and "hfield" in unsupported:
        approximated += tuple(
            str(geom.name)
            for geom in spec.geoms
            if (int(geom.contype) != 0 or int(geom.conaffinity) != 0)
            and type_names.get(int(geom.type), str(geom.type)) == "hfield"
            and str(geom.name) in unsupported_geometry_names
        )
    if policy == "error" and unsupported:
        raise RuntimeError(
            "Collision policy 'error' cannot execute unsupported active geometry types: "
            f"{', '.join(unsupported)}"
        )
    return CollisionCapabilityReport(
        policy=policy,
        backend="mujoco-torch",
        native_supported=tuple(supported),
        approximated=approximated,
        unsupported=tuple(unsupported),
    )


def _convert_unsupported_collision_primitives(
    spec: Any, *, policy: str = "error"
) -> CollisionCapabilityReport:
    """Apply only explicitly allowed approximations to unsupported geometry."""

    report = collision_capability_report(spec, policy=policy)
    import mujoco

    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            continue
        if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER and str(geom.name) in report.approximated:
            radius, half_height = float(geom.size[0]), float(geom.size[1])
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = [radius, radius, half_height]
        elif geom.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID and str(geom.name) in report.approximated:
            geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            geom.size = [max(float(value) for value in geom.size), 0.0, 0.0]
        elif geom.type == mujoco.mjtGeom.mjGEOM_HFIELD and str(geom.name) in report.approximated:
            if policy != "approximate":
                raise RuntimeError(
                    "Active heightfield collision is unsupported by the selected backend"
                )
            hfield = spec.hfield(geom.hfieldname)
            half_x, half_y, vertical_scale, base_height = (float(value) for value in hfield.size)
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = [half_x, half_y, max(vertical_scale, 1.0e-4)]
            geom.pos = [float(geom.pos[0]), float(geom.pos[1]), base_height]
            geom.hfieldname = ""
    return report


def _disable_mesh_mesh_contact_candidates(spec: Any) -> None:
    """Disable mesh-to-mesh candidates before ``mujoco_torch`` precomputation.

    Mesh geometry remains present for rendering and keeps contact with planes,
    boxes, and generated terrain.  MuJoCo's contact masks make the choice
    explicit by giving active meshes a one-way collision mask: a mesh can
    receive contacts from terrain, but two mesh masks do not overlap.  This is
    the same semantic choice exposed by ``skip_mesh_mesh_contacts`` in the
    torch backend, applied early enough to avoid constructing the expensive
    convex mesh-pair tables in the first place.
    """

    import mujoco

    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    mesh_names = {str(geom.name) for geom in spec.geoms if int(geom.type) == mesh_type}
    # Explicit pairs bypass the ordinary contype/conaffinity broad phase. Drop
    # mesh-mesh pairs from the authored spec as well, so native MuJoCo and the
    # Torch candidate cache receive the same safety contract.
    for pair in tuple(spec.pairs):
        if pair.geomname1 in mesh_names and pair.geomname2 in mesh_names:
            spec.delete(pair)
    for geom in spec.geoms:
        if int(geom.type) != mesh_type:
            continue
        if int(geom.contype) == 0 and int(geom.conaffinity) == 0:
            continue
        geom.contype = 1
        geom.conaffinity = 0


def _move_mesh_geometry(model: Any, *, device: torch.device, dtype: torch.dtype) -> Any:
    """Move optional convex-mesh tensors and precomputed metadata to a device.

    Dtype conversion belongs in ``mujoco_torch.device_put``. This device-only
    step is retained because optional tuple fields and nested precomputed
    metadata are not reliably moved by ``TensorClass.to`` on every backend.
    """

    def move_tensor(value: torch.Tensor) -> torch.Tensor:
        value_dtype = getattr(value, "dtype", None)
        if isinstance(value_dtype, torch.dtype) and value_dtype.is_floating_point:
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)

    def move_optional(values: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(None if value is None else move_tensor(value) for value in values)

    convex_fields = {
        "geom_convex_vert": move_optional(model.geom_convex_vert),
        "geom_convex_facenormal": move_optional(model.geom_convex_facenormal),
        "mesh_convex": move_optional(model.mesh_convex),
    }
    # These fields are tuples of optional tensors. TensorClass's public
    # replacement path normalizes tuple contents back to the host, so update
    # the underlying field map to preserve accelerator devices.
    model._tensordict._tensordict.update(convex_fields)

    def move_auxiliary(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return move_tensor(value)
        if isinstance(value, dict):
            return {key: move_auxiliary(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(move_auxiliary(item) for item in value)
        if isinstance(value, list):
            return [move_auxiliary(item) for item in value]
        return value

    precomputed = getattr(model, "_device_precomp", None)
    if isinstance(precomputed, dict):
        moved_precomputed = move_auxiliary(precomputed)
        precomputed.clear()
        precomputed.update(moved_precomputed)
    return model


def _sensor_slices(model: Any) -> dict[str, slice]:
    """Return the exact compiled sensor names.

    ``MjSpec.attach`` namespaces entity-owned sensors (for example,
    ``robot/imu_ang_vel``). Consumers resolve logical task sensor names to
    this canonical scoped name at their entity boundary; the model bundle does
    not maintain a second unscoped compatibility namespace.
    """

    result: dict[str, slice] = {}
    for sensor_id in range(int(model.nsensor)):
        name = mujoco_api.mj_id2name(model, mujoco_api.mjtObj.mjOBJ_SENSOR, sensor_id)
        if name is None:
            continue
        sensor_slice = slice(
            int(model.sensor_adr[sensor_id]),
            int(model.sensor_adr[sensor_id]) + int(model.sensor_dim[sensor_id]),
        )
        result[name] = sensor_slice
    return result


def load_model_bundle(
    xml_path: Path | None = None,
    *,
    entity_cfg: EntityCfg | None = None,
    entities: Mapping[str, EntityCfg] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    timestep: float = 0.005,
    decimation: int = 4,
    fixed_iterations: bool = False,
    solver_iterations: int | None = None,
    line_search_iterations: int | None = None,
    disable_contacts: bool = False,
    disable_mesh_mesh_contacts: bool = False,
    actuator_mode: str = "bam",
    bam_parameters: BamM6Parameters | None = None,
    collision_policy: str = "error",
    nconmax: int | None = None,
) -> ModelBundle:
    """Load a generic composed MuJoCo scene into a device model bundle.

    ``actuator_mode="bam"`` is the policy actuator path. The
        ``xml`` mode remains available as a diagnostic baseline for isolating
    actuator effects.
    """

    if entity_cfg is None:
        raise TypeError("load_model_bundle requires an explicit primary entity_cfg")
    raw_scene_entities = dict(entities or {entity_cfg.name: entity_cfg})
    # The SceneCfg mapping key is the authoritative namespace, matching
    # mjlab's ``scene.entities`` contract.  EntityCfg.name is only the local
    # asset identity and may be reused when the same asset is instantiated
    # more than once.
    scene_entities = {
        name: (cfg if cfg.name == name else replace(cfg, name=name))
        for name, cfg in raw_scene_entities.items()
    }
    if entity_cfg.name not in scene_entities:
        # When the same asset is instantiated under aliases (for example
        # ``robot_a`` and ``robot_b``), the mapping key is the scene identity.
        # Resolve a positional ``entity_cfg`` to its matching first instance
        # instead of silently adding a third unscoped entity.
        matching_name = next(
            (
                name
                for name, cfg in scene_entities.items()
                if cfg.xml_path.resolve() == entity_cfg.xml_path.resolve()
            ),
            None,
        )
        if matching_name is not None:
            entity_cfg = scene_entities[matching_name]
        else:
            scene_entities[entity_cfg.name] = entity_cfg
    entity_cfg = scene_entities.get(entity_cfg.name, entity_cfg)
    if xml_path is None and any(
        cfg.transforms or cfg.spec_factory is not None or cfg.spec_fn is not None
        for cfg in scene_entities.values()
    ):
        # Direct bundle callers receive the same transform pipeline as
        # ManagerBasedTaskEnv.  Without this seam, a procedural backlash or
        # collision transform would work only when routed through SceneBuilder
        # and silently disappear from the direct loader API.
        from .scene import SceneBuilder, SceneCfg, TerrainCfg

        direct_scene = SceneCfg(
            entities=dict(scene_entities),
            scene_xml=entity_cfg.keyframe_source,
            terrain=TerrainCfg(kind="plane"),
        )
        path = SceneBuilder().build(direct_scene).xml_path.resolve()
    else:
        path = (xml_path or entity_cfg.xml_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if actuator_mode not in {"bam", "xml"}:
        raise ValueError("actuator_mode must be 'bam' or 'xml'")
    if nconmax is not None and nconmax < 1:
        raise ValueError("nconmax must be positive when configured")
    authored_spec = mujoco_api.MjSpec.from_file(str(path))
    collision_report = collision_capability_report(authored_spec, policy=collision_policy)
    bam_config = bam_parameters or BamM6Parameters()
    configured_actuator_names = tuple(
        name for cfg in scene_entities.values() for name in cfg.actuator_joint_names
    )
    expected_actuator_names = configured_actuator_names or entity_cfg.actuator_joint_names
    if not expected_actuator_names and len(scene_entities) == 1 and entity_cfg.kind == "robot":
        expected_actuator_names = SERVO_JOINT_NAMES
    if actuator_mode == "bam":
        native_model = _compile_bam_model(
            path,
            parameters=bam_config,
            expected_actuator_names=expected_actuator_names or None,
            collision_policy=collision_policy,
            disable_mesh_mesh_contacts=disable_mesh_mesh_contacts,
            nconmax=nconmax,
        )
    else:
        xml_spec = mujoco_api.MjSpec.from_file(str(path))
        collision_report = _convert_unsupported_collision_primitives(
            xml_spec, policy=collision_policy
        )
        if disable_mesh_mesh_contacts:
            _disable_mesh_mesh_contact_candidates(xml_spec)
        if nconmax is not None:
            xml_spec.nconmax = nconmax
        native_model = xml_spec.compile()
    geom_type_names = {
        int(mujoco_api.mjtGeom.mjGEOM_PLANE): "plane",
        int(mujoco_api.mjtGeom.mjGEOM_HFIELD): "hfield",
        int(mujoco_api.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco_api.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco_api.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco_api.mjtGeom.mjGEOM_MESH): "mesh",
        int(mujoco_api.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco_api.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    }
    authored_geom_types = {
        str(geom.name): geom_type_names.get(int(geom.type), str(geom.type))
        for geom in authored_spec.geoms
        if int(geom.contype) != 0 or int(geom.conaffinity) != 0
    }
    collision_report = replace(
        collision_report,
        effective_geom_types=tuple(
            (
                str(
                    mujoco_api.mj_id2name(native_model, mujoco_api.mjtObj.mjOBJ_GEOM, geom_id)
                    or geom_id
                ),
                authored_geom_types.get(
                    str(
                        mujoco_api.mj_id2name(native_model, mujoco_api.mjtObj.mjOBJ_GEOM, geom_id)
                        or geom_id
                    ),
                    geom_type_names.get(
                        int(native_model.geom_type[geom_id]), str(native_model.geom_type[geom_id])
                    ),
                ),
            )
            for geom_id in range(int(native_model.ngeom))
            if int(native_model.geom_contype[geom_id]) != 0
            or int(native_model.geom_conaffinity[geom_id]) != 0
        ),
    )
    native_model.opt.timestep = timestep
    if solver_iterations is not None:
        if solver_iterations < 1:
            raise ValueError("solver_iterations must be positive")
        native_model.opt.iterations = solver_iterations
    if line_search_iterations is not None:
        if line_search_iterations < 1:
            raise ValueError("line_search_iterations must be positive")
        native_model.opt.ls_iterations = line_search_iterations
    if disable_contacts:
        native_model.opt.disableflags |= int(mujoco_api.mjtDisableBit.mjDSBL_CONTACT)
    device_obj = torch.device(device)
    if expected_actuator_names and native_model.nu != len(expected_actuator_names):
        raise ValueError(
            f"Expected {len(expected_actuator_names)} actuators, got {native_model.nu}"
        )

    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    actuator_joint_names: list[str] = []
    joint_actuator_mask: list[bool] = []
    for actuator_id in range(native_model.nu):
        joint_id = int(native_model.actuator_trnid[actuator_id, 0])
        is_joint = (
            int(native_model.actuator_trntype[actuator_id]) == int(mujoco_api.mjtTrn.mjTRN_JOINT)
            and joint_id >= 0
        )
        joint_actuator_mask.append(is_joint)
        if is_joint:
            qpos_indices.append(int(native_model.jnt_qposadr[joint_id]))
            qvel_indices.append(int(native_model.jnt_dofadr[joint_id]))
            actuator_joint_names.append(
                mujoco_api.mj_id2name(native_model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_id)
            )
        else:
            qpos_indices.append(-1)
            qvel_indices.append(-1)
            trn_name = mujoco_api.mj_id2name(
                native_model,
                mujoco_api.mjtObj.mjOBJ_TENDON
                if int(native_model.actuator_trntype[actuator_id])
                == int(mujoco_api.mjtTrn.mjTRN_TENDON)
                else mujoco_api.mjtObj.mjOBJ_SITE,
                joint_id,
            )
            actuator_joint_names.append(trn_name or f"actuator_{actuator_id}")

    def actuator_local_name(global_name: str) -> str:
        for entity_name in scene_entities:
            prefix = f"{entity_name}/"
            if global_name.startswith(prefix):
                return global_name.removeprefix(prefix)
        return global_name

    actual_local_actuators = tuple(actuator_local_name(name) for name in actuator_joint_names)
    if expected_actuator_names and actual_local_actuators != expected_actuator_names:
        raise ValueError(
            "Unexpected actuator order: "
            f"{tuple(actuator_joint_names)!r}; expected {expected_actuator_names!r}"
        )

    # Backlash variants add one passive hinge after each servo.  The actuator
    # still drives the servo-side DOF, while the real encoder and firmware
    # position loop read the output-side angle (servo + passive play).  Keep
    # zero-safe addresses and a mask so the same code handles both XML forms.
    backlash_qpos_indices: list[int] = []
    backlash_qvel_indices: list[int] = []
    backlash_mask: list[float] = []
    for servo_name, servo_qpos, servo_qvel in zip(
        actuator_joint_names, qpos_indices, qvel_indices, strict=True
    ):
        backlash_name = f"passive_{servo_name}_backlash"
        backlash_id = mujoco_api.mj_name2id(
            native_model, mujoco_api.mjtObj.mjOBJ_JOINT, backlash_name
        )
        if backlash_id < 0:
            backlash_qpos_indices.append(servo_qpos)
            backlash_qvel_indices.append(servo_qvel)
            backlash_mask.append(0.0)
        else:
            backlash_qpos_indices.append(int(native_model.jnt_qposadr[backlash_id]))
            backlash_qvel_indices.append(int(native_model.jnt_dofadr[backlash_id]))
            backlash_mask.append(1.0)

    entity_views = {
        name: _resolve_entity_view(native_model, cfg, device=device_obj)
        for name, cfg in scene_entities.items()
    }
    actuated_entities = tuple(name for name, view in entity_views.items() if view.actuator_ids)
    mode_mismatches = tuple(
        name
        for name in actuated_entities
        if name != entity_cfg.name and scene_entities[name].actuator_mode != actuator_mode
    )
    if mode_mismatches:
        raise ValueError(
            "Actuated entities must use the task's shared actuator_mode; "
            f"requested {actuator_mode!r}, mismatched entities={mode_mismatches!r}"
        )
    delay_mismatches = tuple(
        name
        for name in actuated_entities
        if scene_entities[name].actuator_delay != entity_cfg.actuator_delay
    )
    if delay_mismatches:
        raise ValueError(
            "Actuated entities currently share one physics-step actuator delay; "
            f"mismatched entities={delay_mismatches!r}. Configure one common delay "
            "or split the scene into separate environments."
        )

    default_qvel = np.zeros(int(native_model.nv), dtype=np.float64)
    default_ctrl = np.zeros(int(native_model.nu), dtype=np.float64)
    if entity_cfg.keyframe_name is not None:
        key_id = mujoco_api.mj_name2id(
            native_model, mujoco_api.mjtObj.mjOBJ_KEY, entity_cfg.keyframe_name
        )
        if key_id < 0:
            # Scene composition deliberately drops global keyframes because
            # each entity owns its initialization. Prefer the entity's
            # separate keyframe source when it has the requested named frame,
            # then fall back to the compiled model's qpos0. A missing STAND
            # must not make a valid root XML unusable (free-body props commonly
            # have no such frame).
            fallback_path = entity_cfg.keyframe_source
            fallback_key_id = -1
            fallback_model = None
            if (
                len(scene_entities) == 1
                and fallback_path is not None
                and fallback_path.resolve() != path
            ):
                fallback_model = mujoco_api.MjModel.from_xml_path(str(fallback_path.resolve()))
                fallback_key_id = mujoco_api.mj_name2id(
                    fallback_model, mujoco_api.mjtObj.mjOBJ_KEY, entity_cfg.keyframe_name
                )
            if len(scene_entities) > 1:
                # A composed scene has a wider state vector than any one
                # entity. Start from its qpos0 and overlay each entity's
                # width-matching keyframe below by local joint name.
                default_qpos = np.asarray(native_model.qpos0, dtype=np.float64).copy()
            elif fallback_model is not None and fallback_key_id >= 0:
                default_qpos = np.asarray(
                    fallback_model.key_qpos[fallback_key_id], dtype=np.float64
                ).copy()
                default_qvel = np.asarray(
                    fallback_model.key_qvel[fallback_key_id], dtype=np.float64
                ).copy()
                default_ctrl = np.asarray(
                    fallback_model.key_ctrl[fallback_key_id], dtype=np.float64
                ).copy()
            else:
                default_qpos = np.asarray(native_model.qpos0, dtype=np.float64).copy()
        else:
            default_qpos = np.asarray(native_model.key_qpos[key_id], dtype=np.float64)
            default_qvel = np.asarray(native_model.key_qvel[key_id], dtype=np.float64)
            default_ctrl = np.asarray(native_model.key_ctrl[key_id], dtype=np.float64)
    else:
        # A free-body prop may have no named keyframe. MuJoCo's qpos0 is the
        # correct neutral fallback.
        default_qpos = np.asarray(native_model.qpos0, dtype=np.float64).copy()
    if len(scene_entities) > 1:
        default_qpos, default_qvel, default_ctrl = _overlay_entity_keyframes(
            native_model,
            default_qpos,
            default_qvel,
            default_ctrl,
            scene_entities,
            entity_views,
        )
    default_qpos, default_qvel = _apply_entity_initial_state(
        native_model, default_qpos, default_qvel, scene_entities, entity_views
    )
    model = mujoco_torch.device_put(native_model, dtype=dtype).to(device_obj)
    model = _move_mesh_geometry(model, device=device_obj, dtype=dtype)
    if disable_mesh_mesh_contacts:
        model._device_precomp["skip_mesh_mesh_contacts"] = True
    primary_view = entity_views[entity_cfg.name]
    root_body_id = primary_view.root_body_id
    head_body_ids_list: list[int] = []
    for body_name in entity_cfg.head_body_names:
        for candidate in (body_name, f"{entity_cfg.name}/{body_name}"):
            body_id = mujoco_api.mj_name2id(native_model, mujoco_api.mjtObj.mjOBJ_BODY, candidate)
            if body_id >= 0:
                head_body_ids_list.append(int(body_id))
                break
    head_body_ids = tuple(head_body_ids_list)
    if entity_cfg.foot_site_selector is None:
        foot_site_ids_resolved: tuple[int, ...] = ()
        foot_geom_groups: tuple[tuple[int, ...], ...] = ()
    else:
        foot_site_ids_resolved = _resolve_selector(
            native_model,
            mujoco_api.mjtObj.mjOBJ_SITE,
            entity_cfg.foot_site_selector,
            entity_cfg.name,
        )
        # Entity resolution is generic. A task may require two feet, while a
        # wheel assembly or helper entity may expose a different site count.
        if entity_cfg.foot_contact_selectors is None:
            raise ValueError("Foot contact selectors are required when foot sites are configured")
        foot_geom_groups = tuple(
            _resolve_selector(native_model, mujoco_api.mjtObj.mjOBJ_GEOM, selector, entity_cfg.name)
            for selector in entity_cfg.foot_contact_selectors
        )
        if any(not group for group in foot_geom_groups):
            raise ValueError("Each foot contact selector must resolve at least one geometry")
    return ModelBundle(
        xml_path=path,
        primary_entity_cfg=entity_cfg,
        primary_entity_name=entity_cfg.name,
        native_model=native_model,
        torch_model=model,
        device=device_obj,
        dtype=dtype,
        timestep=timestep,
        decimation=decimation,
        fixed_iterations=fixed_iterations,
        solver_iterations=int(native_model.opt.iterations),
        line_search_iterations=int(native_model.opt.ls_iterations),
        nconmax=int(native_model.nconmax),
        contacts_enabled=not disable_contacts,
        disable_mesh_mesh_contacts=bool(disable_mesh_mesh_contacts),
        qpos_indices=torch.tensor(qpos_indices, dtype=torch.long, device=device_obj),
        qvel_indices=torch.tensor(qvel_indices, dtype=torch.long, device=device_obj),
        default_qpos=torch.tensor(default_qpos, dtype=dtype, device=device_obj),
        default_qvel=torch.tensor(default_qvel, dtype=dtype, device=device_obj),
        default_ctrl=torch.tensor(default_ctrl, dtype=dtype, device=device_obj),
        backlash_qpos_indices=torch.tensor(
            backlash_qpos_indices, dtype=torch.long, device=device_obj
        ),
        backlash_qvel_indices=torch.tensor(
            backlash_qvel_indices, dtype=torch.long, device=device_obj
        ),
        backlash_mask=torch.tensor(backlash_mask, dtype=dtype, device=device_obj),
        default_pose=torch.tensor(
            [default_qpos[index] if index >= 0 else 0.0 for index in qpos_indices],
            dtype=dtype,
            device=device_obj,
        ),
        actuator_joint_names=tuple(actuator_joint_names),
        actuator_joint_mask=torch.tensor(joint_actuator_mask, dtype=torch.bool, device=device_obj),
        sensor_slices=_sensor_slices(native_model),
        root_body_id=root_body_id,
        task_handles={
            "head_body_ids": head_body_ids,
            "foot_site_ids": tuple(int(value) for value in foot_site_ids_resolved),
            "foot_geom_ids": tuple(int(group[0]) for group in foot_geom_groups if group),
            "foot_geom_groups": tuple(
                tuple(int(value) for value in group) for group in foot_geom_groups
            ),
            "collision_geom_ids": tuple(
                geom_id
                for geom_id in primary_view.geom_ids
                if (
                    (
                        name := mujoco_api.mj_id2name(
                            native_model, mujoco_api.mjtObj.mjOBJ_GEOM, geom_id
                        )
                    )
                    is not None
                    and (
                        (
                            entity_cfg.collision_name_suffix is None
                            and (
                                int(native_model.geom_contype[geom_id]) != 0
                                or int(native_model.geom_conaffinity[geom_id]) != 0
                            )
                        )
                        or (
                            entity_cfg.collision_name_suffix is not None
                            and name.endswith(entity_cfg.collision_name_suffix)
                        )
                    )
                )
            ),
        },
        actuator_mode=actuator_mode,
        actuator_delay=entity_cfg.actuator_delay,
        bam_parameters=bam_config if actuator_mode == "bam" else None,
        friction_dof_count=(
            int(np.count_nonzero(native_model.dof_frictionloss > 0))
            if actuator_mode == "bam"
            else 0
        ),
        entities=entity_views,
        entity_configs=scene_entities,
        action_size=int(native_model.nu),
        collision_report=collision_report,
    )


def clone_model_bundle(bundle: ModelBundle) -> ModelBundle:
    """Clone mutable model handles for one independent environment instance.

    ``mujoco-torch`` model metadata is unbatched. The correctness-first
    vector backend therefore owns one model per environment; this prevents
    model randomization for one environment from leaking into its siblings.
    """

    native_model = copy.deepcopy(bundle.native_model)
    torch_model = mujoco_torch.device_put(native_model, dtype=bundle.dtype).to(bundle.device)
    torch_model = _move_mesh_geometry(torch_model, device=bundle.device, dtype=bundle.dtype)
    if "skip_mesh_mesh_contacts" in getattr(bundle.torch_model, "_device_precomp", {}):
        torch_model._device_precomp["skip_mesh_mesh_contacts"] = True
    return replace(bundle, native_model=native_model, torch_model=torch_model)
