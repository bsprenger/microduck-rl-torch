"""Declarative, entity-scoped model mutation manager."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from ..model import _joint_qvel_width
from ..physics import SAFE_RUNTIME_MODEL_FIELDS
from ..scene import SemanticSelector
from ..task_config import ModelMutationTermCfg, MutationDistributionCfg, TermCollection


@dataclass(frozen=True)
class MutationRecord:
    """Inspectible description of one applied mutation term."""

    name: str
    field: str
    entity: str | None
    selected_indices: tuple[int, ...]
    operation: str
    distribution: str
    env_id: int | None = None
    sampled_value: tuple[float, ...] = ()


@dataclass
class ModelMutationManager:
    """Restore nominal model fields and apply ordered reset mutations.

    Runtime model mutations are restricted to fields captured by the physics
    backend.  Topology-changing edits belong to entity transforms and are
    rejected here rather than invalidating compiled collision/action caches.
    """

    terms: TermCollection = field(default_factory=TermCollection)
    records: list[MutationRecord] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.last_records: tuple[MutationRecord, ...] = ()
        self._records_by_env: dict[int, tuple[MutationRecord, ...]] = {}

    @staticmethod
    def _backends(
        env: Any, env_ids: torch.Tensor | slice | None
    ) -> list[tuple[Any, Any, int | None]]:
        if hasattr(env.physics, "instances"):
            ids = (
                torch.arange(env.num_envs, device=env.bundle.device)
                if env_ids is None
                else env._ids(env_ids)
            )
            return [
                (
                    env.physics.instances[int(index)],
                    env.physics.instances[int(index)].bundle,
                    int(index),
                )
                for index in ids.tolist()
            ]
        return [(env.physics, env.bundle, 0)]

    @staticmethod
    def _object_kind(field_name: str) -> str:
        if field_name.startswith("body_"):
            return "body"
        if field_name.startswith("geom_"):
            return "geom"
        if field_name.startswith("dof_"):
            return "dof"
        if field_name.startswith("jnt_"):
            return "joint"
        if field_name.startswith("actuator_"):
            return "actuator"
        if field_name.startswith("tendon_"):
            return "tendon"
        return "global"

    @staticmethod
    def _all_ids(view: Any, kind: str) -> tuple[int, ...]:
        if kind == "body":
            return view.body_ids
        if kind == "geom":
            return view.geom_ids
        if kind == "dof":
            return tuple(int(value) for value in view.non_free_qvel_indices.tolist())
        if kind == "joint":
            return view.non_free_joint_ids
        if kind == "actuator":
            return view.actuator_ids
        if kind == "tendon":
            return view.tendon_ids
        return ()

    @staticmethod
    def _names(view: Any, kind: str, bundle: Any) -> tuple[str, ...]:
        if kind == "body":
            return view.body_names
        if kind == "geom":
            return tuple(view.geom_names)
        if kind == "joint":
            return tuple(
                name
                for joint_id, name in zip(view.joint_ids, view.joint_names, strict=True)
                if joint_id in view.non_free_joint_ids
            )
        if kind == "actuator":
            return view.actuator_names
        if kind == "tendon":
            return tuple(view.tendon_names)
        if kind == "dof":
            names: list[str] = []
            joint_names = ModelMutationManager._names(view, "joint", bundle)
            for joint_id, name in zip(view.non_free_joint_ids, joint_names, strict=True):
                width = _joint_qvel_width(int(bundle.native_model.jnt_type[joint_id]))
                names.extend([name] * width)
            return tuple(names)
        return ()

    @classmethod
    def _select(cls, bundle: Any, term: ModelMutationTermCfg) -> tuple[int, ...]:
        kind = cls._object_kind(term.field)
        if kind == "global":
            return ()
        entity_name = term.entity or bundle.primary_entity_name
        view = bundle.entity(entity_name)
        candidates = cls._all_ids(view, kind)
        selector = term.selector
        if selector is None or selector == "all":
            return candidates
        if selector == "root":
            if kind != "body":
                raise ValueError("The 'root' selector is valid only for body fields")
            return (view.root_body_id,)
        if selector == "head":
            cfg = bundle.entity_configs[entity_name]
            names = set(cfg.head_body_names)
            return tuple(
                object_id
                for object_id, name in zip(candidates, cls._names(view, kind, bundle), strict=True)
                if name in names
            )
        if not isinstance(selector, SemanticSelector):
            raise TypeError(
                f"Mutation selector must be None, 'all', 'root', 'head', or SemanticSelector; "
                f"got {type(selector).__name__}"
            )
        names = cls._names(view, kind, bundle)
        if selector.mode == "body_subtree":
            pattern = re.compile(selector.pattern or "")
            body_roots = {
                int(body_id)
                for body_id, name in zip(view.body_ids, view.body_names, strict=True)
                if pattern.search(name)
            }
            if not body_roots:
                raise ValueError(
                    f"Mutation selector {selector!r} matched no body roots in {entity_name!r}"
                )
            model = bundle.native_model
            descendants: set[int] = set()
            for body_id in view.body_ids:
                current = int(body_id)
                while current >= 0:
                    if current in body_roots:
                        descendants.add(int(body_id))
                        break
                    parent = int(model.body_parentid[current])
                    if parent == current:
                        break
                    current = parent
            if kind == "body":
                return tuple(object_id for object_id in candidates if object_id in descendants)
            if kind == "geom":
                return tuple(
                    object_id
                    for object_id in candidates
                    if int(model.geom_bodyid[object_id]) in descendants
                )
            if kind == "joint":
                return tuple(
                    object_id
                    for object_id in candidates
                    if int(model.jnt_bodyid[object_id]) in descendants
                )
            if kind == "dof":
                selected: list[int] = []
                for joint_id in view.non_free_joint_ids:
                    if int(model.jnt_bodyid[joint_id]) not in descendants:
                        continue
                    start = int(model.jnt_dofadr[joint_id])
                    width = _joint_qvel_width(int(model.jnt_type[joint_id]))
                    selected.extend(range(start, start + width))
                return tuple(selected)
            if kind == "actuator":
                selected = []
                for actuator_id in candidates:
                    if int(model.actuator_trntype[actuator_id]) == 0:
                        joint_id = int(model.actuator_trnid[actuator_id, 0])
                        if joint_id >= 0 and int(model.jnt_bodyid[joint_id]) in descendants:
                            selected.append(actuator_id)
                return tuple(selected)
            raise ValueError(
                f"Body-subtree selectors are unsupported for model field kind {kind!r}"
            )
        if selector.mode == "names":
            wanted = set(selector.names)
            selected = tuple(
                object_id
                for object_id, name in zip(candidates, names, strict=True)
                if name in wanted
            )
        elif selector.mode == "regex":
            pattern = re.compile(selector.pattern or "")
            selected = tuple(
                object_id
                for object_id, name in zip(candidates, names, strict=True)
                if pattern.search(name)
            )
        else:
            raise ValueError(
                "Body-subtree mutation selectors are not valid for model fields; "
                "select bodies/geoms explicitly"
            )
        if not selected:
            raise ValueError(
                f"Mutation selector {selector!r} matched no {kind} objects in {entity_name!r}"
            )
        return selected

    @staticmethod
    def _sample(
        env: Any,
        distribution: MutationDistributionCfg,
        shape: torch.Size,
        *,
        backend: Any | None = None,
    ) -> torch.Tensor:
        low = torch.as_tensor(distribution.low, dtype=env.bundle.dtype, device=env.bundle.device)
        high = torch.as_tensor(distribution.high, dtype=env.bundle.dtype, device=env.bundle.device)
        try:
            low = torch.broadcast_to(low, shape)
            high = torch.broadcast_to(high, shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Mutation distribution bounds {tuple(low.shape)}/{tuple(high.shape)} "
                f"cannot broadcast to target shape {tuple(shape)}"
            ) from exc
        if distribution.kind == "constant":
            return low
        sampler = backend or env.physics
        if distribution.kind == "uniform":
            return low + sampler.random_tensor(tuple(shape)) * (high - low)
        center = (low + high) / 2.0
        scale = (high - low) / 2.0
        return center + sampler.random_tensor(tuple(shape), normal=True) * scale

    @staticmethod
    def _assign(backend: Any, field: str, indices: tuple[int, ...], value: torch.Tensor) -> None:
        current = backend.model_field(field)
        index = torch.as_tensor(indices, dtype=torch.long, device=current.device)
        if current.ndim == 0:
            raise ValueError(f"Model field {field!r} is scalar and cannot be entity-selected")
        current[index] = value
        native = backend.native_model_field(field)
        native[index.detach().cpu().numpy()] = value.detach().cpu().numpy()

    def _apply_term(
        self, env: Any, backend: Any, bundle: Any, name: str, term: ModelMutationTermCfg
    ) -> None:
        field = term.field
        if field == "body_pseudo_inertia":
            if term.operation != "scale":
                raise ValueError("body_pseudo_inertia only supports the scale operation")
            # Sample one physically consistent factor and apply it to mass and
            # rotational inertia together. Two independent model mutation terms
            # could create impossible inertial combinations.
            body_term = replace(term, field="body_mass")
            indices = self._select(bundle, body_term)
            if not indices:
                raise ValueError(f"Mutation {name!r} selected no bodies")
            index = torch.as_tensor(indices, dtype=torch.long, device=bundle.device)
            mass = backend.model_field("body_mass")[index].clone()
            inertia = backend.model_field("body_inertia")[index].clone()
            sample = self._sample(env, term.distribution, mass.shape, backend=backend)
            self._assign(backend, "body_mass", indices, mass * sample)
            self._assign(backend, "body_inertia", indices, inertia * sample[..., None])
            self.records.append(
                MutationRecord(
                    name,
                    field,
                    term.entity,
                    indices,
                    term.operation,
                    term.distribution.kind,
                    sampled_value=tuple(float(item) for item in sample.detach().reshape(-1).cpu()),
                )
            )
            return
        if field.startswith("bam."):
            self._apply_bam_term(env, backend, field, term)
            self.records.append(
                MutationRecord(name, field, term.entity, (), term.operation, term.distribution.kind)
            )
            return
        if field not in SAFE_RUNTIME_MODEL_FIELDS:
            raise ValueError(
                f"Mutation field {field!r} is not safe to mutate after compilation; "
                "use an EntityTransform for topology, geometry, or transmission changes"
            )
        indices = self._select(bundle, term)
        if not indices:
            raise ValueError(f"Mutation {name!r} selected no objects for field {field!r}")
        current = backend.model_field(field)
        index = torch.as_tensor(indices, dtype=torch.long, device=current.device)
        current = backend.model_field(field)[index].clone()
        sample = self._sample(env, term.distribution, current.shape, backend=backend)
        operation = "multiply" if term.operation == "scale" else term.operation
        if operation == "set":
            value = sample
        elif operation == "add":
            value = current + sample
        elif operation == "multiply":
            value = current * sample
        else:
            raise ValueError(f"Unsupported mutation operation {term.operation!r}")
        self._assign(backend, field, indices, value)
        self.records.append(
            MutationRecord(
                name,
                field,
                term.entity,
                indices,
                term.operation,
                term.distribution.kind,
                sampled_value=tuple(float(item) for item in sample.detach().reshape(-1).cpu()),
            )
        )

    @staticmethod
    def _apply_bam_term(env: Any, backend: Any, field: str, term: ModelMutationTermCfg) -> None:
        parameters = backend.bundle.bam_parameters
        if parameters is None:
            raise ValueError(f"BAM mutation {field!r} requires a BAM physics backend")
        defaults: dict[str, Any] = {
            "bam.vin": parameters.vin,
            "bam.drop_gain": parameters.vin_drop_gain,
            "bam.friction_scale": 1.0,
            "bam.kp_scale": 1.0,
            "bam.kd_scale": 1.0,
        }
        current_values: dict[str, Any] = {
            "bam.vin": backend._bam_vin if backend._bam_vin is not None else defaults["bam.vin"],
            "bam.drop_gain": backend._bam_drop_gain
            if backend._bam_drop_gain is not None
            else defaults["bam.drop_gain"],
            "bam.friction_scale": backend._bam_friction_scale,
            "bam.kp_scale": backend._bam_kp_scale,
            "bam.kd_scale": backend._bam_kd_scale,
        }
        value = ModelMutationManager._sample(
            env, term.distribution, torch.Size(()), backend=backend
        )
        if term.operation not in {"set", "scale", "add"}:
            raise ValueError(f"Unsupported BAM mutation operation {term.operation!r}")
        current = torch.as_tensor(
            current_values[field], dtype=env.bundle.dtype, device=env.bundle.device
        )
        if term.operation == "set":
            value = value
        elif term.operation == "scale":
            value = current * value
        else:
            value = current + value
        if field == "bam.vin":
            backend.configure_bam(
                vin=value,
                drop_gain=backend._bam_drop_gain,
                friction_scale=backend._bam_friction_scale,
                kp_scale=backend._bam_kp_scale,
                kd_scale=backend._bam_kd_scale,
            )
        elif field == "bam.drop_gain":
            backend.configure_bam(
                vin=backend._bam_vin,
                drop_gain=value,
                friction_scale=backend._bam_friction_scale,
                kp_scale=backend._bam_kp_scale,
                kd_scale=backend._bam_kd_scale,
            )
        elif field == "bam.friction_scale":
            backend.configure_bam(
                vin=backend._bam_vin,
                drop_gain=backend._bam_drop_gain,
                friction_scale=value,
                kp_scale=backend._bam_kp_scale,
                kd_scale=backend._bam_kd_scale,
            )
        elif field == "bam.kp_scale":
            backend.configure_bam(
                vin=backend._bam_vin,
                drop_gain=backend._bam_drop_gain,
                friction_scale=backend._bam_friction_scale,
                kp_scale=value,
                kd_scale=backend._bam_kd_scale,
            )
        elif field == "bam.kd_scale":
            backend.configure_bam(
                vin=backend._bam_vin,
                drop_gain=backend._bam_drop_gain,
                friction_scale=backend._bam_friction_scale,
                kp_scale=backend._bam_kp_scale,
                kd_scale=value,
            )
        else:
            raise ValueError(f"Unsupported backend mutation field {field!r}")

    def reset(self, env: Any, env_ids: torch.Tensor | slice | None = None) -> None:
        """Restore nominal fields, then apply ordered mutations to selected rows."""

        env.physics.restore_model_defaults(env_ids)
        self.records.clear()
        ids = list(range(env.num_envs)) if env_ids is None else env._ids(env_ids).tolist()
        for index in ids:
            self._records_by_env.pop(int(index), None)
        if not env.domain_randomization:
            env.physics.configure_bam(
                vin=None,
                drop_gain=None,
                friction_scale=1.0,
                kp_scale=1.0,
                kd_scale=1.0,
            )
            self.last_records = ()
            return
        selected_backends = self._backends(env, env_ids)
        per_env_records: dict[int, list[MutationRecord]] = {index: [] for index in ids}
        for backend, _bundle, env_index in selected_backends:
            # Reinitialize backend-only parameters even when this task has no
            # BAM mutation terms; partial resets must not inherit a sibling's
            # or a prior episode's actuator randomization.
            parameters = backend.bundle.bam_parameters
            if parameters is not None:
                backend.configure_bam(
                    vin=parameters.vin,
                    drop_gain=parameters.vin_drop_gain,
                    friction_scale=1.0,
                    kp_scale=1.0,
                    kd_scale=1.0,
                )
            local_start = len(self.records)
            for name, term in self.terms.items():
                if not term.enabled:
                    continue
                self._apply_term(env, backend, backend.bundle, name, term)
                backend.recompute_model_constants()
            if env_index is not None:
                local_records = [
                    MutationRecord(
                        record.name,
                        record.field,
                        record.entity,
                        record.selected_indices,
                        record.operation,
                        record.distribution,
                        env_id=env_index,
                        sampled_value=record.sampled_value,
                    )
                    for record in self.records[local_start:]
                ]
                per_env_records[env_index].extend(local_records)
                self.records[local_start:] = local_records
        self._records_by_env.update(
            {index: tuple(records) for index, records in per_env_records.items()}
        )
        self.last_records = tuple(
            record
            for index in sorted(self._records_by_env)
            for record in self._records_by_env[index]
        )


__all__ = ["ModelMutationManager", "MutationRecord"]
