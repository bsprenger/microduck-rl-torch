"""Composed action terms and action routing."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from ..task_config import ActionCfg, ActionTermCfg, JointPositionActionTermCfg
from .base import reset_term


class ActionTerm(ABC):
    """Runtime action term matching mjlab's process/apply contract."""

    def __init__(self, cfg: ActionTermCfg, env: Any) -> None:
        self.cfg = cfg
        self.env = env
        self._raw_action: torch.Tensor | None = None

    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def process_actions(self, actions: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_actions(self) -> None:
        """Write this term's target through the environment action sink."""
        raise NotImplementedError

    @property
    def raw_action(self) -> torch.Tensor:
        if self._raw_action is None:
            return torch.zeros(
                (self.action_dim,)
                if self.env.num_envs == 1
                else (self.env.num_envs, self.action_dim),
                dtype=self.env.bundle.dtype,
                device=self.env.bundle.device,
            )
        return self._raw_action

    @property
    def actuator_ids(self) -> torch.Tensor | None:
        """Actuator ids covered by this term, when statically knowable."""

        return None

    @property
    def target_type(self) -> str:
        return str(getattr(self.cfg, "target_type", "position"))

    @property
    def target(self) -> torch.Tensor | None:
        """Processed target values for duck-typed custom terms."""

        return None

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        del env_ids
        self._raw_action = None


def _as_vector(
    value: float | tuple[float, ...] | Mapping[str, float],
    size: int,
    *,
    name: str,
    target_names: tuple[str, ...] = (),
) -> torch.Tensor:
    if isinstance(value, Mapping):
        result = torch.zeros(size, dtype=torch.float32)
        if len(target_names) != size:
            raise ValueError(f"{name} has no target-name mapping for {size} values")
        for index, target_name in enumerate(target_names):
            matches = [
                float(item)
                for pattern, item in value.items()
                if target_name == pattern or re.search(pattern, target_name) is not None
            ]
            if len(matches) > 1:
                raise ValueError(f"{name} patterns overlap target {target_name!r}")
            if matches:
                result[index] = matches[0]
        return result
    if isinstance(value, tuple):
        if len(value) != size:
            raise ValueError(f"{name} has {len(value)} values; expected {size}")
        values = value
    else:
        values = (value,) * size
    return torch.tensor(values, dtype=torch.float32)


class JointPositionActionTerm(ActionTerm):
    """Map one raw action slice to semantic entity actuator targets."""

    def __init__(self, cfg: JointPositionActionTermCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        if self.target_type != "position":
            raise ValueError("JointPositionActionTerm only supports position targets")
        entity = env.entity(cfg.entity)
        self._entity = entity
        joint_actuators = tuple(
            actuator_id
            for actuator_id in entity.actuator_ids
            if bool(env.bundle.actuator_joint_mask[actuator_id])
        )
        if cfg.joint_names:
            selected: list[int] = []
            for actuator_id in joint_actuators:
                name = env.bundle.actuator_joint_names[actuator_id].removeprefix(f"{cfg.entity}/")
                if any(
                    name == pattern or re.search(pattern, name) is not None
                    for pattern in cfg.joint_names
                ):
                    selected.append(actuator_id)
            self._actuator_ids = tuple(selected)
        else:
            self._actuator_ids = joint_actuators
        if not self._actuator_ids:
            raise ValueError(f"Action term entity {cfg.entity!r} has no actuators")
        self._action_dim = len(self._actuator_ids) if cfg.size is None else int(cfg.size)
        if self._action_dim != len(self._actuator_ids):
            raise ValueError(
                f"Action term for {cfg.entity!r} declares {self._action_dim} values but resolves "
                f"{len(self._actuator_ids)} actuators"
            )
        model = env.bundle.native_model
        qpos_indices = [
            int(model.jnt_qposadr[int(model.actuator_trnid[actuator_id, 0])])
            for actuator_id in self._actuator_ids
        ]
        self._qpos_indices = torch.tensor(qpos_indices, dtype=torch.long, device=env.bundle.device)
        selected_names = tuple(
            env.bundle.actuator_joint_names[actuator_id].removeprefix(f"{cfg.entity}/")
            for actuator_id in self._actuator_ids
        )
        self._target_names = selected_names
        self._scale = _as_vector(
            cfg.scale,
            self.action_dim,
            name="action scale",
            target_names=selected_names,
        ).to(device=env.bundle.device, dtype=env.bundle.dtype)
        if cfg.offset == "default":
            self._offset = env.bundle.default_qpos.index_select(-1, self._qpos_indices)
        elif isinstance(cfg.offset, (tuple, Mapping)):
            self._offset = _as_vector(
                cfg.offset,
                self.action_dim,
                name="action offset",
                target_names=selected_names,
            ).to(device=env.bundle.device, dtype=env.bundle.dtype)
        else:
            self._offset = torch.full(
                (self.action_dim,),
                float(cfg.offset),
                dtype=env.bundle.dtype,
                device=env.bundle.device,
            )
        self._target = self._offset.clone()

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def process_actions(self, actions: torch.Tensor) -> None:
        expected = (
            (self.action_dim,) if self.env.num_envs == 1 else (self.env.num_envs, self.action_dim)
        )
        if actions.shape != expected:
            raise ValueError(
                f"Action term expected ({self.action_dim},), got {tuple(actions.shape)}"
            )
        self._raw_action = actions.clone()
        processed = self._offset + self._scale * actions
        if self.cfg.clip is not None:
            if isinstance(self.cfg.clip, dict):
                lower = torch.full_like(processed, -torch.inf)
                upper = torch.full_like(processed, torch.inf)
                for index, actuator_id in enumerate(self._actuator_ids):
                    joint_name = self.env.bundle.actuator_joint_names[actuator_id].removeprefix(
                        f"{self.cfg.entity}/"
                    )
                    for pattern, bounds in self.cfg.clip.items():
                        if joint_name == pattern or re.search(pattern, joint_name) is not None:
                            lower[..., index], upper[..., index] = bounds
                            break
                processed = torch.maximum(torch.minimum(processed, upper), lower)
            else:
                processed = torch.clamp(processed, *self.cfg.clip)
        self._target = processed

    def apply_actions(self) -> None:
        self.env.action_manager.write_target(
            self.actuator_ids, self._target, target_type=self.target_type
        )

    @property
    def actuator_ids(self) -> torch.Tensor:
        return torch.tensor(self._actuator_ids, dtype=torch.long, device=self.env.bundle.device)

    @property
    def target_ids(self) -> torch.Tensor:
        return self.actuator_ids

    @property
    def target_names(self) -> list[str]:
        return list(self._target_names)

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    @property
    def offset(self) -> torch.Tensor:
        return self._offset

    @property
    def target(self) -> torch.Tensor:
        return self._target


class FunctionalActionTerm(ActionTerm):
    """Adapter for a function-based custom action term.

    A function receives ``(env, actions, **params)`` and returns target values
    for the term's entity actuators.  Class-based terms should implement the
    ``ActionTerm`` methods directly and are instantiated with ``(cfg, env)``.
    """

    def __init__(self, cfg: ActionTermCfg, env: Any, function: Any) -> None:
        super().__init__(cfg, env)
        self.function = function
        entity = env.entity(cfg.entity)
        self._entity = entity
        self._destination = getattr(cfg, "destination", "actuator")
        if self._destination not in {"actuator", "ctrl", "external_wrench"}:
            raise ValueError(f"Unsupported action destination {self._destination!r}")
        if self._destination == "external_wrench":
            self._actuator_ids = ()
            self._target_names = ()
            if cfg.size is None:
                raise ValueError("External-wrench action terms must declare size")
        elif cfg.actuator_names:
            self._actuator_ids = tuple(
                actuator_id
                for actuator_id in entity.actuator_ids
                if any(
                    env.bundle.actuator_joint_names[actuator_id].removeprefix(f"{cfg.entity}/")
                    == pattern
                    or re.search(
                        pattern,
                        env.bundle.actuator_joint_names[actuator_id].removeprefix(f"{cfg.entity}/"),
                    )
                    is not None
                    for pattern in cfg.actuator_names
                )
            )
        else:
            self._actuator_ids = tuple(entity.actuator_ids)
        if self._destination != "external_wrench":
            self._target_names = tuple(
                env.bundle.actuator_joint_names[actuator_id].removeprefix(f"{cfg.entity}/")
                for actuator_id in self._actuator_ids
            )
        self._action_dim = len(self._actuator_ids) if cfg.size is None else int(cfg.size)
        if self._destination != "external_wrench" and self._action_dim != len(self._actuator_ids):
            raise ValueError(
                f"Functional action term declares {self._action_dim} values but entity "
                f"{cfg.entity!r} "
                f"has {len(self._actuator_ids)} actuators"
            )
        self._target = torch.zeros(
            (env.num_envs, self._action_dim) if env.num_envs > 1 else (self._action_dim,),
            dtype=env.bundle.dtype,
            device=env.bundle.device,
        )
        self._qpos_indices = torch.tensor(
            [
                int(
                    env.bundle.native_model.jnt_qposadr[
                        int(env.bundle.native_model.actuator_trnid[aid, 0])
                    ]
                )
                for aid in self._actuator_ids
                if bool(env.bundle.actuator_joint_mask[aid])
            ],
            dtype=torch.long,
            device=env.bundle.device,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def process_actions(self, actions: torch.Tensor) -> None:
        expected = (
            (self.action_dim,) if self.env.num_envs == 1 else (self.env.num_envs, self.action_dim)
        )
        if actions.shape != expected:
            raise ValueError(
                f"Action term expected ({self.action_dim},), got {tuple(actions.shape)}"
            )
        self._raw_action = actions.clone()
        value = self.function(self.env, actions, **self.cfg.params)
        value_tensor = torch.as_tensor(
            value, dtype=self.env.bundle.dtype, device=self.env.bundle.device
        )
        if value_tensor.shape != expected:
            raise ValueError(
                f"Functional action term returned {value_tensor.numel()} values; "
                f"expected {self.action_dim}"
            )
        if (
            self.cfg.offset == "default"
            and self.target_type == "position"
            and self._destination == "actuator"
        ):
            if len(self._qpos_indices) != len(self._actuator_ids):
                raise ValueError("A position action with offset='default' requires joint actuators")
            offset = self.env.bundle.default_qpos.index_select(-1, self._qpos_indices)
        elif isinstance(self.cfg.offset, (tuple, Mapping)):
            offset = _as_vector(
                self.cfg.offset,
                self.action_dim,
                name="action offset",
                target_names=self._target_names,
            ).to(self.env.bundle.device, self.env.bundle.dtype)
        elif self.cfg.offset == "default":
            offset = torch.zeros_like(value_tensor)
        else:
            offset = torch.full_like(value_tensor, float(self.cfg.offset))
        scale = _as_vector(
            self.cfg.scale,
            self.action_dim,
            name="action scale",
            target_names=self._target_names,
        ).to(self.env.bundle.device, self.env.bundle.dtype)
        self._target = offset + scale * value_tensor
        if self.cfg.clip is not None:
            if isinstance(self.cfg.clip, dict):
                lower = torch.full_like(self._target, -torch.inf)
                upper = torch.full_like(self._target, torch.inf)
                for index, name in enumerate(self._target_names):
                    for pattern, bounds in self.cfg.clip.items():
                        if name == pattern or re.search(pattern, name) is not None:
                            lower[..., index], upper[..., index] = bounds
                            break
                self._target = torch.maximum(torch.minimum(self._target, upper), lower)
            else:
                self._target = torch.clamp(self._target, *self.cfg.clip)

    def apply_actions(self) -> None:
        if self._destination == "external_wrench":
            values = self._target
            body_names = tuple(self.cfg.params.get("body_names", ()))
            if body_names:
                body_indices = torch.tensor(
                    [self._entity.body_names.index(name) for name in body_names],
                    dtype=torch.long,
                    device=self.env.bundle.device,
                )
            else:
                body_indices = torch.tensor([0], dtype=torch.long, device=self.env.bundle.device)
            expected = 6 * body_indices.numel()
            if values.shape[-1] != expected:
                raise ValueError(
                    f"External-wrench action expected {expected} values, got {values.shape[-1]}"
                )
            wrench = values.reshape(*values.shape[:-1], body_indices.numel(), 6)
            self._entity.data.write_external_wrench(
                wrench[..., :3], wrench[..., 3:], body_ids=body_indices
            )
            return
        if self._destination == "ctrl":
            self.env.action_manager.write_ctrl(self.actuator_ids, self._target)
            return
        self.env.action_manager.write_target(
            self.actuator_ids, self._target, target_type=self.target_type
        )

    @property
    def actuator_ids(self) -> torch.Tensor:
        return torch.tensor(self._actuator_ids, dtype=torch.long, device=self.env.bundle.device)

    @property
    def target(self) -> torch.Tensor:
        return self._target

    @property
    def target_ids(self) -> torch.Tensor:
        return self.actuator_ids

    @property
    def target_names(self) -> list[str]:
        return list(self._target_names)


@dataclass
class ActionManager:
    """Aggregate named action terms and compose their actuator targets."""

    config: ActionCfg
    _terms: dict[str, Any] = field(default_factory=dict, init=False)
    _term_slices: dict[str, slice] = field(default_factory=dict, init=False)
    _prepared_action: torch.Tensor | None = field(default=None, init=False)
    _raw_action: torch.Tensor | None = field(default=None, init=False)
    _prev_action: torch.Tensor | None = field(default=None, init=False)
    _prev_prev_action: torch.Tensor | None = field(default=None, init=False)
    _env: Any | None = field(default=None, init=False)
    _target: torch.Tensor | None = field(default=None, init=False)
    _direct_ctrl: torch.Tensor | None = field(default=None, init=False)
    _direct_ctrl_mask: torch.Tensor | None = field(default=None, init=False)
    _written_actuators: set[int] = field(default_factory=set, init=False)
    _ctrl_written: set[int] = field(default_factory=set, init=False)
    _target_type: str | tuple[str, ...] = field(default="position", init=False)
    _actuator_target_types: list[str] = field(default_factory=list, init=False)

    @property
    def total_action_dim(self) -> int:
        return self.config.total_size

    @property
    def action_term_dim(self) -> list[int]:
        return [term.action_dim for term in self._terms.values()]

    @property
    def active_terms(self) -> list[str]:
        return [name for name, cfg in self.config.items() if cfg.enabled]

    @property
    def action(self) -> torch.Tensor:
        if self._raw_action is None:
            raise RuntimeError("Action manager has not processed an action")
        return self._raw_action

    @property
    def last_action(self) -> torch.Tensor:
        """Current action-history value, with a zero reset baseline."""

        if self._raw_action is not None:
            return self._raw_action
        if self._env is None:
            raise RuntimeError("Action manager is not attached to an environment")
        shape = (
            (self.total_action_dim,)
            if self._env.num_envs == 1
            else (self._env.num_envs, self.total_action_dim)
        )
        return torch.zeros(shape, dtype=self._env.bundle.dtype, device=self._env.bundle.device)

    @property
    def prev_action(self) -> torch.Tensor:
        if self._prev_action is None:
            raise RuntimeError("Action manager has not processed an action")
        return self._prev_action

    @property
    def prev_prev_action(self) -> torch.Tensor:
        if self._prev_prev_action is None:
            raise RuntimeError("Action manager has not processed an action")
        return self._prev_prev_action

    @property
    def current_target(self) -> torch.Tensor | None:
        """Composed actuator target before any backend actuator delay."""

        return self._target

    @property
    def current_ctrl(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return direct ``data.ctrl`` values and their written-channel mask.

        Actuator-target terms and raw-control terms are separate action
        destinations. Keeping the mask alongside the values lets a
        backend compose both without treating an unwritten zero as a real
        command.
        """

        if self._direct_ctrl is None or self._direct_ctrl_mask is None:
            return None
        return self._direct_ctrl, self._direct_ctrl_mask

    @property
    def target_type(self) -> str | tuple[str, ...]:
        return self._target_type

    @property
    def actuator_target_types(self) -> tuple[str, ...]:
        """Per-actuator target semantics for entity data consumers."""

        return tuple(self._actuator_target_types)

    @staticmethod
    def _term_target_type(term: Any) -> str:
        return str(
            getattr(
                term, "target_type", getattr(getattr(term, "cfg", None), "target_type", "position")
            )
        )

    def _configured_actuator_target_types(self, env: Any) -> list[str]:
        """Resolve static per-actuator target semantics for reset-time state."""

        target_types = ["position"] * int(env.bundle.action_size)
        for term in self._terms.values():
            if getattr(getattr(term, "cfg", None), "destination", "actuator") != "actuator":
                continue
            actuator_ids = getattr(term, "actuator_ids", None)
            if actuator_ids is None:
                continue
            target_type = self._term_target_type(term)
            if target_type not in {"position", "velocity", "effort"}:
                raise ValueError(f"Unsupported action target type {target_type!r}")
            for actuator_id in torch.as_tensor(actuator_ids).reshape(-1).tolist():
                if actuator_id < 0 or actuator_id >= len(target_types):
                    raise ValueError(f"Action term contains out-of-range actuator {actuator_id}")
                target_types[actuator_id] = target_type
        return target_types

    def _reset_target_baseline(self, env: Any) -> tuple[torch.Tensor | None, list[str]]:
        """Build the target baseline appropriate for configured actuator modes."""

        target_types = self._configured_actuator_target_types(env)
        has_actuator_destination = any(
            getattr(getattr(term, "cfg", None), "destination", "actuator") == "actuator"
            for term in self._terms.values()
        )
        if not has_actuator_destination:
            return None, target_types
        term_target_types = {self._term_target_type(term) for term in self._terms.values()}
        invalid = term_target_types - {"position", "velocity", "effort"}
        if invalid:
            raise ValueError(f"Unsupported action target types {sorted(invalid)!r}")
        if term_target_types == {"position"}:
            return env.bundle.default_pose.clone(), target_types
        target = env.bundle.default_ctrl.clone()
        position_ids = [index for index, value in enumerate(target_types) if value == "position"]
        if position_ids:
            ids = torch.as_tensor(position_ids, dtype=torch.long, device=target.device)
            target[ids] = env.bundle.default_pose.index_select(-1, ids)
        return target, target_types

    def _set_target_type_summary(self, target_types: list[str]) -> None:
        if len(set(target_types)) > 1:
            self._target_type = tuple(target_types)
        elif target_types:
            self._target_type = target_types[0]
        else:
            self._target_type = "position"

    def _prepare_terms(self, env: Any) -> None:
        if self._terms:
            return
        offset = 0
        declared_sizes = [
            int(cfg.size) for cfg in self.config.values() if cfg.enabled and cfg.size is not None
        ]
        has_unresolved_declared_size = any(
            cfg.enabled and cfg.size is None for cfg in self.config.values()
        )
        declared_total = None if has_unresolved_declared_size else sum(declared_sizes)
        used_actuators: set[int] = set()
        for name, cfg in self.config.items():
            if not cfg.enabled:
                continue
            term = cfg.build(env)
            if not isinstance(term, ActionTerm):
                for method in ("process_actions", "apply_actions"):
                    if not callable(getattr(term, method, None)):
                        raise TypeError(f"Action term {name!r} must implement {method}")
            term_size = int(term.action_dim)
            if term_size < 1:
                raise ValueError(f"Action term {name!r} must have positive size")
            if cfg.size is not None and cfg.size < 1:
                raise ValueError(f"Action term {name!r} must have positive size")
            if cfg.size is not None and term_size != cfg.size:
                raise ValueError(
                    f"Action term {name!r} reports dimension {term.action_dim}; expected {cfg.size}"
                )
            ids = getattr(term, "actuator_ids", None)
            if ids is not None:
                for actuator_id in ids.tolist():
                    if actuator_id in used_actuators:
                        raise ValueError(f"Action terms overlap actuator {actuator_id}")
                    used_actuators.add(actuator_id)
            self._terms[name] = term
            self._term_slices[name] = slice(offset, offset + term_size)
            offset += term_size
        # The resolved width becomes the task's public action-space width. This
        # is what lets configs omit a redundant ``size`` field.
        self.config._resolved_size = offset
        if declared_total is not None and offset != declared_total:
            raise RuntimeError(f"Action configuration width is {declared_total}; resolved {offset}")

    def prepare_terms(self, env: Any) -> None:
        """Resolve action terms once during environment construction."""

        self._env = env
        self._prepare_terms(env)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        env = self._env
        if env is None:
            raise RuntimeError("Action manager is not attached to an environment")
        self._prepare_terms(env)
        for term in self._terms.values():
            reset_term(term, env_ids)
        partial = env_ids is not None and self._raw_action is not None
        ids = self._ids(env_ids, env.num_envs) if partial else None
        if partial:
            assert ids is not None
            for field_name in (
                "_prepared_action",
                "_raw_action",
                "_prev_action",
                "_prev_prev_action",
            ):
                value = getattr(self, field_name)
                if isinstance(value, torch.Tensor) and value.ndim == 2:
                    value[ids] = 0
            baseline, target_types = self._reset_target_baseline(env)
            if baseline is None:
                self._target = None
            elif isinstance(self._target, torch.Tensor) and self._target.ndim == 2:
                self._target[ids] = baseline
            else:
                self._target = baseline.unsqueeze(0).expand(env.num_envs, -1).clone()
            if isinstance(self._direct_ctrl, torch.Tensor) and self._direct_ctrl.ndim == 2:
                self._direct_ctrl[ids] = 0
            if (
                isinstance(self._direct_ctrl_mask, torch.Tensor)
                and self._direct_ctrl_mask.ndim == 2
            ):
                self._direct_ctrl_mask[ids] = False
        else:
            self._prepared_action = None
            self._raw_action = None
            self._prev_action = None
            self._prev_prev_action = None
            self._target = None
            self._direct_ctrl = None
            self._direct_ctrl_mask = None
            target_types = ["position"] * env.bundle.action_size
        self._actuator_target_types = target_types
        self._set_target_type_summary(target_types)
        self._written_actuators.clear()

    @staticmethod
    def _ids(env_ids: torch.Tensor | slice, num_envs: int) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(num_envs)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        env = self._env
        if env is None:
            raise RuntimeError("Action manager is not attached to an environment")
        if env.data is None or env.state is None:
            raise RuntimeError("Call reset() before processing an action")
        self._prepare_terms(env)
        action = torch.as_tensor(action, dtype=env.bundle.dtype, device=env.bundle.device)
        if action.ndim == 2 and action.shape == (1, self.total_action_dim) and env.num_envs == 1:
            action = action[0]
        expected = (
            (self.total_action_dim,) if env.num_envs == 1 else (env.num_envs, self.total_action_dim)
        )
        if action.shape != expected:
            raise ValueError(f"Expected action shape {expected}, got {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            raise ValueError("Action contains non-finite values")
        self._prev_prev_action = (
            torch.zeros_like(action) if self._prev_action is None else self._prev_action.clone()
        )
        self._prev_action = (
            torch.zeros_like(action) if self._raw_action is None else self._raw_action.clone()
        )
        self._raw_action = action.clone()
        applied_action = action
        for name, term in self._terms.items():
            term.process_actions(
                applied_action[self._term_slices[name]]
                if applied_action.ndim == 1
                else applied_action[:, self._term_slices[name]]
            )
        self._prepared_action = applied_action.clone()
        return self._prepared_action

    def write_target(
        self,
        actuator_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        target_type: str = "position",
    ) -> None:
        """Validated target sink used by ``apply_actions`` terms."""

        if self._target is None:
            raise RuntimeError("Action target sink is not active")
        if target_type not in {"position", "velocity", "effort"}:
            raise ValueError(f"Unsupported action target type {target_type!r}")
        ids = torch.as_tensor(actuator_ids, dtype=torch.long, device=self._target.device).reshape(
            -1
        )
        values = torch.as_tensor(values, dtype=self._target.dtype, device=self._target.device)
        if values.ndim == 2:
            if self._target.ndim != 2 or values.shape[0] != self._target.shape[0]:
                raise ValueError("Batched action contribution has the wrong environment dimension")
            if values.shape[1] != ids.numel():
                raise ValueError(
                    f"Action contribution has {values.shape[1]} values for {ids.numel()} ids"
                )
        elif ids.numel() != values.numel():
            raise ValueError(
                f"Action contribution has {values.numel()} values for {ids.numel()} ids"
            )
        if ids.numel() and (ids.min() < 0 or ids.max() >= self._target.shape[-1]):
            raise ValueError("Action contribution contains an out-of-range actuator id")
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError("Action contribution contains duplicate actuator ids")
        overlap = self._written_actuators.intersection(ids.tolist())
        if overlap:
            raise ValueError(f"Action terms overlap actuators {sorted(overlap)!r}")
        self._written_actuators.update(ids.tolist())
        for actuator_id in ids.tolist():
            self._actuator_target_types[actuator_id] = target_type
        if self._target.ndim == 2:
            self._target[:, ids] = values
        else:
            self._target[ids] = values.reshape(-1)

    def write_ctrl(self, actuator_ids: torch.Tensor, values: torch.Tensor) -> None:
        """Write a raw-control action contribution into its own sink."""

        if self._target is None:
            raise RuntimeError("Action control sink is not active")
        ids = torch.as_tensor(actuator_ids, dtype=torch.long, device=self._target.device).reshape(
            -1
        )
        values = torch.as_tensor(values, dtype=self._target.dtype, device=self._target.device)
        if values.ndim == 2:
            if self._target.ndim != 2 or values.shape != (self._target.shape[0], ids.numel()):
                raise ValueError("Batched direct-control contribution has the wrong shape")
        elif values.ndim == 1:
            if self._target.ndim != 1 or values.numel() != ids.numel():
                raise ValueError("Direct-control contribution has the wrong shape")
        else:
            raise ValueError("Direct-control contribution must be one- or two-dimensional")
        if ids.numel() and (ids.min() < 0 or ids.max() >= self._target.shape[-1]):
            raise ValueError("Direct-control contribution contains an out-of-range actuator id")
        if self._direct_ctrl is None or self._direct_ctrl_mask is None:
            raise RuntimeError("Action control sink is not active")
        actuator_ids_list = ids.tolist()
        overlap = self._written_actuators.union(self._ctrl_written).intersection(actuator_ids_list)
        if overlap:
            raise ValueError(f"Direct-control terms overlap actuators {sorted(overlap)!r}")
        self._ctrl_written.update(actuator_ids_list)
        self._written_actuators.update(actuator_ids_list)
        if self._direct_ctrl.ndim == 2:
            self._direct_ctrl[:, ids] = values
            self._direct_ctrl_mask[:, ids] = True
        else:
            self._direct_ctrl[ids] = values
            self._direct_ctrl_mask[ids] = True

    def apply_action(self) -> torch.Tensor:
        """Compose all processed term targets into the full actuator vector."""

        env = self._env
        if env is None:
            raise RuntimeError("Action manager is not attached to an environment")
        if self._prepared_action is None:
            raise RuntimeError("Call process_action() before apply_action()")
        term_target_types = {self._term_target_type(term) for term in self._terms.values()}
        if not term_target_types:
            term_target_types = {"position"}
        invalid = term_target_types - {"position", "velocity", "effort"}
        if invalid:
            raise ValueError(f"Unsupported action target types {sorted(invalid)!r}")
        target = (
            env.bundle.default_pose.clone()
            if term_target_types == {"position"}
            else env.bundle.default_ctrl.clone()
        )
        has_actuator_destination = any(
            getattr(getattr(term, "cfg", None), "destination", "actuator") == "actuator"
            for term in self._terms.values()
        )
        if not has_actuator_destination and env.data is not None:
            # A task with only external-wrench/raw-control terms must not
            # synthesize a hidden joint-position command. Hold the current
            # actuator state as the backend's required full-width no-op target.
            measured, _ = env.physics.actuator_measurements()
            target = measured.clone()
        if env.num_envs > 1:
            target = target.unsqueeze(0).expand(env.num_envs, -1).clone()
        self._target = target
        self._direct_ctrl = torch.zeros_like(target)
        self._direct_ctrl_mask = torch.zeros_like(target, dtype=torch.bool)
        self._written_actuators.clear()
        self._ctrl_written.clear()
        if env.data is not None:
            env.data = env.data.replace(xfrc_applied=torch.zeros_like(env.data.xfrc_applied))
        for term in self._terms.values():
            contribution = term.apply_actions()
            if contribution is None:
                if isinstance(term, ActionTerm):
                    # Native terms write through ``write_target`` themselves.
                    continue
                actuator_ids = getattr(term, "actuator_ids", None)
                values = getattr(term, "target", None)
                if actuator_ids is None or values is None:
                    raise TypeError(
                        f"Action term {type(term).__name__} must return a contribution or "
                        "expose actuator_ids and target"
                    )
                self.write_target(
                    actuator_ids,
                    values,
                    target_type=self._term_target_type(term),
                )
            else:
                actuator_ids, values = contribution
                self.write_target(
                    actuator_ids,
                    values,
                    target_type=self._term_target_type(term),
                )
        self._set_target_type_summary(self._actuator_target_types)
        return target

    def get_term(self, name: str) -> Any:
        try:
            return self._terms[name]
        except KeyError as exc:
            raise KeyError(f"Action term {name!r} is not active") from exc


__all__ = [
    "ActionManager",
    "ActionTerm",
    "FunctionalActionTerm",
    "JointPositionActionTerm",
]
