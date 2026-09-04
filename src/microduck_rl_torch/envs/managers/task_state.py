"""Explicit lifecycle for persistent task-specific state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..task_config import TaskStateTermCfg, TermCollection
from .base import reset_term, resolve_term


class TaskStateTerm:
    """Base contract for stateful task components.

    Subclasses may implement ``reset(env_ids)``, ``pre_physics(dt)``,
    ``post_physics(dt)``, and ``compute(dt)``. The callback names are explicit
    so task state has one lifecycle vocabulary.
    """

    def __init__(self, cfg: TaskStateTermCfg, env: Any) -> None:
        self.cfg = cfg
        self.env = env

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        del env_ids

    def pre_physics(self, dt: float) -> None:
        del dt
        return None

    def post_physics(self, dt: float) -> None:
        del dt
        return None

    def compute(self, dt: float) -> None:
        del dt
        return None


@dataclass
class TaskStateManager:
    """Own and dispatch all persistent task-state components."""

    terms: TermCollection
    _resolved_terms: dict[str, Any] = field(default_factory=dict, init=False)
    data: dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def active_terms(self) -> tuple[str, ...]:
        return tuple(name for name, cfg in self.terms.items() if cfg.enabled)

    def _term(self, name: str, cfg: Any, env: Any) -> Any:
        if name not in self._resolved_terms:
            self._resolved_terms[name] = resolve_term(cfg, env)
        return self._resolved_terms[name]

    def _invoke(self, term: Any, method_name: str, env: Any) -> None:
        method = getattr(term, method_name, None)
        if not callable(method):
            return
        if method_name == "reset":
            raise RuntimeError("Use TaskStateManager.reset for reset callbacks")
        method(self._dt)

    def reset(self, env: Any, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.data.clear()
            # Lifecycle callers use ``None`` to mean a full reset, while state
            # terms still receive explicit rows.
            reset_ids: torch.Tensor | slice | None = torch.arange(
                env.num_envs, device=env.bundle.device
            )
        else:
            reset_ids = env_ids
        for name, cfg in self.terms.items():
            if not cfg.enabled:
                continue
            if getattr(cfg, "func", None) is not None and not isinstance(cfg.func, type):
                raise TypeError(
                    f"Task-state term {name!r} must be a class with lifecycle methods; "
                    "plain functions belong in command/event/observation terms"
                )
            term = self._term(name, cfg, env)
            reset_term(term, reset_ids)
            if hasattr(term, "data"):
                self.data[name] = term.data

    def pre_physics(self, env: Any, dt: float) -> None:
        self._dt = dt
        for name, cfg in self.terms.items():
            if cfg.enabled:
                self._invoke(self._term(name, cfg, env), "pre_physics", env)

    def post_physics(self, env: Any, dt: float) -> None:
        self._dt = dt
        for name, cfg in self.terms.items():
            if cfg.enabled:
                self._invoke(self._term(name, cfg, env), "post_physics", env)

    def compute(self, env: Any, dt: float) -> None:
        self._dt = dt
        for name, cfg in self.terms.items():
            if cfg.enabled:
                term = self._term(name, cfg, env)
                if not callable(getattr(term, "compute", None)):
                    raise TypeError(f"Task-state term {name!r} must implement compute(dt)")
                self._invoke(term, "compute", env)
                if hasattr(term, "data"):
                    self.data[name] = term.data

    def get_term(self, name: str) -> Any:
        try:
            return self._resolved_terms[name]
        except KeyError as exc:
            raise KeyError(f"Task-state term {name!r} is not active") from exc


__all__ = ["TaskStateManager", "TaskStateTerm"]
