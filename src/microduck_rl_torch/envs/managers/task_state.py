"""Explicit lifecycle for persistent task-specific state."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import torch

from ..task_config import TaskStateTermCfg, TermCollection
from .base import reset_term, resolve_term


class TaskStateTerm:
    """Base contract for stateful task components.

    Subclasses may implement any of ``reset``, ``pre_physics``,
    ``post_physics``, ``compute``, and ``step``.  The methods receive
    ``env_ids`` on reset and ``dt`` on lifecycle callbacks.  ``compute(dt)`` is
    the upstream-style control-step hook; ``step(dt)`` is retained as a clear
    alias for task code written against this backend.
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

    def step(self, dt: float) -> None:
        self.compute(dt)


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
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            method()
            return
        if method_name == "reset":
            raise RuntimeError("Use TaskStateManager.reset for reset callbacks")
        if "dt" in signature.parameters:
            method(dt=self._dt)
        elif "env" in signature.parameters:
            method(env)
        elif any(
            parameter.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            and parameter.default is inspect.Parameter.empty
            for parameter in signature.parameters.values()
        ):
            # Legacy no-context task hooks are supported, but a required
            # positional argument is never guessed as an environment or dt.
            method()
        else:
            method()

    def reset(self, env: Any, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.data.clear()
            # Lifecycle callers use ``None`` to mean a full reset, while the
            # upstream state-term contract still receives explicit rows.
            reset_ids: torch.Tensor | slice | None = torch.arange(
                env.num_envs, device=env.bundle.device
            )
        else:
            reset_ids = env_ids
        for name, cfg in self.terms.items():
            if not cfg.enabled:
                continue
            if getattr(cfg, "func", None) is not None and not inspect.isclass(cfg.func):
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

    def step(self, env: Any, dt: float) -> None:
        self._dt = dt
        for name, cfg in self.terms.items():
            if cfg.enabled:
                term = self._term(name, cfg, env)
                # ``TaskStateTerm`` supplies a no-op ``compute`` default and
                # a ``step`` alias.  Prefer a subclass's explicit override;
                # otherwise a term implementing only the legacy ``step`` hook
                # must not be silently skipped by the inherited no-op.
                compute_overridden = "compute" in type(term).__dict__
                if compute_overridden and callable(getattr(term, "compute", None)):
                    self._invoke(term, "compute", env)
                elif callable(getattr(term, "step", None)):
                    self._invoke(term, "step", env)
                if hasattr(term, "data"):
                    self.data[name] = term.data

    def get_term(self, name: str) -> Any:
        try:
            return self._resolved_terms[name]
        except KeyError as exc:
            raise KeyError(f"Task-state term {name!r} is not active") from exc


__all__ = ["TaskStateManager", "TaskStateTerm"]
