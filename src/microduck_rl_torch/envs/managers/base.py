"""Small manager primitives used by directly-instantiated task environments."""

from __future__ import annotations

import inspect
from typing import Any

import torch

from ..dispatch import construct


def resolve_term(term: Any, env: Any) -> Any:
    """Resolve a function or stateful term instance.

    ``func`` may be either a plain callable or a class that is constructed once
    with ``(cfg, env)``. Keeping resolution here gives every manager the same
    extension point without a task-specific runtime class.
    """

    func = term.func
    if func is None:
        return None
    if inspect.isclass(func):
        return construct(func, term, env)
    return func


def call_term(
    func: Any,
    env: Any,
    params: dict[str, Any] | None = None,
) -> Any:
    """Call a non-event term through the canonical ``func(env, **params)`` protocol."""

    kwargs = dict(params or {})
    return func(env, **kwargs)


def call_event_term(
    func: Any,
    env: Any,
    env_ids: torch.Tensor | slice | None,
    params: dict[str, Any] | None = None,
) -> Any:
    """Call an event through the explicit ``(env, env_ids, ...)`` protocol."""

    return func(env, env_ids, **dict(params or {}))


def call_env_ids_term(
    func: Any,
    env: Any,
    env_ids: torch.Tensor | slice | None,
    params: dict[str, Any] | None = None,
) -> Any:
    """Call a command term through the explicit ``env_ids=`` protocol."""

    kwargs = dict(params or {})
    if "env_ids" in kwargs:
        raise ValueError("env_ids is manager-owned and cannot be overridden in term.params")
    kwargs["env_ids"] = env_ids
    return func(env, **kwargs)


def reset_term(func: Any, env_ids: torch.Tensor | slice | None) -> Any:
    """Reset an optional stateful term using the ``env_ids`` contract."""

    reset = getattr(func, "reset", None)
    if not callable(reset):
        return None
    return reset(env_ids)


class Manager:
    """Base class documenting the lifecycle surface shared by all managers."""

    def reset(self, _env: Any) -> None:
        return None


__all__ = [
    "Manager",
    "call_env_ids_term",
    "call_event_term",
    "call_term",
    "reset_term",
    "resolve_term",
]
