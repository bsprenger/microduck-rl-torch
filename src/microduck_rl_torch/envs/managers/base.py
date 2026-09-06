"""Small manager primitives used by directly-instantiated task environments."""

from __future__ import annotations

import inspect
from typing import Any

import torch

from ..dispatch import construct


def resolve_term(term: Any, env: Any) -> Any:
    """Resolve a function or upstream-style stateful term instance.

    Upstream manager configs permit ``func`` to be either a plain callable or
    a class that is constructed once with ``(cfg, env)``.  Keeping resolution
    here lets every Torch manager expose the same extension point without
    making the environment depend on a task-specific runtime class.
    """

    func = getattr(term, "class_type", None) or term.func
    if func is None:
        return None
    if inspect.isclass(func):
        return construct(func, term, env)
    return func


def call_term(
    func: Any,
    env: Any,
    params: dict[str, Any] | None = None,
    *,
    env_ids: torch.Tensor | slice | None | object = ...,
) -> Any:
    """Call a term with compatible function- or upstream-style arguments.

    Plain Torch terms generally accept ``func(env, **params)``.  Upstream
    reset/step event terms commonly accept ``func(env, env_ids, **params)``.
    Signature inspection chooses the latter only when the callable declares
    ``env_ids`` (or a second positional parameter), so errors raised inside a
    term are not hidden by a speculative retry.
    """

    kwargs = dict(params or {})
    args: tuple[Any, ...] = (env,)
    if env_ids is not ...:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            parameters = tuple(signature.parameters.values())
            if "env_ids" in signature.parameters:
                parameter = signature.parameters["env_ids"]
                if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                    args = (env, env_ids)
                else:
                    kwargs["env_ids"] = env_ids
            elif (
                len(parameters) >= 2
                and parameters[1].kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                }
                and parameters[1].default is inspect.Parameter.empty
            ):
                args = (env, env_ids)
    return func(*args, **kwargs)


def reset_term(func: Any, env_ids: torch.Tensor | slice | None) -> Any:
    """Reset an optional stateful term using upstream's ``env_ids`` contract."""

    reset = getattr(func, "reset", None)
    if not callable(reset):
        return None
    try:
        signature = inspect.signature(reset)
    except (TypeError, ValueError):
        return reset(env_ids)
    if "env_ids" in signature.parameters:
        return reset(env_ids=env_ids)
    if signature.parameters:
        return reset(env_ids)
    return reset()


class Manager:
    """Base class documenting the lifecycle surface shared by all managers."""

    def reset(self, _env: Any) -> None:
        return None


__all__ = ["Manager", "call_term", "reset_term", "resolve_term"]
