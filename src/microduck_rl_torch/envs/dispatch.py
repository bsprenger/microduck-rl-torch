"""Signature-safe dispatch helpers for declarative extension points."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any


def construct(factory: Callable[..., Any], cfg: Any, env: Any = ...) -> Any:
    """Construct a configured extension exactly once after signature binding.

    Configuration objects in the upstream ecosystem are commonly implemented
    by classes accepting either ``(cfg, env)`` or just ``(cfg)``.  Binding the
    supported call shapes before invoking the factory preserves that ergonomic
    boundary without catching a ``TypeError`` raised *inside* user code.
    """

    if env is ...:
        candidates: Sequence[tuple[tuple[Any, ...], dict[str, Any]]] = (
            ((), {"cfg": cfg}),
            ((cfg,), {}),
        )
    else:
        candidates = (
            ((), {"cfg": cfg, "env": env}),
            ((cfg, env), {}),
            ((), {"cfg": cfg}),
            ((cfg,), {}),
        )
    return invoke_compatible(factory, candidates)


def invoke_compatible(
    function: Callable[..., Any],
    candidates: Sequence[tuple[tuple[Any, ...], dict[str, Any]]],
) -> Any:
    """Invoke the first candidate that binds to ``function``'s signature."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        # Opaque extension callables cannot be inspected.  Invoke the declared
        # primary shape once; any exception is then the extension's exception.
        args, kwargs = candidates[0]
        return function(*args, **kwargs)
    for args, kwargs in candidates:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return function(*args, **kwargs)
    raise TypeError(
        f"{function!r} does not accept any supported extension signature: "
        f"{tuple((args, tuple(kwargs)) for args, kwargs in candidates)!r}"
    )


__all__ = ["construct", "invoke_compatible"]
