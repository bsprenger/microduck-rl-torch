"""Canonical construction helpers for declarative extension points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def construct(factory: Callable[..., Any], cfg: Any, env: Any = ...) -> Any:
    """Construct an extension through one explicit protocol.

    Stateful extensions always use ``factory(cfg, env)``. Spec-only factories
    use ``factory(cfg)`` by calling this helper without ``env``. There is no
    signature probing or retry: malformed extensions fail at their boundary.
    """

    return factory(cfg) if env is ... else factory(cfg, env)


__all__ = ["construct"]
