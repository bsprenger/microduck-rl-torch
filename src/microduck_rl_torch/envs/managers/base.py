"""Small manager primitives used by directly-instantiated task environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskRuntimeContext:
    """Stable callback context; task terms should depend on this, not globals."""

    env: Any

    @property
    def bundle(self) -> Any:
        return self.env.bundle

    @property
    def data(self) -> Any:
        return self.env.data

    @property
    def state(self) -> Any:
        return self.env.state


class Manager:
    """Base class documenting the lifecycle surface shared by all managers."""

    def reset(self, _env: Any) -> None:
        return None
