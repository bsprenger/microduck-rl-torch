"""Mutable, upstream-shaped task configuration primitives.

Task modules use fresh base configurations and mutate ordered term collections
in the same style as ``mjlab_microduck.tasks``.  The runtime is independent of
the configuration classes, which keeps task composition testable without
loading MuJoCo.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import CommandConfig
from .scene import SceneCfg

TermFunction = Callable[[Any], torch.Tensor]


@dataclass
class TermCfg:
    """One named manager term."""

    func: TermFunction | None = None
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def clone(self) -> TermCfg:
        return deepcopy(self)


class TermCollection(MutableMapping[str, TermCfg]):
    """Ordered term mapping with explicit add/replace/remove mutations."""

    def __init__(self, values: OrderedDict[str, TermCfg] | None = None) -> None:
        self._values: OrderedDict[str, TermCfg] = values or OrderedDict()

    def __getitem__(self, key: str) -> TermCfg:
        return self._values[key]

    def __setitem__(self, key: str, value: TermCfg) -> None:
        if not isinstance(value, TermCfg):
            raise TypeError(f"Expected TermCfg for {key!r}, got {type(value).__name__}")
        self._values[key] = value

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def add(self, name: str, term: TermCfg) -> None:
        if name in self._values:
            raise KeyError(f"Term {name!r} already exists; use replace() for mutation")
        self[name] = term

    def replace(self, name: str, term: TermCfg) -> None:
        if name not in self._values:
            raise KeyError(f"Cannot replace missing term {name!r}")
        self[name] = term

    def remove(self, name: str) -> None:
        del self[name]

    def clone(self) -> TermCollection:
        return TermCollection(OrderedDict((name, term.clone()) for name, term in self.items()))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._values)


@dataclass
class ObservationGroupCfg:
    """One actor or critic observation group."""

    builder: Callable[[Any], torch.Tensor] | None = None
    expected_size: int | None = None
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationGroupsCfg:
    groups: dict[str, ObservationGroupCfg] = field(default_factory=dict)

    def clone(self) -> ObservationGroupsCfg:
        return deepcopy(self)


@dataclass
class ActionCfg:
    size: int = 14
    scale: float = 1.0
    delay_lag: int | tuple[int, int] = 0
    actuator_mode: str = "bam"


@dataclass
class TaskEnvCfg:
    """Complete declarative configuration for one directly-instantiated task."""

    task_name: str
    scene: SceneCfg
    actions: ActionCfg
    commands: CommandConfig
    observations: ObservationGroupsCfg
    rewards: TermCollection
    terminations: TermCollection
    events: TermCollection
    curriculum: TermCollection
    # The first task uses MicroDuckVelocityConfig. Future task families may
    # add their own runtime state/configuration without changing this schema.
    runtime: Any
    play: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action_size(self) -> int:
        return self.actions.size

    def clone(self) -> TaskEnvCfg:
        return deepcopy(self)


def empty_terms() -> TermCollection:
    return TermCollection()
