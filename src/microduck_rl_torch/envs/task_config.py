"""Mutable, upstream-shaped task configuration primitives.

Task modules use fresh base configurations and mutate ordered term collections
in the same style as ``mjlab_microduck.tasks``.  The environment lifecycle is
independent of the configuration classes, which keeps task composition
testable without loading MuJoCo.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import CommandConfig
from .scene import SceneCfg

TermFunction = Callable[..., Any]


@dataclass
class TermCfg:
    """One named manager term."""

    func: Callable[..., Any] | None = None
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    # Upstream termination configs classify timeout terms explicitly rather
    # than relying on a reserved dictionary key.
    time_out: bool = False

    def clone(self) -> TermCfg:
        return deepcopy(self)


class TermCollection(MutableMapping[str, TermCfg]):
    """Ordered heterogeneous term mapping with explicit mutations.

    Upstream has separate term configuration classes for each manager.  The
    Torch implementation keeps one ordered mutation surface while allowing
    manager-specific term dataclasses to live in this module.
    """

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        self._values: OrderedDict[str, Any] = OrderedDict(values or ())

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = value

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def add(self, name: str, term: Any) -> None:
        if name in self._values:
            raise KeyError(f"Term {name!r} already exists; use replace() for mutation")
        self[name] = term

    def replace(self, name: str, term: Any) -> None:
        if name not in self._values:
            raise KeyError(f"Cannot replace missing term {name!r}")
        self[name] = term

    def remove(self, name: str) -> None:
        del self[name]

    def clone(self) -> TermCollection:
        return TermCollection(
            OrderedDict(
                (
                    name,
                    term.clone() if hasattr(term, "clone") else deepcopy(term),
                )
                for name, term in self.items()
            )
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._values)


@dataclass
class ObservationTermCfg:
    """One ordered observation term in an upstream-style observation group."""

    func: TermFunction | None = None
    params: dict[str, Any] = field(default_factory=dict)
    scale: float = 1.0
    noise: Callable[..., torch.Tensor] | None = None
    noise_params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def clone(self) -> ObservationTermCfg:
        return deepcopy(self)


@dataclass
class ObservationGroupCfg:
    """One actor or critic group made from ordered observation terms."""

    terms: TermCollection = field(default_factory=TermCollection)
    expected_size: int | None = None
    enabled: bool = True


@dataclass
class ObservationGroupsCfg:
    groups: dict[str, ObservationGroupCfg] = field(default_factory=dict)

    def clone(self) -> ObservationGroupsCfg:
        return deepcopy(self)


@dataclass
class EventTermCfg:
    """One event callback and its upstream-compatible execution mode."""

    func: TermFunction | None = None
    mode: str = "pre_physics"
    interval_range_s: tuple[float, float] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    requires_domain_randomization: bool = False

    def clone(self) -> EventTermCfg:
        return deepcopy(self)


@dataclass
class ActionCfg:
    size: int = 14
    scale: float = 1.0
    delay_lag: int | tuple[int, int] = 0
    actuator_mode: str = "bam"
    actuator_delay_lag: int | tuple[int, int] = 0


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
    # attach their own task configuration without changing the manager/env
    # lifecycle.
    task: Any
    physics_timestep: float = 0.005
    decimation: int = 4
    # Upstream reward managers multiply the configured weighted sum by the
    # control timestep. The first Torch policy contract predates that behavior,
    # so it is an explicit task-level choice rather than an implicit surprise.
    reward_scale_by_dt: bool = False
    play: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action_size(self) -> int:
        return self.actions.size

    def clone(self) -> TaskEnvCfg:
        return deepcopy(self)


def empty_terms() -> TermCollection:
    return TermCollection()
