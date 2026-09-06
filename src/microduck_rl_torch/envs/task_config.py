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

from ..rendering.config import RenderConfig
from .config import CommandConfig
from .dispatch import construct
from .scene import SceneCfg

TermFunction = Callable[..., Any]


@dataclass
class TermCfg:
    """One named manager term."""

    func: Callable[..., Any] | None = None
    class_type: type[Any] | None = None
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    # Upstream termination configs classify timeout terms explicitly rather
    # than relying on a reserved dictionary key.
    time_out: bool = False

    def clone(self) -> TermCfg:
        return deepcopy(self)


@dataclass
class TaskStateTermCfg:
    """Configuration for one persistent task-state component.

    The referenced class is constructed once with ``(cfg, env)`` and receives
    explicit reset and lifecycle callbacks from ``TaskStateManager``.  State
    therefore cannot accidentally leak into a global module or the physics
    backend between episodes.
    """

    func: Callable[..., Any] | None = None
    class_type: type[Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def clone(self) -> TaskStateTermCfg:
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
class ActionTermCfg:
    """Configuration for one composed action term.

    The term owns the semantics of one policy-action slice.  The action
    manager only validates, splits, and routes those slices, matching mjlab's
    ``ActionTermCfg``/``ActionTerm`` boundary.
    """

    entity: str = "robot"
    # Upstream calls this field ``entity_name``.  ``entity`` remains a concise
    # spelling for the Torch task configs; when both are provided the
    # upstream-compatible name wins.
    entity_name: str | None = None
    func: Callable[..., Any] | None = None
    class_type: type[Any] | None = None
    # Upstream action configs derive their width from the built term's
    # ``action_dim``.  ``None`` preserves that extension point; built-in task
    # configs may still declare a size to validate their policy contract.
    size: int | None = None
    actuator_names: tuple[str, ...] = ()
    scale: float | tuple[float, ...] | dict[str, float] = 1.0
    offset: float | tuple[float, ...] | dict[str, float] | str = 0.0
    clip: tuple[float, float] | dict[str, tuple[float, float]] | None = None
    target_type: str = "position"
    # ``actuator`` is the normal compiled control channel.  External task
    # terms may instead write a body wrench through the entity data view; this
    # keeps free props and task-specific actuators composable without faking
    # an actuator for them.
    destination: str = "actuator"
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def clone(self) -> ActionTermCfg:
        return deepcopy(self)

    @property
    def target_entity(self) -> str:
        return self.entity_name or self.entity

    def build(self, env: Any) -> Any:
        """Build the action term through the upstream extension boundary."""

        from .managers.actions import FunctionalActionTerm

        factory = self.class_type or self.func
        if factory is None:
            raise ValueError("Action term has no class_type or func")
        if isinstance(factory, type):
            return construct(factory, self, env)
        return FunctionalActionTerm(self, env, factory)


@dataclass
class JointPositionActionTermCfg(ActionTermCfg):
    """Built-in position-target action term.

    ``offset="default"`` makes the action relative to the selected entity's
    actuator home pose, which preserves the current MicroDuck policy contract.
    If ``joint_names`` is empty, all actuators belonging to ``entity`` are
    controlled in compiled actuator order.
    """

    joint_names: tuple[str, ...] = ()

    def build(self, env: Any) -> Any:
        from .managers.actions import JointPositionActionTerm

        return JointPositionActionTerm(self, env)


@dataclass
class ActionCfg(MutableMapping[str, ActionTermCfg]):
    """Ordered action-term composition with current scalar defaults.

    Constructing ``ActionCfg(size=14, scale=1.0)`` creates one built-in
    upstream-named ``joint_pos`` term. Passing ``terms=OrderedDict()`` is an
    explicit no-action configuration; omission of ``terms`` requests the
    default. New task families should mutate ``cfg.actions`` directly and can
    replace this with any number of action terms.
    """

    size: int = 14
    scale: float = 1.0
    delay_lag: int | tuple[int, int] = 0
    actuator_mode: str = "bam"
    actuator_delay_lag: int | tuple[int, int] = 0
    terms: OrderedDict[str, ActionTermCfg] | None = None

    def __post_init__(self) -> None:
        if self.terms is None:
            self.terms = OrderedDict()
            self.terms["joint_pos"] = JointPositionActionTermCfg(
                entity="robot",
                size=self.size,
                scale=self.scale,
                offset="default",
            )
        else:
            self.size = self.total_size

    def __getitem__(self, name: str) -> ActionTermCfg:
        return self.terms[name]

    def __setitem__(self, name: str, term: ActionTermCfg) -> None:
        if not isinstance(term, ActionTermCfg):
            raise TypeError(f"Expected ActionTermCfg for {name!r}")
        self.terms[name] = term
        self.size = self.total_size

    def __delitem__(self, name: str) -> None:
        del self.terms[name]
        self.size = self.total_size

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def add(self, name: str, term: ActionTermCfg) -> None:
        if name in self.terms:
            raise KeyError(f"Action term {name!r} already exists; use replace()")
        self[name] = term

    def replace(self, name: str, term: ActionTermCfg) -> None:
        if name not in self.terms:
            raise KeyError(f"Cannot replace missing action term {name!r}")
        self[name] = term

    def remove(self, name: str) -> None:
        del self[name]

    @property
    def total_size(self) -> int:
        return sum(term.size or 0 for term in self.terms.values() if term.enabled)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.terms)


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
    # Stateful task terms have an explicit environment-owned lifecycle.  This
    # is intentionally separate from physics and from velocity SensorState.
    task_state: TermCollection = field(default_factory=TermCollection)
    physics_timestep: float = 0.005
    decimation: int = 4
    # Rendering is opt-in at runtime, but camera/backend defaults belong to
    # the task configuration just as they do in mjlab's viewer config.
    viewer: RenderConfig = field(default_factory=RenderConfig)
    # Upstream reward managers multiply the configured weighted sum by the
    # control timestep. The first Torch policy contract predates that behavior,
    # so it is an explicit task-level choice rather than an implicit surprise.
    reward_scale_by_dt: bool = False
    # Match the manager-based RL lifecycle: completed rows are reset at the
    # end of the transition, while callers can opt into explicit reset mode.
    auto_reset: bool = True
    # The current mujoco-torch device compiler lacks some MuJoCo collision
    # kernels. ``error`` makes that limitation explicit for parity runs;
    # ``approximate`` is the usable CPU compatibility mode for those scenes.
    collision_policy: str = "approximate"
    play: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action_size(self) -> int:
        return self.actions.total_size

    def clone(self) -> TaskEnvCfg:
        return deepcopy(self)


def empty_terms() -> TermCollection:
    return TermCollection()
