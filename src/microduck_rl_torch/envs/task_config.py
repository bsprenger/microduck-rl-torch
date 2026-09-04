"""Mutable task configuration primitives.

Task modules use fresh base configurations and mutate ordered term collections.
The environment lifecycle is independent of the configuration classes, which
keeps task composition testable without loading MuJoCo.
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


@dataclass(frozen=True)
class TaskRuntimeCfg:
    """Task-independent episode settings consumed by the environment core.

    Task-specific configuration remains in ``TaskEnvCfg.task``.  Keeping
    lifecycle settings here prevents generic managers from reaching into a
    velocity-task dataclass just to implement timeout or orientation checks.
    """

    episode_length_steps: int = 1000
    bad_orientation_degrees: float = 70.0

    def __post_init__(self) -> None:
        if self.episode_length_steps < 1:
            raise ValueError("episode_length_steps must be positive")
        if self.bad_orientation_degrees < 0.0:
            raise ValueError("bad_orientation_degrees must be non-negative")


@dataclass
class TermCfg:
    """One named manager term."""

    func: Callable[..., Any] | None = None
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    # A zero-weight term is normally omitted from arithmetic, but this explicit
    # escape hatch preserves terms with required transition side effects.
    execute_when_zero_weight: bool = False
    # Timeout terms are classified explicitly rather than by a reserved key.
    time_out: bool = False

    def clone(self) -> TermCfg:
        return deepcopy(self)


@dataclass
class RewardTermCfg(TermCfg):
    """Reward-term configuration."""


@dataclass
class TerminationTermCfg(TermCfg):
    """Termination-term configuration."""


@dataclass
class CurriculumTermCfg:
    """One curriculum callback using the ``(env, env_ids)`` protocol."""

    func: Callable[..., Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def clone(self) -> CurriculumTermCfg:
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
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def clone(self) -> TaskStateTermCfg:
        return deepcopy(self)


@dataclass
class ResetStateTermCfg:
    """One declarative mutation of an episode-start physical state.

    Reset terms run after the generic scene defaults/origins have been
    assembled and before the backend receives them.  They therefore provide a
    stable place for task-specific prone poses, prop placement, or initial
    momentum without teaching the lifecycle owner about any task family.
    Callbacks receive the canonical ``(env, reset_state, env_ids, **params)``
    arguments and may mutate or return ``ResetState``.
    """

    func: Callable[..., Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def clone(self) -> ResetStateTermCfg:
        return deepcopy(self)


@dataclass(frozen=True)
class MutationDistributionCfg:
    """Distribution used by a declarative model mutation."""

    kind: str = "uniform"
    low: float | tuple[float, ...] = 1.0
    high: float | tuple[float, ...] = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"uniform", "normal", "constant"}:
            raise ValueError(f"Unsupported mutation distribution {self.kind!r}")


@dataclass
class ModelMutationTermCfg:
    """Declarative mutation of a compiled model or backend parameter.

    ``field`` is a MuJoCo model field such as ``body_mass`` or
    ``geom_friction``.  ``entity`` and ``selector`` scope the field to an
    entity view; selectors may be ``None`` (all objects), ``"root"``, or a
    :class:`SemanticSelector`.  ``operation`` is applied to the captured
    default, so repeated resets never accumulate mutations.
    """

    field: str
    entity: str | None = None
    selector: Any = None
    operation: str = "scale"
    distribution: MutationDistributionCfg = field(default_factory=MutationDistributionCfg)
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> ModelMutationTermCfg:
        return deepcopy(self)


class TermCollection(MutableMapping[str, Any]):
    """Ordered heterogeneous term mapping with explicit mutations.

    The collection provides one ordered mutation surface while allowing
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
    """One ordered observation term in an observation group."""

    func: TermFunction | None = None
    params: dict[str, Any] = field(default_factory=dict)
    scale: float | tuple[float, ...] | torch.Tensor | None = 1.0
    noise: Callable[..., torch.Tensor] | Any | None = None
    noise_params: dict[str, Any] = field(default_factory=dict)
    clip: tuple[float, float] | None = None
    delay_min_lag: int = 0
    delay_max_lag: int = 0
    delay_per_env: bool = True
    delay_hold_prob: float = 0.0
    delay_update_period: int = 0
    delay_per_env_phase: bool = True
    history_length: int = 0
    flatten_history_dim: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.delay_min_lag < 0 or self.delay_max_lag < self.delay_min_lag:
            raise ValueError("Observation delay must satisfy 0 <= min_lag <= max_lag")
        if not 0.0 <= self.delay_hold_prob <= 1.0:
            raise ValueError("Observation delay hold probability must be in [0, 1]")
        if self.delay_update_period < 0 or self.history_length < 0:
            raise ValueError("Observation delay period and history length must be non-negative")

    def clone(self) -> ObservationTermCfg:
        return deepcopy(self)


@dataclass
class ObservationGroupCfg:
    """One actor or critic group made from ordered observation terms."""

    terms: TermCollection = field(default_factory=TermCollection)
    expected_size: int | None = None
    enabled: bool = True
    concatenate_terms: bool = True
    concatenate_dim: int = -1
    enable_corruption: bool = True
    history_length: int | None = None
    flatten_history_dim: bool = True


@dataclass
class ObservationGroupsCfg:
    groups: dict[str, ObservationGroupCfg] = field(default_factory=dict)

    def clone(self) -> ObservationGroupsCfg:
        return deepcopy(self)


@dataclass
class EventTermCfg:
    """One event callback and its execution mode."""

    func: TermFunction | None = None
    mode: str = "pre_physics"
    interval_range_s: tuple[float, float] | None = None
    is_global_time: bool = False
    min_step_count_between_reset: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    requires_domain_randomization: bool = False
    # State-writing events opt into a manager-owned forward barrier.  This is
    # required for events that write qpos/qvel/model state after integration;
    # task-local bookkeeping events can leave it false.
    mutates_physics: bool = False

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
    func: Callable[..., Any] | None = None
    # ``None`` derives the width from the built term's ``action_dim``; built-in
    # task configs may still declare a size to validate their policy contract.
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

    def build(self, env: Any) -> Any:
        """Build the action term through the configured extension boundary."""

        from .managers.actions import FunctionalActionTerm

        factory = self.func
        if factory is None:
            raise ValueError("Action term has no function or action-term class")
        if isinstance(factory, type):
            return construct(factory, self, env)
        return FunctionalActionTerm(self, env, factory)


@dataclass
class JointPositionActionTermCfg(ActionTermCfg):
    """Built-in position-target action term.

    ``offset="default"`` makes the action relative to the selected entity's
    actuator home pose, which preserves the current Microduck policy contract.
    If ``joint_names`` is empty, all actuators belonging to ``entity`` are
    controlled in compiled actuator order.
    """

    joint_names: tuple[str, ...] = ()

    def build(self, env: Any) -> Any:
        from .managers.actions import JointPositionActionTerm

        return JointPositionActionTerm(self, env)


@dataclass
class ActionCfg(MutableMapping[str, ActionTermCfg]):
    """Ordered action-term composition.

    Action destinations are always explicit.  An empty ``terms`` collection
    is a valid action-less configuration; task factories must declare which
    entity/action term owns each policy slice rather than inheriting a hidden
    Microduck robot action.
    """

    actuator_mode: str = "bam"
    terms: OrderedDict[str, ActionTermCfg] = field(default_factory=OrderedDict)
    _resolved_size: int | None = field(default=None, init=False, repr=False, compare=False)

    def __getitem__(self, name: str) -> ActionTermCfg:
        return self.terms[name]

    def __setitem__(self, name: str, term: ActionTermCfg) -> None:
        if not isinstance(term, ActionTermCfg):
            raise TypeError(f"Expected ActionTermCfg for {name!r}")
        self.terms[name] = term
        self._resolved_size = None

    def __delitem__(self, name: str) -> None:
        del self.terms[name]
        self._resolved_size = None

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
        if self._resolved_size is not None:
            return self._resolved_size
        sizes = [term.size for term in self.terms.values() if term.enabled]
        if any(size is None for size in sizes):
            raise RuntimeError(
                "Action width is unresolved; construct the environment so action terms "
                "can derive their dimensions"
            )
        return sum(int(size) for size in sizes if size is not None)

    @property
    def size(self) -> int:
        """Resolved policy width; action terms own their individual dimensions."""

        return self.total_size

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
    # Task-specific configuration attaches here without changing the
    # manager/environment lifecycle.
    task: Any = None
    # Physical episode-start state is assembled generically, then these terms
    # may mutate it for a task (for example a prone recovery or prop spawn).
    reset_state: TermCollection = field(default_factory=TermCollection)
    # Stateful task terms have an explicit environment-owned lifecycle.  This
    # is intentionally separate from physics and from sensor state.
    task_state: TermCollection = field(default_factory=TermCollection)
    # Model mutations are restored from captured defaults before every reset
    # and then sampled declaratively. The collection is generic so each task
    # can provide its own randomizer.
    model_mutations: TermCollection = field(default_factory=TermCollection)
    # Temporal sensor state is an explicit task/scene component.  ``None``
    # means the generic action/joint baseline only; velocity tasks opt in to
    # IMU/contact histories through ``SensorStateCfg``.
    sensor_state: Any | None = None
    physics_timestep: float = 0.005
    decimation: int = 4
    # Solver policy is part of the task contract.  In particular, generated
    # terrain needs a bounded fixed-iteration solve so a difficult contact
    # state cannot spend unbounded time in line search.  ``None`` preserves
    # MuJoCo's native default for tasks that do not need this guard.
    fixed_iterations: bool = False
    solver_iterations: int | None = None
    line_search_iterations: int | None = None
    # MuJoCo's native per-world contact allocation.  ``None`` preserves the
    # simulator heuristic; tasks with dense terrain/contact manifolds can set
    # the same explicit contact-allocation contract used by the task config.
    nconmax: int | None = None
    # Rendering is opt-in at runtime, but camera/backend defaults belong to
    # the task configuration just as they do in mjlab's viewer config.
    viewer: RenderConfig = field(default_factory=RenderConfig)
    # Reward scaling by the control timestep is an explicit task-level choice
    # rather than an implicit behavior.
    reward_scale_by_dt: bool = False
    # Match the manager-based RL lifecycle: completed rows are reset at the
    # end of the transition, while callers can opt into explicit reset mode.
    auto_reset: bool = True
    # ``error`` is the semantic default: unsupported geometry must be fixed or
    # explicitly opted into ``approximate`` rather than silently rewritten.
    collision_policy: str = "error"
    play: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    # Generic lifecycle settings are intentionally separate from the task
    # family's command/reward/randomization configuration.
    runtime: TaskRuntimeCfg | None = None
    # The primary entity is explicit when a task has more than one entity. If
    # omitted, the environment derives it from action terms, then ``robot``,
    # then the first scene entity.
    primary_entity: str | None = None

    def __post_init__(self) -> None:
        if self.collision_policy not in {"approximate", "error"}:
            raise ValueError("collision_policy must be 'approximate' or 'error'")
        if self.nconmax is not None and self.nconmax < 1:
            raise ValueError("nconmax must be positive when configured")
        if self.runtime is None:
            # Populate the generic runtime boundary once from task-level values
            # so managers can use one runtime configuration on every call.
            self.runtime = TaskRuntimeCfg(
                episode_length_steps=int(getattr(self.task, "episode_length_steps", 1000)),
                bad_orientation_degrees=float(getattr(self.task, "bad_orientation_degrees", 70.0)),
            )
        if self.primary_entity is not None and self.primary_entity not in self.scene.entities:
            raise ValueError(f"primary_entity {self.primary_entity!r} is not in scene.entities")

    @property
    def action_size(self) -> int:
        return self.actions.total_size

    def clone(self) -> TaskEnvCfg:
        return deepcopy(self)

    def default_disable_mesh_mesh_contacts(self) -> bool:
        """Return the task-scoped mesh-mesh safety policy."""

        configured = self.metadata.get("disable_mesh_mesh_contacts")
        return bool(configured) if configured is not None else False


def empty_terms() -> TermCollection:
    return TermCollection()
