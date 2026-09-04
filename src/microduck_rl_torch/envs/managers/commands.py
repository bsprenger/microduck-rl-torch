"""Command-term manager."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import torch

from ..config import CommandConfig, CommandTermCfg, sample_twist, sample_uniform
from .base import call_env_ids_term, resolve_term


@dataclass(frozen=True)
class CommandTermView:
    """Live scalar-environment view used while evaluating command terms."""

    name: str
    manager: CommandManager

    @property
    def cfg(self) -> CommandTermCfg:
        return self.manager.get_term_cfg(self.name)

    @property
    def command(self) -> torch.Tensor:
        return self.manager.get_command(self.name)


def velocity_command(
    env: Any,
    *,
    twist_ranges: tuple[tuple[float, float], ...] | None = None,
    turn_in_place_fraction: float | None = None,
    standing_fraction: float | None = None,
    env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
    """Sample the current task's three-element velocity command."""

    ids = None if env_ids is None else env._ids(env_ids)
    generators = getattr(env.physics, "generators", None)
    if ids is not None and generators is not None:
        generators = tuple(generators[int(index)] for index in ids.tolist())
    result = sample_twist(
        getattr(env.config, "command", None),
        generator=generators if generators is not None else env._generator,
        device=env.bundle.device,
        dtype=env.bundle.dtype,
        twist_ranges=twist_ranges,
        turn_in_place_fraction=turn_in_place_fraction,
        standing_fraction=standing_fraction,
        batch_size=getattr(env, "num_envs", 1) if ids is None else ids.numel(),
    )
    return result.unsqueeze(0) if ids is not None and result.ndim == 1 else result


def head_pose_command(
    env: Any,
    *,
    ranges: tuple[tuple[float, float], ...] | None = None,
    env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
    """Sample the current task's four-element head-pose command."""

    ids = None if env_ids is None else env._ids(env_ids)
    generators = getattr(env.physics, "generators", None)
    if ids is not None and generators is not None:
        generators = tuple(generators[int(index)] for index in ids.tolist())
    result = sample_uniform(
        ranges or env.config.command.head_ranges,
        generator=generators if generators is not None else env._generator,
        device=env.bundle.device,
        dtype=env.bundle.dtype,
        batch_size=getattr(env, "num_envs", 1) if ids is None else ids.numel(),
    )
    return result.unsqueeze(0) if ids is not None and result.ndim == 1 else result


def body_pose_command(
    env: Any,
    *,
    ranges: tuple[tuple[float, float], ...] | None = None,
    env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
    """Sample the current task's six-element body-pose command."""

    ids = None if env_ids is None else env._ids(env_ids)
    generators = getattr(env.physics, "generators", None)
    if ids is not None and generators is not None:
        generators = tuple(generators[int(index)] for index in ids.tolist())
    result = sample_uniform(
        ranges or env.config.command.body_ranges,
        generator=generators if generators is not None else env._generator,
        device=env.bundle.device,
        dtype=env.bundle.dtype,
        batch_size=getattr(env, "num_envs", 1) if ids is None else ids.numel(),
    )
    return result.unsqueeze(0) if ids is not None and result.ndim == 1 else result


@dataclass
class CommandManager:
    """Concatenate named command terms and resample them independently.

    The manager has no knowledge of task-specific command slices. A task
    defines ordered terms with sizes and sampling functions, allowing posture,
    phase, and prop commands to share the same environment lifecycle.
    """

    config: CommandConfig
    command: torch.Tensor | None = None
    _fixed: bool = False
    _next_steps: dict[str, int | torch.Tensor] = field(default_factory=dict)
    _resolved_terms: dict[str, Any] = field(default_factory=dict, init=False)
    _resolved_sizes: dict[str, int] = field(default_factory=dict, init=False)
    _env: Any | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.command is not None:
            self.command = torch.as_tensor(self.command).clone()
            self._fixed = True

    @property
    def fixed(self) -> bool:
        return self._fixed

    @property
    def size(self) -> int:
        sizes: list[int] = []
        for name, term in self.config.terms.items():
            if not term.enabled:
                continue
            size = self._resolved_sizes.get(name, term.size)
            if size is None:
                raise RuntimeError(
                    f"Command term {name!r} has unresolved width; initialize the manager "
                    "with an environment before querying command size"
                )
            sizes.append(int(size))
        return sum(sizes)

    def _resolve_size(self, name: str, term: CommandTermCfg, env: Any) -> int:
        if term.size is not None:
            size = int(term.size)
        else:
            function = self._function(name, term, env)
            value = getattr(function, "command", None)
            if value is None:
                raise ValueError(
                    f"Command term {name!r} must declare size when its function is not "
                    "a stateful command term exposing command"
                )
            value = torch.as_tensor(value)
            if value.ndim < 1:
                raise ValueError(f"Command term {name!r} exposes a scalar command")
            size = int(value.shape[-1])
        if size < 1:
            raise ValueError(f"Command term {name!r} must have a positive size")
        self._resolved_sizes[name] = size
        return size

    def _validate_terms(self, env: Any) -> None:
        for name, term in self.config.terms.items():
            if not isinstance(term, CommandTermCfg):
                raise TypeError(f"Command term {name!r} is not a CommandTermCfg")
            if not term.enabled:
                continue
            self._resolve_size(name, term, env)
            if term.func is None:
                raise ValueError(f"Enabled command term {name!r} has no sampling function")
            if term.resampling_time_range is not None:
                low, high = term.resampling_time_range
                if low < 0 or low > high:
                    raise ValueError(
                        f"Command term {name!r} interval must satisfy 0 <= low <= high"
                    )

    def _slices(self) -> dict[str, slice]:
        offset = 0
        slices: dict[str, slice] = {}
        for name, term in self.config.terms.items():
            if not term.enabled:
                continue
            size = self._resolved_sizes.get(name, term.size)
            if size is None:
                raise RuntimeError(f"Command term {name!r} has unresolved width")
            slices[name] = slice(offset, offset + int(size))
            offset += int(size)
        return slices

    def _function(self, name: str, term: CommandTermCfg, env: Any) -> Any:
        if name not in self._resolved_terms:
            self._resolved_terms[name] = resolve_term(term, env)
        return self._resolved_terms[name]

    def _class_command(
        self,
        function: Any,
        env: Any,
        *,
        size: int,
        env_ids: torch.Tensor | slice | None = None,
    ) -> torch.Tensor | None:
        value = getattr(function, "command", None)
        if value is None:
            return None
        num_envs = getattr(env, "num_envs", 1)
        value = torch.as_tensor(value, dtype=env.bundle.dtype, device=env.bundle.device)
        if value.ndim == 2 and value.shape[0] == 1 and num_envs == 1:
            value = value[0]
        if env_ids is not None and num_envs > 1 and value.ndim == 2:
            ids = self._ids(env_ids, num_envs, env.bundle.device)
            if value.shape[0] == num_envs:
                value = value[ids]
            expected = (ids.numel(), size)
        else:
            expected = (size,) if num_envs == 1 else (num_envs, size)
        if value.shape != expected:
            raise ValueError(
                f"Command term returned {tuple(value.shape)} values; expected {expected}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("Command term returned non-finite values")
        return value

    def _sample(
        self,
        env: Any,
        name: str,
        term: CommandTermCfg,
        env_ids: torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        function = self._function(name, term, env)
        size = self._resolved_sizes[name]
        class_command = self._class_command(function, env, size=size)
        if class_command is not None:
            if env_ids is not None and class_command.ndim == 2:
                return class_command[self._ids(env_ids, env.num_envs, env.bundle.device)]
            return class_command
        if function is None:
            raise RuntimeError("Cannot sample a command term without a function")
        value = torch.as_tensor(
            call_env_ids_term(function, env, env_ids, term.params),
            dtype=env.bundle.dtype,
            device=env.bundle.device,
        )
        num_envs = getattr(env, "num_envs", 1)
        expected = (
            (size,)
            if env_ids is None and num_envs == 1
            else (
                (self._ids(env_ids, num_envs, env.bundle.device).numel(), size)
                if env_ids is not None
                else (num_envs, size)
            )
        )
        if value.shape != expected:
            raise ValueError(
                f"Command term returned {tuple(value.shape)} values; expected {expected}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("Command term returned non-finite values")
        return value

    def _reset_class_term(
        self, function: Any, env: Any, env_ids: torch.Tensor | slice | None
    ) -> None:
        reset = getattr(function, "reset", None)
        if not callable(reset):
            return
        if env_ids is None:
            env_ids = torch.arange(getattr(env, "num_envs", 1), device=env.bundle.device)
        reset(env_ids)

    def _compute_class_term(
        self,
        function: Any,
        env: Any,
        dt: float,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        compute = getattr(function, "compute", None)
        if not callable(compute):
            return
        if env_ids is not None:
            parameters = inspect.signature(compute).parameters
            accepts_env_ids = "env_ids" in parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
            if accepts_env_ids:
                compute(dt, env_ids=env_ids)
                return
        compute(dt)

    def compute(
        self,
        env: Any,
        *,
        dt: float = 0.0,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Update stateful command terms and publish their command tensors.

        Commands are computed once with ``dt=0`` after every reset. A separate
        method keeps that lifecycle boundary explicit and makes stateful
        phase/posture commands usable by directly-instantiated tasks.
        """

        if self.command is None:
            raise RuntimeError("Command manager must be reset before computing")
        if self._fixed:
            return
        self._validate_terms(env)
        slices = self._slices()
        for name, term in self.config.terms.items():
            if not term.enabled:
                continue
            function = self._function(name, term, env)
            if not callable(getattr(function, "compute", None)):
                continue
            self._compute_class_term(function, env, dt, env_ids)
            class_command = self._class_command(
                function,
                env,
                size=self._resolved_sizes[name],
                env_ids=env_ids,
            )
            if class_command is None:
                raise RuntimeError(
                    f"Stateful command term {name!r} must expose a command after compute()"
                )
            if self.command.ndim == 1:
                self.command[slices[name]] = class_command
            elif env_ids is not None:
                ids = self._ids(env_ids, env.num_envs, env.bundle.device)
                self.command[ids, slices[name]] = class_command
            else:
                self.command[:, slices[name]] = class_command

    def _schedule(
        self,
        env: Any,
        name: str,
        term: CommandTermCfg,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if term.resampling_time_range is None:
            self._next_steps.pop(name, None)
        else:
            num_envs = getattr(env, "num_envs", 1)
            sample_range = getattr(env, "_sample_range", None)
            if not callable(sample_range):
                next_step = getattr(env, "_next_interval_step", None)
                if not callable(next_step):
                    raise AttributeError(
                        "Command scheduling requires _sample_range or _next_interval_step"
                    )
                fallback = int(next_step(term.resampling_time_range))
                if num_envs == 1:
                    self._next_steps[name] = fallback
                else:
                    self._next_steps[name] = torch.full(
                        (num_envs,),
                        fallback,
                        dtype=torch.long,
                        device=env.bundle.device,
                    )
                return
            ids = None if env_ids is None else self._ids(env_ids, num_envs, env.bundle.device)
            sampled = torch.as_tensor(
                sample_range(*term.resampling_time_range, env_ids=ids),
                device=env.bundle.device,
            )
            step_counts = getattr(
                env,
                "step_counts",
                torch.tensor([env.step_count], dtype=torch.long, device=env.bundle.device),
            )
            if ids is not None:
                step_counts = step_counts[ids]
            steps = (
                torch.ceil(
                    sampled / (getattr(env.bundle, "timestep", 1.0) * getattr(env, "decimation", 1))
                )
                .clamp_min(1)
                .to(torch.long)
                + step_counts
            )
            if num_envs == 1:
                self._next_steps[name] = int(steps.reshape(-1)[0].item())
            elif env_ids is not None:
                scheduled = self._next_steps.get(name)
                if not isinstance(scheduled, torch.Tensor):
                    schedule = step_counts.clone()
                else:
                    schedule = scheduled.clone()
                assert ids is not None
                schedule[ids] = steps
                self._next_steps[name] = schedule
            else:
                self._next_steps[name] = steps.expand(num_envs).clone()

    def set_command(
        self,
        command: torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if env_ids is not None:
            if self._env is None or self.command is None:
                raise RuntimeError("A partial command update requires an initialized manager")
            ids = self._ids(
                env_ids,
                getattr(self._env, "num_envs", 1),
                self._env.bundle.device,
            )
            value = torch.as_tensor(command, dtype=self.command.dtype, device=self.command.device)
            expected = (ids.numel(), self.size)
            if value.ndim == 1 and value.shape == (self.size,):
                value = value.unsqueeze(0).expand(ids.numel(), -1)
            if value.shape != expected:
                raise ValueError(
                    f"Expected partial command shape {expected}, got {tuple(value.shape)}"
                )
            self.command[ids] = value
            return
        if self.command is None:
            value = torch.as_tensor(command)
        else:
            value = torch.as_tensor(command, dtype=self.command.dtype, device=self.command.device)
        num_envs = 1 if self._env is None else getattr(self._env, "num_envs", 1)
        if num_envs > 1 and value.shape == (self.size,):
            value = value.unsqueeze(0).expand(num_envs, -1).clone()
        expected = (self.size,) if num_envs == 1 else (num_envs, self.size)
        if value.shape != expected:
            raise ValueError(f"Expected command shape {expected}, got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError("Command contains non-finite values")
        self.command = value.clone()
        self._fixed = True

    @property
    def active_terms(self) -> list[str]:
        return [name for name, term in self.config.items() if term.enabled]

    def get_command(self, name: str) -> torch.Tensor:
        if name not in self.config.terms or not self.config.terms[name].enabled:
            raise KeyError(f"Command term {name!r} is not active")
        if self.command is None:
            raise RuntimeError("Command manager must be reset before reading commands")
        selection = self._slices()[name]
        return self.command[selection] if self.command.ndim == 1 else self.command[:, selection]

    def get_term(self, name: str) -> Any:
        if name not in self.config.terms:
            raise KeyError(f"Command term {name!r} is not configured")
        if name in self._resolved_terms:
            return self._resolved_terms[name]
        return CommandTermView(name, self)

    def get_term_cfg(self, name: str) -> CommandTermCfg:
        try:
            return self.config.terms[name]
        except KeyError as exc:
            raise KeyError(f"Command term {name!r} is not configured") from exc

    def reset(self, env: Any, env_ids: torch.Tensor | slice | None = None) -> None:
        self._env = env
        self._validate_terms(env)
        slices = self._slices()
        num_envs = getattr(env, "num_envs", 1)
        ids = (
            self._ids(env_ids, num_envs, env.bundle.device)
            if env_ids is not None
            else torch.arange(num_envs, device=env.bundle.device)
        )
        partial = env_ids is not None and self.command is not None
        if partial:
            command = self.command.to(device=env.bundle.device, dtype=env.bundle.dtype).clone()
            if not self._fixed:
                for name, term in self.config.terms.items():
                    if not term.enabled:
                        continue
                    function = self._function(name, term, env)
                    self._reset_class_term(function, env, env_ids)
                    if not term.sample_on_reset:
                        continue
                    sampled = self._sample(env, name, term, env_ids=ids)
                    if command.ndim == 1:
                        command[slices[name]] = sampled
                    else:
                        command[ids, slices[name]] = sampled
        elif self.command is not None and self._fixed:
            command = self.command.to(device=env.bundle.device, dtype=env.bundle.dtype).clone()
            if num_envs > 1 and command.shape == (self.size,):
                command = command.unsqueeze(0).expand(num_envs, -1).clone()
            expected = (self.size,) if num_envs == 1 else (num_envs, self.size)
            if command.shape != expected:
                raise ValueError(
                    f"Expected fixed command shape {expected}, got {tuple(command.shape)}"
                )
        else:
            command = torch.zeros(
                (self.size,) if num_envs == 1 else (num_envs, self.size),
                dtype=env.bundle.dtype,
                device=env.bundle.device,
            )
            for name, term in self.config.terms.items():
                if not term.enabled:
                    continue
                function = self._function(name, term, env)
                self._reset_class_term(function, env, env_ids)
                if not term.sample_on_reset:
                    continue
                sampled = self._sample(env, name, term, env_ids=ids if partial else None)
                if command.ndim == 1:
                    command[slices[name]] = sampled
                else:
                    if partial:
                        command[ids, slices[name]] = sampled
                    else:
                        command[:, slices[name]] = sampled
        self.command = command
        if not partial:
            self._next_steps.clear()
        for name, term in self.config.terms.items():
            if term.enabled:
                self._schedule(env, name, term, env_ids if partial else None)

    @staticmethod
    def _ids(
        env_ids: torch.Tensor | slice,
        num_envs: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(num_envs, device=device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=device).reshape(-1)

    def step(self, env: Any) -> None:
        if self.command is None:
            raise RuntimeError("CommandManager must be reset before stepping")
        if self._fixed:
            return
        slices = self._slices()
        self.compute(
            env,
            dt=getattr(env.bundle, "timestep", 1.0) * getattr(env, "decimation", 1),
        )
        for name, term in self.config.terms.items():
            if not term.enabled:
                continue
            function = self._function(name, term, env)
            if callable(getattr(function, "compute", None)):
                continue
            if term.resampling_time_range is None:
                continue
            schedule = self._next_steps[name]
            if isinstance(schedule, torch.Tensor):
                due = schedule <= env.step_counts
                if not bool(due.any()):
                    continue
                sampled = self._sample(env, name, term, env_ids=due)
                self.command[due, slices[name]] = sampled
                self._schedule(env, name, term, due.nonzero(as_tuple=False).flatten())
            elif env.step_count >= schedule:
                sampled = self._sample(env, name, term)
                if self.command.ndim == 1:
                    self.command[slices[name]] = sampled
                else:
                    self.command[:, slices[name]] = sampled
                self._schedule(env, name, term)


__all__ = [
    "CommandManager",
    "CommandTermView",
    "body_pose_command",
    "head_pose_command",
    "velocity_command",
]
