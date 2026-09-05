"""Upstream-shaped command-term manager."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import torch

from ..config import CommandConfig, CommandTermCfg, sample_twist, sample_uniform
from .base import call_term, resolve_term


@dataclass(frozen=True)
class CommandTermView:
    """Live scalar-environment view matching upstream ``get_term`` access."""

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
) -> torch.Tensor:
    """Sample the current task's three-element velocity command."""

    return sample_twist(
        getattr(env.config, "command", None),
        generator=env._generator,
        device=env.bundle.device,
        dtype=env.bundle.dtype,
        twist_ranges=twist_ranges,
        turn_in_place_fraction=turn_in_place_fraction,
        standing_fraction=standing_fraction,
    )


def head_pose_command(
    env: Any,
    *,
    ranges: tuple[tuple[float, float], ...] | None = None,
) -> torch.Tensor:
    """Sample the current task's four-element head-pose command."""

    return sample_uniform(
        ranges or env.config.command.head_ranges,
        generator=env._generator,
        device=env.bundle.device,
        dtype=env.bundle.dtype,
    )


def body_pose_command(
    env: Any,
    *,
    ranges: tuple[tuple[float, float], ...] | None = None,
) -> torch.Tensor:
    """Sample the current task's six-element body-pose command."""

    return sample_uniform(
        ranges or env.config.command.body_ranges,
        generator=env._generator,
        device=env.bundle.device,
        dtype=env.bundle.dtype,
    )


@dataclass
class CommandManager:
    """Concatenate named command terms and resample them independently.

    Unlike the previous velocity-only implementation, this manager has no
    knowledge of command slices. A task defines ordered terms with sizes and
    sampling functions; posture, phase, and prop tasks can therefore replace
    the velocity command without changing the environment lifecycle.
    """

    config: CommandConfig
    command: torch.Tensor | None = None
    _fixed: bool = False
    _next_steps: dict[str, int] = field(default_factory=dict)
    _resolved_terms: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.command is not None:
            self.command = torch.as_tensor(self.command).clone()
            self._fixed = True

    @property
    def fixed(self) -> bool:
        return self._fixed

    @property
    def size(self) -> int:
        return sum(term.size for term in self.config.terms.values() if term.enabled)

    def _validate_terms(self) -> None:
        for name, term in self.config.terms.items():
            if not isinstance(term, CommandTermCfg):
                raise TypeError(f"Command term {name!r} is not a CommandTermCfg")
            if term.size < 1:
                raise ValueError(f"Command term {name!r} must have a positive size")
            if term.enabled and term.func is None and term.class_type is None:
                raise ValueError(f"Enabled command term {name!r} has no sampling function")
            if term.resample_interval_s is not None:
                low, high = term.resample_interval_s
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
            slices[name] = slice(offset, offset + term.size)
            offset += term.size
        return slices

    def _function(self, name: str, term: CommandTermCfg, env: Any) -> Any:
        if name not in self._resolved_terms:
            self._resolved_terms[name] = resolve_term(term, env)
        return self._resolved_terms[name]

    def _class_command(self, function: Any, env: Any, *, size: int) -> torch.Tensor | None:
        value = getattr(function, "command", None)
        if value is None:
            return None
        value = torch.as_tensor(value, dtype=env.bundle.dtype, device=env.bundle.device)
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.shape != (size,):
            raise ValueError(
                f"Command term returned {tuple(value.shape)} values; expected ({size},)"
            )
        if not torch.isfinite(value).all():
            raise ValueError("Command term returned non-finite values")
        return value

    def _sample(self, env: Any, name: str, term: CommandTermCfg) -> torch.Tensor:
        function = self._function(name, term, env)
        class_command = self._class_command(function, env, size=term.size)
        if class_command is not None:
            return class_command
        if function is None:
            raise RuntimeError("Cannot sample a command term without a function")
        value = torch.as_tensor(
            call_term(function, env, term.params), dtype=env.bundle.dtype, device=env.bundle.device
        )
        if value.shape != (term.size,):
            raise ValueError(
                f"Command term returned {tuple(value.shape)} values; expected ({term.size},)"
            )
        if not torch.isfinite(value).all():
            raise ValueError("Command term returned non-finite values")
        return value

    def _reset_class_term(self, function: Any, env: Any) -> None:
        reset = getattr(function, "reset", None)
        if not callable(reset):
            return
        env_ids = torch.zeros(1, dtype=torch.long, device=env.bundle.device)
        signature = inspect.signature(reset)
        if "env_ids" in signature.parameters:
            reset(env_ids=env_ids)
        elif signature.parameters:
            reset(env_ids)
        else:
            reset()

    def _compute_class_term(self, function: Any, env: Any) -> None:
        compute = getattr(function, "compute", None)
        if not callable(compute):
            return
        compute(env.bundle.timestep * env.decimation)

    def _schedule(self, env: Any, name: str, term: CommandTermCfg) -> None:
        if term.resample_interval_s is None:
            self._next_steps.pop(name, None)
        else:
            self._next_steps[name] = env.step_count + env._next_interval_step(
                term.resample_interval_s
            )

    def set_command(self, command: torch.Tensor) -> None:
        if self.command is None:
            value = torch.as_tensor(command)
        else:
            value = torch.as_tensor(command, dtype=self.command.dtype, device=self.command.device)
        if value.shape != (self.size,):
            raise ValueError(f"Expected a {self.size}-element command, got {tuple(value.shape)}")
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
        return self.command[self._slices()[name]]

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

    def reset(self, env: Any) -> None:
        self._validate_terms()
        slices = self._slices()
        if self.command is not None and self._fixed:
            command = self.command.to(device=env.bundle.device, dtype=env.bundle.dtype).clone()
        else:
            command = torch.zeros(self.size, dtype=env.bundle.dtype, device=env.bundle.device)
            for name, term in self.config.terms.items():
                if not term.enabled or not term.sample_on_reset:
                    continue
                function = self._function(name, term, env)
                self._reset_class_term(function, env)
                command[slices[name]] = self._sample(env, name, term)
        self.command = command
        self._next_steps.clear()
        for name, term in self.config.terms.items():
            if term.enabled:
                self._schedule(env, name, term)

    def step(self, env: Any) -> None:
        if self.command is None:
            raise RuntimeError("CommandManager must be reset before stepping")
        if self._fixed:
            return
        slices = self._slices()
        for name, term in self.config.terms.items():
            if not term.enabled:
                continue
            function = self._function(name, term, env)
            self._compute_class_term(function, env)
            if term.resample_interval_s is None or hasattr(function, "compute"):
                class_command = self._class_command(function, env, size=term.size)
                if class_command is not None:
                    self.command[slices[name]] = class_command
                continue
            if env.step_count >= self._next_steps[name]:
                self.command[slices[name]] = self._sample(env, name, term)
                self._schedule(env, name, term)


__all__ = [
    "CommandManager",
    "CommandTermView",
    "body_pose_command",
    "head_pose_command",
    "velocity_command",
]
