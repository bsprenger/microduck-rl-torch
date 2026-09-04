"""Startup, reset, interval, and step event lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..task_config import EventTermCfg, TermCollection
from .base import call_event_term, resolve_term

EventStage = Literal["reset", "step", "interval", "pre_physics", "post_physics"]
EventMode = Literal["startup", "reset", "interval", "step", "pre_physics", "post_physics"]


@dataclass
class EventManager:
    terms: TermCollection

    def __post_init__(self) -> None:
        self._resolved_terms: dict[str, Any] = {}
        self._global_next_steps: dict[str, int] = {}

    @staticmethod
    def _global_step(env: Any) -> int:
        if hasattr(env, "common_step_counter"):
            return int(env.common_step_counter)
        step_counts = getattr(env, "step_counts", None)
        if isinstance(step_counts, torch.Tensor) and step_counts.numel():
            return int(step_counts.max().item())
        return int(getattr(env, "step_count", 0))

    def _function(self, name: str, term: Any, env: Any) -> Any:
        if name not in self._resolved_terms:
            self._resolved_terms[name] = resolve_term(term, env)
        return self._resolved_terms[name]

    @staticmethod
    def _device(env: Any) -> torch.device:
        bundle = getattr(env, "bundle", None)
        return getattr(bundle, "device", torch.device("cpu"))

    @staticmethod
    def _active(term: Any, env: Any) -> bool:
        return not (
            isinstance(term, EventTermCfg)
            and term.requires_domain_randomization
            and not getattr(env, "domain_randomization", True)
        )

    def _mode(self, term: Any) -> EventMode:
        if not isinstance(term, EventTermCfg):
            raise TypeError("EventManager terms must be EventTermCfg instances")
        mode = term.mode
        allowed = {"startup", "reset", "interval", "step", "pre_physics", "post_physics"}
        if mode not in allowed:
            raise ValueError(f"Unsupported event mode {mode!r}")
        return mode  # type: ignore[return-value]

    def _interval(self, term: Any) -> tuple[float, float] | None:
        if not isinstance(term, EventTermCfg):
            raise TypeError("EventManager terms must be EventTermCfg instances")
        return term.interval_range_s

    def _apply_term(
        self,
        env: Any,
        name: str,
        term: Any,
        *,
        env_ids: torch.Tensor | slice | None = None,
    ) -> bool:
        function = self._function(name, term, env)
        if function is None:
            raise RuntimeError("Enabled event term has no function")
        params = dict(getattr(term, "params", {}))
        call_event_term(
            function,
            env,
            env_ids,
            params,
        )
        return bool(getattr(term, "mutates_physics", False))

    def startup(self, env: Any) -> bool:
        """Apply construction-time events after all managers exist.

        Model-only startup mutations do not have a data object to refresh yet;
        state-writing startup callbacks are refreshed when the first reset
        creates data, while normal runtime stages use the immediate barrier.
        """

        mutates_physics = False
        for name, term in self.terms.items():
            if term.enabled and self._active(term, env) and self._mode(term) == "startup":
                mutates_physics |= self._apply_term(env, name, term, env_ids=None)
        if mutates_physics and getattr(env, "data", None) is not None:
            env.physics.forward()
        return mutates_physics

    def apply_reset(self, env: Any, env_ids: torch.Tensor | slice | None = None) -> None:
        """Apply mode=reset callbacks at the scene-reset boundary."""

        if env.state is None:
            raise RuntimeError("Environment state must exist before resetting events")
        num_envs = getattr(env, "num_envs", 1)
        device = self._device(env)
        selected = (
            torch.arange(num_envs, device=device, dtype=torch.long)
            if env_ids is None
            else self._ids(env_ids, num_envs).to(device)
        )
        reset_state = env.state.manager_data.setdefault("event_reset_state", {})
        global_step = self._global_step(env)
        for name, term in self.terms.items():
            if term.enabled and self._active(term, env) and self._mode(term) == "reset":
                minimum = int(term.min_step_count_between_reset)
                if minimum < 0:
                    raise ValueError(
                        f"Reset event {name!r} min_step_count_between_reset must be non-negative"
                    )
                if minimum == 0:
                    eligible = selected
                else:
                    state = reset_state.setdefault(
                        name,
                        {
                            "step": torch.zeros(num_envs, dtype=torch.long, device=device),
                            "seen": torch.zeros(num_envs, dtype=torch.bool, device=device),
                        },
                    )
                    elapsed = global_step - state["step"][selected]
                    eligible = selected[(~state["seen"][selected]) | (elapsed >= minimum)]
                if not eligible.numel():
                    continue
                state = reset_state.setdefault(
                    name,
                    {
                        "step": torch.zeros(num_envs, dtype=torch.long, device=device),
                        "seen": torch.zeros(num_envs, dtype=torch.bool, device=device),
                    },
                )
                state["step"][eligible] = global_step
                state["seen"][eligible] = True
                callback_ids: torch.Tensor | slice | None = (
                    None if env_ids is None and minimum == 0 else eligible
                )
                self._apply_term(env, name, term, env_ids=callback_ids)

    def reset(
        self,
        env: Any,
        env_ids: torch.Tensor | slice | None = None,
        *,
        apply_terms: bool = True,
    ) -> None:
        """Reset manager-owned schedules, optionally applying reset events.

        ``apply_terms=False`` is used by the environment after manager state
        resets. The callback runs immediately after scene/physics reset because
        later terms may depend on its writes.
        """

        num_envs = getattr(env, "num_envs", 1)
        reset_ids = torch.arange(num_envs, device=self._device(env)) if env_ids is None else env_ids
        if apply_terms:
            self.apply_reset(env, env_ids)
        next_steps: dict[str, Any] = {}
        for name, term in self.terms.items():
            if not term.enabled or not self._active(term, env):
                continue
            mode = self._mode(term)
            if mode == "interval":
                interval = self._interval(term)
                if interval is None:
                    raise ValueError(f"Interval event {name!r} needs interval_range_s")
                prior = env.state.manager_data.get("event_next_steps", {}).get(name)
                if term.is_global_time:
                    # A global timer is wall-clock state shared by all rows;
                    # partial episode resets must not restart it.
                    prior = self._global_next_steps.get(name, prior)
                    sampled = (
                        prior
                        if prior is not None
                        else self._next_interval_steps(env, interval, global_time=True)
                    )
                    self._global_next_steps[name] = int(sampled)
                    next_steps[name] = sampled
                else:
                    sampled = self._next_interval_steps(env, interval, env_ids=reset_ids)
                    if num_envs == 1:
                        next_steps[name] = sampled
                    else:
                        if isinstance(prior, torch.Tensor) and prior.shape == (num_envs,):
                            schedule = prior.clone()
                            ids = self._ids(reset_ids, num_envs)
                            schedule[ids] = sampled
                            next_steps[name] = schedule
                        else:
                            next_steps[name] = sampled
        if env.state is None:
            raise RuntimeError("Environment state must exist before resetting events")
        env.state.manager_data["event_next_steps"] = next_steps

    def apply(self, env: Any, stage: EventStage) -> bool:
        """Apply one runtime stage and enforce a post-write forward barrier.

        Event callbacks are allowed to write simulation state, but that fact
        must be explicit in their configuration.  The barrier is deliberately
        manager-owned so task authors do not need to sprinkle backend-specific
        ``forward()`` calls through otherwise portable event functions.
        """

        next_steps = env.state.manager_data.setdefault("event_next_steps", {})
        num_envs = getattr(env, "num_envs", 1)
        mutates_physics = False
        for name, term in self.terms.items():
            if not term.enabled or not self._active(term, env):
                continue
            mode = self._mode(term)
            if stage == "pre_physics":
                if mode == "pre_physics":
                    mutates_physics |= self._apply_term(env, name, term, env_ids=None)
            elif stage == "step":
                if mode == "step":
                    mutates_physics |= self._apply_term(env, name, term, env_ids=None)
            elif stage == "interval":
                if mode == "interval":
                    if name not in next_steps:
                        raise RuntimeError(f"Interval event {name!r} was not initialized")
                    schedule = next_steps[name]
                    if term.is_global_time:
                        if self._global_step(env) < int(schedule):
                            continue
                        env_ids = None
                    elif isinstance(schedule, torch.Tensor):
                        step_counts = getattr(
                            env,
                            "step_counts",
                            torch.tensor(
                                [env.step_count], dtype=torch.long, device=schedule.device
                            ),
                        )
                        due = schedule <= step_counts
                        if not bool(due.any()):
                            continue
                        env_ids = due.nonzero(as_tuple=False).flatten()
                    elif env.step_count < schedule:
                        continue
                    else:
                        env_ids = None
                    if term.is_global_time or env_ids is not None or num_envs == 1:
                        mutates_physics |= self._apply_term(env, name, term, env_ids=env_ids)
                        interval = self._interval(term)
                        if interval is None:
                            raise ValueError(f"Interval event {name!r} needs interval_range_s")
                        if term.is_global_time:
                            next_steps[name] = self._next_interval_steps(
                                env, interval, global_time=True
                            )
                            self._global_next_steps[name] = int(next_steps[name])
                        elif isinstance(schedule, torch.Tensor):
                            schedule = schedule.clone()
                            schedule[env_ids] = self._next_interval_steps(
                                env, interval, env_ids=env_ids
                            )
                            next_steps[name] = schedule
                        else:
                            next_steps[name] = self._next_interval_steps(env, interval)
            elif stage == "post_physics" and mode == "post_physics":
                mutates_physics |= self._apply_term(env, name, term, env_ids=None)
        if mutates_physics:
            physics = getattr(env, "physics", None)
            forward = getattr(physics, "forward", None)
            if not callable(forward):
                raise RuntimeError(
                    f"Event stage {stage!r} contains a physics-mutating term but the "
                    "environment has no forward-capable physics backend"
                )
            forward()
        return mutates_physics

    def _next_interval_steps(
        self,
        env: Any,
        interval: tuple[float, float],
        *,
        env_ids: torch.Tensor | slice | None = None,
        global_time: bool = False,
    ) -> Any:
        if not hasattr(env, "_sample_range"):
            return int(env._next_interval_step(interval)) + int(getattr(env, "step_count", 0))
        sampled = env._sample_range(*interval, env_ids=env_ids)
        steps = (
            torch.ceil(
                torch.as_tensor(sampled, device=env.bundle.device)
                / (env.bundle.timestep * env.decimation)
            )
            .clamp_min(1)
            .to(torch.long)
        )
        num_envs = getattr(env, "num_envs", 1)
        if global_time:
            return int(steps.reshape(-1)[0].item()) + self._global_step(env)
        if num_envs == 1:
            return int(steps.item()) + int(env.step_counts[0].item())
        return steps + (env.step_counts if env_ids is None else env.step_counts[env_ids])

    @staticmethod
    def _ids(env_ids: torch.Tensor | slice, num_envs: int) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(num_envs)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)
