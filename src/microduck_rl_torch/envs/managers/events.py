"""Upstream-shaped startup, reset, interval, and step event lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..task_config import EventTermCfg, TermCollection
from .base import call_term, resolve_term

EventStage = Literal["reset", "step", "interval", "pre_physics", "post_physics"]
EventMode = Literal["startup", "reset", "interval", "step", "pre_physics", "post_physics"]


@dataclass
class EventManager:
    terms: TermCollection

    def __post_init__(self) -> None:
        self._resolved_terms: dict[str, Any] = {}

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
        if isinstance(term, EventTermCfg):
            mode = term.mode
        else:
            params = getattr(term, "params", {})
            mode = params.get("mode", params.get("stage", "pre_physics"))
        allowed = {"startup", "reset", "interval", "step", "pre_physics", "post_physics"}
        if mode not in allowed:
            raise ValueError(f"Unsupported event mode {mode!r}")
        return mode  # type: ignore[return-value]

    def _interval(self, term: Any) -> tuple[float, float] | None:
        if isinstance(term, EventTermCfg):
            return term.interval_range_s
        return getattr(term, "params", {}).get("interval_range_s")

    def _apply_term(
        self,
        env: Any,
        name: str,
        term: Any,
        *,
        env_ids: torch.Tensor | slice | None | object = ...,
    ) -> None:
        function = self._function(name, term, env)
        if function is None:
            raise RuntimeError("Enabled event term has no function")
        params = dict(getattr(term, "params", {}))
        # The legacy stage key is still understood while task configs migrate
        # to EventTermCfg.mode. It controls dispatch and is not a callback arg.
        params.pop("stage", None)
        params.pop("mode", None)
        params.pop("interval_range_s", None)
        call_term(function, env, params, env_ids=env_ids)

    def startup(self, env: Any) -> None:
        """Apply construction-time events after all managers exist."""

        for name, term in self.terms.items():
            if term.enabled and self._active(term, env) and self._mode(term) == "startup":
                self._apply_term(env, name, term, env_ids=None)

    def reset(self, env: Any, env_ids: torch.Tensor | slice | None = None) -> None:
        """Apply reset events and initialize interval schedules."""

        num_envs = getattr(env, "num_envs", 1)
        if env_ids is None:
            env_ids = torch.arange(num_envs, device=self._device(env))
        next_steps: dict[str, Any] = {}
        for name, term in self.terms.items():
            if not term.enabled or not self._active(term, env):
                continue
            mode = self._mode(term)
            if mode == "reset":
                self._apply_term(env, name, term, env_ids=env_ids)
            elif mode == "interval":
                interval = self._interval(term)
                if interval is None:
                    raise ValueError(f"Interval event {name!r} needs interval_range_s")
                sampled = self._next_interval_steps(env, interval, env_ids=env_ids)
                if num_envs == 1:
                    next_steps[name] = sampled
                else:
                    prior = env.state.manager_data.get("event_next_steps", {}).get(name)
                    if isinstance(prior, torch.Tensor) and prior.shape == (num_envs,):
                        schedule = prior.clone()
                        ids = self._ids(env_ids, num_envs)
                        schedule[ids] = sampled
                        next_steps[name] = schedule
                    else:
                        next_steps[name] = sampled
        if env.state is None:
            raise RuntimeError("Environment state must exist before resetting events")
        env.state.manager_data["event_next_steps"] = next_steps

    def apply(self, env: Any, stage: EventStage) -> None:
        next_steps = env.state.manager_data.setdefault("event_next_steps", {})
        num_envs = getattr(env, "num_envs", 1)
        for name, term in self.terms.items():
            if not term.enabled or not self._active(term, env):
                continue
            mode = self._mode(term)
            if stage == "pre_physics":
                if mode == "pre_physics":
                    self._apply_term(env, name, term)
            elif stage == "step":
                if mode == "step":
                    self._apply_term(env, name, term, env_ids=None)
            elif stage == "interval":
                if mode == "interval":
                    if name not in next_steps:
                        raise RuntimeError(f"Interval event {name!r} was not initialized")
                    schedule = next_steps[name]
                    if isinstance(schedule, torch.Tensor):
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
                    if env_ids is not None or num_envs == 1:
                        self._apply_term(env, name, term, env_ids=env_ids)
                        interval = self._interval(term)
                        if interval is None:
                            raise ValueError(f"Interval event {name!r} needs interval_range_s")
                        if isinstance(schedule, torch.Tensor):
                            schedule = schedule.clone()
                            schedule[env_ids] = self._next_interval_steps(
                                env, interval, env_ids=env_ids
                            )
                            next_steps[name] = schedule
                        else:
                            next_steps[name] = self._next_interval_steps(env, interval)
            elif stage == "post_physics" and mode == "post_physics":
                self._apply_term(env, name, term, env_ids=None)

    def _next_interval_steps(
        self,
        env: Any,
        interval: tuple[float, float],
        *,
        env_ids: torch.Tensor | slice | None = None,
    ) -> Any:
        if not hasattr(env, "_sample_range"):
            return int(env._next_interval_step(interval)) + int(getattr(env, "step_count", 0))
        sampled = env._sample_range(*interval, env_ids=env_ids)
        steps = (
            torch.round(
                torch.as_tensor(sampled, device=env.bundle.device)
                / (env.bundle.timestep * env.decimation)
            )
            .clamp_min(1)
            .to(torch.long)
        )
        num_envs = getattr(env, "num_envs", 1)
        if num_envs == 1:
            return int(steps.item()) + int(env.step_counts[0].item())
        return steps + (env.step_counts if env_ids is None else env.step_counts[env_ids])

    @staticmethod
    def _ids(env_ids: torch.Tensor | slice, num_envs: int) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(num_envs)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)
