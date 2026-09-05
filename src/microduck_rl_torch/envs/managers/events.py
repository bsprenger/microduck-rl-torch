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

    def reset(self, env: Any) -> None:
        """Apply reset events and initialize interval schedules."""

        next_steps: dict[str, int] = {}
        for name, term in self.terms.items():
            if not term.enabled or not self._active(term, env):
                continue
            mode = self._mode(term)
            if mode == "reset":
                env_ids = torch.zeros(1, dtype=torch.long, device=self._device(env))
                self._apply_term(env, name, term, env_ids=env_ids)
            elif mode == "interval":
                interval = self._interval(term)
                if interval is None:
                    raise ValueError(f"Interval event {name!r} needs interval_range_s")
                next_steps[name] = env.step_count + env._next_interval_step(interval)
        if env.state is None:
            raise RuntimeError("Environment state must exist before resetting events")
        env.state.manager_data["event_next_steps"] = next_steps

    def apply(self, env: Any, stage: EventStage) -> None:
        next_steps = env.state.manager_data.setdefault("event_next_steps", {})
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
                    if env.step_count >= next_steps[name]:
                        self._apply_term(env, name, term, env_ids=None)
                        interval = self._interval(term)
                        if interval is None:
                            raise ValueError(f"Interval event {name!r} needs interval_range_s")
                        next_steps[name] = env.step_count + env._next_interval_step(interval)
            elif stage == "post_physics" and mode == "post_physics":
                self._apply_term(env, name, term, env_ids=None)
