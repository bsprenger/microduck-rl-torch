"""Reward-term manager with raw-term and weighted-total separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..task_config import TermCollection
from .base import call_term, reset_term, resolve_term


@dataclass
class RewardManager:
    terms: TermCollection
    scale_by_dt: bool = False

    def __post_init__(self) -> None:
        self._resolved_terms: dict[str, Any] = {}
        self._transition_cache: dict[str, Any] = {}

    def reset(self, env: Any, env_ids: torch.Tensor | None = None) -> None:
        self._transition_cache.clear()
        for name, term in self.terms.items():
            if term.enabled and (term.weight != 0.0 or term.execute_when_zero_weight):
                function = self._resolved_terms.get(name)
                if function is None:
                    function = resolve_term(term, env)
                    self._resolved_terms[name] = function
                reset_term(function, env_ids)

    def compute(self, env: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # A reward manager cache is scoped to exactly one transition.  Terms
        # may share expensive derived features without putting task-specific
        # state on the lifecycle owner.
        self._transition_cache.clear()
        raw: dict[str, torch.Tensor] = {}
        for name, term in self.terms.items():
            if not term.enabled:
                continue
            if term.weight == 0.0 and not term.execute_when_zero_weight:
                continue
            if name not in self._resolved_terms:
                self._resolved_terms[name] = resolve_term(term, env)
            function = self._resolved_terms[name]
            if function is None:
                raise RuntimeError(f"Reward term {name!r} has no function")
            raw[name] = torch.as_tensor(
                call_term(function, env, term.params),
                dtype=env.bundle.dtype,
                device=env.bundle.device,
            )
            num_envs = getattr(env, "num_envs", 1)
            if raw[name].ndim == 0 and num_envs > 1:
                raw[name] = raw[name].expand(num_envs)
            if raw[name].shape != (() if num_envs == 1 else (num_envs,)):
                raise ValueError(
                    f"Reward term {name!r} returned {tuple(raw[name].shape)}; "
                    f"expected {'scalar' if num_envs == 1 else f'({num_envs},)'}"
                )
        missing = [
            name
            for name, term in self.terms.items()
            if term.enabled
            and (term.weight != 0.0 or term.execute_when_zero_weight)
            and name not in raw
        ]
        if missing:
            raise RuntimeError(f"Configured reward terms are not produced: {missing!r}")
        weighted = [
            raw[name] * term.weight
            for name, term in self.terms.items()
            if term.enabled and term.weight != 0.0
        ]
        if not weighted:
            shape = (env.num_envs,) if getattr(env, "num_envs", 1) > 1 else ()
            reward = torch.zeros(shape, dtype=env.bundle.dtype, device=env.bundle.device)
        else:
            reward = torch.stack(weighted, dim=0).sum(dim=0)
        if self.scale_by_dt:
            reward = reward * (env.bundle.timestep * env.decimation)
        return reward.to(dtype=env.bundle.dtype), raw

    def cached(self, key: str, factory: Any) -> Any:
        """Return one transition-local derived value for a reward term."""

        if key not in self._transition_cache:
            self._transition_cache[key] = factory()
        return self._transition_cache[key]


__all__ = ["RewardManager"]
