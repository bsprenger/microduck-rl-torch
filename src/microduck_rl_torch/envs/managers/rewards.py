"""Reward-term manager with raw-term and weighted-total separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..rewards import velocity_term
from ..task_config import TermCollection
from .base import call_term, reset_term, resolve_term


@dataclass
class RewardManager:
    terms: TermCollection
    scale_by_dt: bool = False

    def __post_init__(self) -> None:
        self._resolved_terms: dict[str, Any] = {}

    def reset(self, env: Any, env_ids: torch.Tensor | None = None) -> None:
        for name, term in self.terms.items():
            if term.enabled and term.weight != 0.0:
                function = self._resolved_terms.get(name)
                if function is None:
                    function = resolve_term(term, env)
                    self._resolved_terms[name] = function
                reset_term(function, env_ids)

    def compute(self, env: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw: dict[str, torch.Tensor] = {}
        for name, term in self.terms.items():
            if not term.enabled:
                continue
            if term.weight == 0.0:
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
        missing = [
            name
            for name, term in self.terms.items()
            if term.enabled and term.weight != 0.0 and name not in raw
        ]
        if missing:
            raise RuntimeError(f"Configured reward terms are not produced: {missing!r}")
        weighted = [
            raw[name] * term.weight
            for name, term in self.terms.items()
            if term.enabled and term.weight != 0.0
        ]
        if not weighted:
            reward = torch.zeros((), dtype=env.bundle.dtype, device=env.bundle.device)
        else:
            reward = torch.stack([value.reshape(()) for value in weighted]).sum()
        if self.scale_by_dt:
            reward = reward * (env.bundle.timestep * env.decimation)
        return reward.to(dtype=torch.float32), raw


__all__ = ["RewardManager", "velocity_term"]
