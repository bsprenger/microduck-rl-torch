"""Reward-term manager with raw-term and weighted-total separation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ..rewards import compute_velocity_reward_terms
from ..task_config import TermCollection

RewardEvaluator = Callable[[Any, dict[str, Any]], dict[str, torch.Tensor]]


def default_velocity_evaluator(env: Any, values: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Adapt the existing parity-tested velocity evaluator to the manager API."""

    return compute_velocity_reward_terms(
        env.bundle,
        env.data,
        command=env.command,
        action=values["action"],
        previous_action=values["previous_action"],
        previous_foot_positions=values["previous_foot_positions"],
        foot_air_time=values["foot_air_time"],
        foot_contact=values["foot_contact"],
        config=env.config.rewards,
        foot_touchdown=values["foot_touchdown"],
    )


@dataclass
class RewardManager:
    terms: TermCollection
    evaluator: RewardEvaluator = default_velocity_evaluator

    def compute(self, env: Any, **values: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.evaluator(env, values)
        missing = [name for name, term in self.terms.items() if term.enabled and name not in raw]
        if missing:
            raise RuntimeError(f"Configured reward terms are not produced: {missing!r}")
        weighted = [raw[name] * term.weight for name, term in self.terms.items() if term.enabled]
        if not weighted:
            reward = torch.zeros((), dtype=env.bundle.dtype, device=env.bundle.device)
        else:
            reward = torch.stack([value.reshape(()) for value in weighted]).sum()
        return reward.to(dtype=torch.float32), raw
