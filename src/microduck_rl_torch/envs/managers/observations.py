"""Observation-group manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..task_config import ObservationGroupCfg, ObservationGroupsCfg, ObservationTermCfg
from .base import call_term, reset_term, resolve_term


@dataclass
class ObservationManager:
    config: ObservationGroupsCfg

    def __post_init__(self) -> None:
        self._resolved_terms: dict[tuple[str, str], Any] = {}

    def _term_function(self, group: str, name: str, term: Any, env: Any) -> Any:
        key = (group, name)
        if key not in self._resolved_terms:
            self._resolved_terms[key] = resolve_term(term, env)
        return self._resolved_terms[key]

    def reset(self, env: Any, env_ids: torch.Tensor | None = None) -> None:
        for group_name, group_cfg in self.config.groups.items():
            for name, term in group_cfg.terms.items():
                if term.enabled:
                    reset_term(self._term_function(group_name, name, term, env), env_ids)

    def compute(self, env: Any, group: str = "actor") -> torch.Tensor:
        if group not in self.config.groups:
            raise KeyError(f"Observation group {group!r} is not configured")
        group_cfg = self.config.groups[group]
        if not isinstance(group_cfg, ObservationGroupCfg):
            raise TypeError(f"Observation group {group!r} is not an ObservationGroupCfg")
        if not group_cfg.enabled:
            raise RuntimeError(f"Observation group {group!r} is disabled")
        values: list[torch.Tensor] = []
        for name, term in group_cfg.terms.items():
            if not isinstance(term, ObservationTermCfg):
                raise TypeError(f"Observation term {name!r} is not an ObservationTermCfg")
            if not term.enabled:
                continue
            function = self._term_function(group, name, term, env)
            if function is None:
                raise RuntimeError(f"Observation term {name!r} has no function")
            value = torch.as_tensor(
                call_term(function, env, term.params),
                dtype=env.bundle.dtype,
                device=env.bundle.device,
            )
            if getattr(env, "num_envs", 1) > 1:
                # Every batched observation term has the canonical shape
                # ``(num_envs, features)``.  This accepts scalar constants,
                # one shared feature vector, and already-batched terms from
                # custom task code without making the manager inspect the
                # term's implementation.
                num_envs = env.num_envs
                if value.ndim == 0:
                    value = value.expand(num_envs).unsqueeze(-1)
                elif value.ndim == 1:
                    if value.shape[0] == num_envs:
                        value = value.unsqueeze(-1)
                    else:
                        value = value.unsqueeze(0).expand(num_envs, -1)
                elif value.ndim != 2 or value.shape[0] != num_envs:
                    raise ValueError(
                        f"Observation term {name!r} returned {tuple(value.shape)}; "
                        f"expected a vector or ({num_envs}, features)"
                    )
            elif value.ndim == 0:
                value = value.reshape(1)
            value = value * term.scale
            if term.noise is not None:
                value = value + term.noise(env, value, **term.noise_params)
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Observation term {name!r} returned non-finite values")
            values.append(value)
        if not values:
            raise RuntimeError(f"Observation group {group!r} has no enabled terms")
        observation = torch.cat(values, dim=-1).to(dtype=torch.float32)
        if group_cfg.expected_size is not None and observation.shape[-1] != group_cfg.expected_size:
            raise RuntimeError(
                f"Observation group {group!r} returned {observation.shape[-1]} values; "
                f"expected {group_cfg.expected_size}"
            )
        return observation
