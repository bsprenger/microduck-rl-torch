"""Termination and truncation manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..task_config import TermCollection
from .base import call_term, reset_term, resolve_term


def bad_orientation(env: Any) -> torch.Tensor:
    quaternion = env.data.xquat[..., env.bundle.root_body_id, :]
    cos_tilt = 1.0 - 2.0 * (quaternion[..., 1] ** 2 + quaternion[..., 2] ** 2)
    limit = torch.cos(
        torch.deg2rad(
            torch.tensor(
                env.config.bad_orientation_degrees,
                dtype=quaternion.dtype,
                device=quaternion.device,
            )
        )
    )
    return cos_tilt < limit


def timeout(env: Any) -> torch.Tensor:
    num_envs = getattr(env, "num_envs", 1)
    if num_envs > 1:
        step_counts = getattr(
            env,
            "step_counts",
            torch.tensor([env.step_count], dtype=torch.long, device=env.bundle.device),
        )
        return step_counts >= env.config.episode_length_steps
    return torch.as_tensor(env.step_count >= env.config.episode_length_steps)


@dataclass
class TerminationManager:
    terms: TermCollection

    def __post_init__(self) -> None:
        self._resolved_terms: dict[str, Any] = {}

    def reset(self, env: Any, env_ids: torch.Tensor | None = None) -> None:
        for name, term in self.terms.items():
            if term.enabled and (
                term.func is not None or getattr(term, "class_type", None) is not None
            ):
                function = self._resolved_terms.get(name)
                if function is None:
                    function = resolve_term(term, env)
                    self._resolved_terms[name] = function
                reset_term(function, env_ids)

    def evaluate(self, env: Any, *, finite: bool | torch.Tensor) -> tuple[Any, Any, dict[str, Any]]:
        values: dict[str, Any] = {}
        for name, term in self.terms.items():
            if not term.enabled:
                continue
            if name == "non_finite":
                values[name] = ~finite if isinstance(finite, torch.Tensor) else not finite
            elif term.func is not None or getattr(term, "class_type", None) is not None:
                if name not in self._resolved_terms:
                    self._resolved_terms[name] = resolve_term(term, env)
                value = call_term(self._resolved_terms[name], env, term.params)
                value = torch.as_tensor(value, dtype=torch.bool, device=env.bundle.device)
                num_envs = getattr(env, "num_envs", 1)
                if num_envs > 1:
                    if value.ndim == 0:
                        value = value.expand(num_envs)
                    if value.shape != (num_envs,):
                        raise ValueError(
                            f"Termination term {name!r} returned {tuple(value.shape)}; "
                            f"expected ({num_envs},)"
                        )
                    values[name] = value
                else:
                    values[name] = bool(value.item())
            else:
                raise RuntimeError(f"Termination term {name!r} has no function")
        timeout_names = {
            name
            for name, term in self.terms.items()
            if getattr(term, "time_out", False) or name in {"timeout", "time_out"}
        }
        num_envs = getattr(env, "num_envs", 1)
        if num_envs > 1:
            terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.bundle.device)
            truncated = torch.zeros_like(terminated)
            for name, value in values.items():
                if name in timeout_names:
                    truncated |= torch.as_tensor(value, device=terminated.device)
                else:
                    terminated |= torch.as_tensor(value, device=terminated.device)
        else:
            terminated = any(value for name, value in values.items() if name not in timeout_names)
            truncated = any(value for name, value in values.items() if name in timeout_names)
        return terminated, truncated, values
