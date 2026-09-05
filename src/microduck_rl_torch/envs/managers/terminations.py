"""Termination and truncation manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..task_config import TermCollection
from .base import call_term, reset_term, resolve_term


def bad_orientation(env: Any) -> torch.Tensor:
    quaternion = env.data.xquat[env.bundle.trunk_body_id]
    cos_tilt = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
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

    def evaluate(self, env: Any, *, finite: bool) -> tuple[bool, bool, dict[str, bool]]:
        values: dict[str, bool] = {}
        for name, term in self.terms.items():
            if not term.enabled:
                continue
            if name == "non_finite":
                values[name] = not finite
            elif term.func is not None or getattr(term, "class_type", None) is not None:
                if name not in self._resolved_terms:
                    self._resolved_terms[name] = resolve_term(term, env)
                value = call_term(self._resolved_terms[name], env, term.params)
                values[name] = (
                    bool(value.item()) if isinstance(value, torch.Tensor) else bool(value)
                )
            else:
                raise RuntimeError(f"Termination term {name!r} has no function")
        timeout_names = {
            name
            for name, term in self.terms.items()
            if getattr(term, "time_out", False) or name in {"timeout", "time_out"}
        }
        terminated = any(value for name, value in values.items() if name not in timeout_names)
        truncated = any(value for name, value in values.items() if name in timeout_names)
        return terminated, truncated, values
