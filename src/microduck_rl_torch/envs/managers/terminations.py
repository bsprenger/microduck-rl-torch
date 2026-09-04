"""Termination and truncation manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..task_config import TermCollection
from .base import TaskRuntimeContext


def bad_orientation(ctx: TaskRuntimeContext) -> torch.Tensor:
    env = ctx.env
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


def timeout(ctx: TaskRuntimeContext) -> torch.Tensor:
    return torch.as_tensor(ctx.env.step_count >= ctx.env.config.episode_length_steps)


@dataclass
class TerminationManager:
    terms: TermCollection

    def evaluate(self, env: Any, *, finite: bool) -> tuple[bool, bool, dict[str, bool]]:
        values: dict[str, bool] = {}
        for name, term in self.terms.items():
            if not term.enabled:
                continue
            if name == "non_finite":
                values[name] = not finite
            elif term.func is not None:
                value = term.func(TaskRuntimeContext(env))
                values[name] = (
                    bool(value.item()) if isinstance(value, torch.Tensor) else bool(value)
                )
            else:
                raise RuntimeError(f"Termination term {name!r} has no function")
        terminated = any(
            value for name, value in values.items() if name not in {"timeout", "time_out"}
        )
        truncated = bool(values.get("timeout", values.get("time_out", False)))
        return terminated, truncated, values
