"""Observation-group manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..task_config import ObservationGroupsCfg
from .base import TaskRuntimeContext


@dataclass
class ObservationManager:
    config: ObservationGroupsCfg

    def compute(self, env: Any, group: str = "actor") -> torch.Tensor:
        if group not in self.config.groups:
            raise KeyError(f"Observation group {group!r} is not configured")
        group_cfg = self.config.groups[group]
        if not group_cfg.enabled or group_cfg.builder is None:
            raise RuntimeError(f"Observation group {group!r} has no enabled builder")
        observation = group_cfg.builder(TaskRuntimeContext(env))
        if group_cfg.expected_size is not None and observation.shape[-1] != group_cfg.expected_size:
            raise RuntimeError(
                f"Observation group {group!r} returned {observation.shape[-1]} values; "
                f"expected {group_cfg.expected_size}"
            )
        return observation
