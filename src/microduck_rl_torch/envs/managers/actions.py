"""Action scaling, validation, and delay management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..task_config import ActionCfg


@dataclass
class ActionManager:
    config: ActionCfg

    def prepare(self, env: Any, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if env.data is None or env.state is None:
            raise RuntimeError("Call reset() before preparing an action")
        action = torch.as_tensor(action, dtype=env.bundle.dtype, device=env.bundle.device)
        if action.shape != (self.config.size,):
            raise ValueError(
                f"Expected action shape ({self.config.size},), got {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all():
            raise ValueError("Action contains non-finite values")
        sensor = env.state.sensors
        sensor.previous_joint_velocity = env._encoder_velocity().clone()
        sensor.previous_action = sensor.last_action.clone()
        sensor.delay_buffer[env.step_count % len(sensor.delay_buffer)] = action.clone()
        delayed_index = (env.step_count - sensor.delay_lag) % len(sensor.delay_buffer)
        applied_action = sensor.delay_buffer[delayed_index]
        target = env.bundle.default_pose + self.config.scale * applied_action
        return applied_action, target
