"""Command sampling manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import CommandConfig, sample_twist, sample_uniform


@dataclass
class CommandManager:
    config: CommandConfig

    def step(self, env: Any) -> None:
        if not env.domain_randomization or env._fixed_command or env.state is None:
            return
        if env.step_count >= env.state.next_twist_step:
            sampled = sample_twist(
                self.config,
                generator=env._generator,
                device=env.bundle.device,
                dtype=env.bundle.dtype,
            )
            env.command[:3] = sampled[:3]
            env.state.next_twist_step = env.step_count + env._next_interval_step(
                self.config.twist_resample_seconds
            )
        if env.step_count >= env.state.next_head_step:
            env.command[3:7] = sample_uniform(
                self.config.head_ranges,
                generator=env._generator,
                device=env.bundle.device,
                dtype=env.bundle.dtype,
            )
            env.state.next_head_step = env.step_count + env._next_interval_step(
                self.config.head_resample_seconds
            )
        if env.step_count >= env.state.next_body_step:
            env.command[7:13] = sample_uniform(
                self.config.body_ranges,
                generator=env._generator,
                device=env.bundle.device,
                dtype=env.bundle.dtype,
            )
            env.state.next_body_step = env.step_count + env._next_interval_step(
                self.config.body_resample_seconds
            )
