"""Minimal functional environment for the first policy validation slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco_torch
import torch

from .model import MicroDuckModelBundle
from .observations import build_actor_observation, command_vector


@dataclass(frozen=True)
class EnvStep:
    observation: torch.Tensor
    reward: torch.Tensor
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class NominalMicroDuckEnv:
    """A small, deterministic policy-driven env around a `mujoco-torch` model.

    This class intentionally contains only the physics/observation boundary. It does not pretend
    to reproduce the upstream reward, termination, actuator-backlash, or delay implementations.
    """

    def __init__(
        self,
        bundle: MicroDuckModelBundle,
        *,
        command: torch.Tensor | None = None,
        action_scale: float = 1.0,
        decimation: int | None = None,
    ) -> None:
        self.bundle = bundle
        self.action_scale = action_scale
        self.decimation = decimation if decimation is not None else bundle.decimation
        if self.decimation < 1:
            raise ValueError("decimation must be positive")
        self.command = (
            command_vector(device=bundle.device)
            if command is None
            else torch.as_tensor(command, dtype=torch.float32, device=bundle.device)
        )
        if self.command.shape != (13,):
            raise ValueError(f"Expected a 13-element command, got {tuple(self.command.shape)}")
        self.data: Any | None = None
        self.last_action = torch.zeros(bundle.action_size, dtype=bundle.dtype, device=bundle.device)
        self.step_count = 0

    def reset(self, command: torch.Tensor | None = None) -> torch.Tensor:
        if command is not None:
            self.command = torch.as_tensor(command, dtype=torch.float32, device=self.bundle.device)
        if self.command.shape != (13,):
            raise ValueError(f"Expected a 13-element command, got {tuple(self.command.shape)}")
        self.data = self.bundle.new_data()
        self.last_action = torch.zeros_like(self.last_action)
        self.step_count = 0
        return self.observation()

    def observation(self) -> torch.Tensor:
        if self.data is None:
            raise RuntimeError("Call reset() before observation()")
        return build_actor_observation(self.bundle, self.data, self.last_action, self.command)

    def step(self, action: torch.Tensor) -> EnvStep:
        if self.data is None:
            self.reset()
        action = torch.as_tensor(action, dtype=self.bundle.dtype, device=self.bundle.device)
        if action.shape != (self.bundle.action_size,):
            raise ValueError(f"Expected action shape (14,), got {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            raise ValueError("Action contains non-finite values")
        target = self.bundle.default_pose + self.action_scale * action
        data = self.data
        if data is None:
            raise RuntimeError("Call reset() before step()")
        self.data = data.replace(ctrl=target)
        for _ in range(self.decimation):
            self.data = mujoco_torch.step(
                self.bundle.torch_model,
                self.data,
                fixed_iterations=self.bundle.fixed_iterations,
            )
        self.last_action = action
        self.step_count += 1
        observation = self.observation()
        finite = bool(
            torch.isfinite(self.data.qpos).all()
            and torch.isfinite(self.data.qvel).all()
            and torch.isfinite(observation).all()
        )
        return EnvStep(
            observation=observation,
            reward=torch.zeros((), dtype=torch.float32, device=self.bundle.device),
            terminated=not finite,
            truncated=False,
            info={"step": self.step_count, "time": float(self.data.time), "finite": finite},
        )

    def snapshot(self) -> dict[str, Any]:
        if self.data is None:
            raise RuntimeError("Call reset() before snapshot()")
        return {
            "qpos": self.data.qpos.detach().clone(),
            "qvel": self.data.qvel.detach().clone(),
            "sensordata": self.data.sensordata.detach().clone(),
            "time": float(self.data.time),
        }
