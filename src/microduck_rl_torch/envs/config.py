"""Explicit configuration for the upstream MicroDuck velocity task."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CommandTermCfg:
    """One named command source with an optional resampling schedule."""

    func: Callable[..., torch.Tensor] | None = None
    class_type: type[Any] | None = None
    size: int = 0
    resample_interval_s: tuple[float, float] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    sample_on_reset: bool = True

    def clone(self) -> CommandTermCfg:
        return deepcopy(self)


@dataclass
class CommandConfig(MutableMapping[str, CommandTermCfg]):
    """Command term collection plus velocity-task sampling defaults.

    The ranges remain here because the first task's command functions use
    them. New tasks can ignore those fields and provide their own command
    terms, just as upstream command term configs replace the velocity command
    class for posture and prop tasks.
    """

    twist_ranges: tuple[tuple[float, float], ...] = ((-0.4, 0.4), (-0.3, 0.3), (-1.0, 1.0))
    head_ranges: tuple[tuple[float, float], ...] = (
        (-0.05, 0.05),
        (-0.05, 0.05),
        (-0.07, 0.07),
        (-0.015, 0.015),
    )
    body_ranges: tuple[tuple[float, float], ...] = (
        (-0.005, 0.005),
        (-0.005, 0.005),
        (-0.005, 0.005),
        (-0.05, 0.05),
        (-0.05, 0.05),
        (-0.05, 0.05),
    )
    twist_resample_seconds: tuple[float, float] = (10.0, 10.0)
    head_resample_seconds: tuple[float, float] = (2.0, 5.0)
    body_resample_seconds: tuple[float, float] = (2.0, 5.0)
    turn_in_place_fraction: float = 0.15
    standing_fraction: float = 0.02
    terms: OrderedDict[str, CommandTermCfg] = field(default_factory=OrderedDict)

    def __getitem__(self, name: str) -> CommandTermCfg:
        return self.terms[name]

    def __setitem__(self, name: str, term: CommandTermCfg) -> None:
        if not isinstance(term, CommandTermCfg):
            raise TypeError(f"Expected CommandTermCfg for {name!r}")
        self.terms[name] = term

    def __delitem__(self, name: str) -> None:
        del self.terms[name]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def add(self, name: str, term: CommandTermCfg) -> None:
        if name in self.terms:
            raise KeyError(f"Command term {name!r} already exists; use replace()")
        self[name] = term

    def replace(self, name: str, term: CommandTermCfg) -> None:
        if name not in self.terms:
            raise KeyError(f"Cannot replace missing command term {name!r}")
        self[name] = term

    def remove(self, name: str) -> None:
        del self[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.terms)


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Reset/interval randomization ranges used by upstream training."""

    com_range: float = 0.003
    head_com_range: float = 0.003
    mass_inertia_range: tuple[float, float] = (0.95, 1.05)
    joint_friction_range: tuple[float, float] = (0.9, 1.1)
    armature_range: tuple[float, float] = (0.9, 1.1)
    foot_friction_range: tuple[float, float] = (0.7, 1.3)
    velocity_push_interval: tuple[float, float] = (3.0, 6.0)
    velocity_push_range: tuple[float, float] = (-0.3, 0.3)
    imu_angle_degrees: float = 6.0
    encoder_bias_range: tuple[float, float] = (-0.015, 0.015)
    base_pitch_degrees: float = 10.0
    base_roll_degrees: float = 5.0


@dataclass(frozen=True)
class RewardConfig:
    """Main velocity-task reward weights and shaping constants."""

    pose: float = 1.0
    upright: float = 2.0
    track_linear_velocity: float = 2.0
    track_angular_velocity: float = 2.0
    air_time: float = 3.0
    head_pose_tracking: float = 2.0
    foot_slip: float = -0.1
    body_ang_vel: float = -0.05
    angular_momentum: float = -0.02
    action_rate_l2: float = -0.1
    foot_clearance: float = -2.0
    foot_swing_height: float = -0.25
    self_collisions: float = -1.0
    pose_standing_std: tuple[float, ...] = (0.1, 0.05, 0.15, 0.15, 0.1) * 2
    pose_walking_std: tuple[float, ...] = (0.3, 0.05, 0.4, 0.4, 0.25) * 2
    walking_threshold: float = 0.01
    velocity_std: float = 0.1**0.5
    angular_velocity_std: float = 0.5**0.5
    upright_std: float = 0.05**0.5
    air_time_range: tuple[float, float] = (0.125, 0.300)
    foot_target_height: float = 0.02


@dataclass(frozen=True)
class MicroDuckVelocityConfig:
    """Task-specific settings used by the velocity manager terms."""

    episode_length_steps: int = 1000
    command: CommandConfig = field(default_factory=CommandConfig)
    randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    actor_noise: tuple[float, float, float, float] = (0.03, 0.01, 0.001, 0.25)
    imu_delay_lag: tuple[int, int] = (0, 1)
    joint_velocity_delay_lag: int = 1
    delay_update_period: int = 64
    initial_height_range: tuple[float, float] = (0.12, 0.13)
    bad_orientation_degrees: float = 70.0
    use_projected_gravity: bool = True
    randomize_com: bool = True
    randomize_head_com: bool = True
    randomize_mass_inertia: bool = True
    randomize_joint_friction: bool = True
    randomize_foot_friction: bool = True
    randomize_armature: bool = True
    randomize_velocity_pushes: bool = True
    randomize_imu_orientation: bool = True
    randomize_encoder_bias: bool = True
    randomize_base_orientation: bool = False
    randomize_actuator_delay: bool = False


def sample_uniform(
    ranges: tuple[tuple[float, float], ...],
    *,
    generator: torch.Generator | None = None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample one vector without changing the caller's RNG state policy."""

    values = torch.empty(len(ranges), device=device, dtype=dtype)
    for index, (low, high) in enumerate(ranges):
        values[index] = (
            torch.rand((), generator=generator, device=device, dtype=dtype) * (high - low) + low
        )
    return values


def sample_twist(
    config: CommandConfig | None,
    *,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
    twist_ranges: tuple[tuple[float, float], ...] | None = None,
    turn_in_place_fraction: float | None = None,
    standing_fraction: float | None = None,
) -> torch.Tensor:
    """Sample only the 3D velocity command used by twist resampling."""

    if config is None and twist_ranges is None:
        raise ValueError("sample_twist needs a config or explicit twist_ranges")
    if twist_ranges is None:
        assert config is not None
        ranges = config.twist_ranges
    else:
        ranges = twist_ranges
    turn_fraction = (
        config.turn_in_place_fraction
        if turn_in_place_fraction is None and config is not None
        else (turn_in_place_fraction or 0.0)
    )
    standing_fraction_value = (
        config.standing_fraction
        if standing_fraction is None and config is not None
        else (standing_fraction or 0.0)
    )
    twist = sample_uniform(ranges, generator=generator, device=device, dtype=dtype)
    standing = (
        torch.rand((), generator=generator, device=device, dtype=dtype) < standing_fraction_value
    )
    if standing:
        twist[:] = 0.0
    turn_in_place = torch.rand((), generator=generator, device=device, dtype=dtype) < turn_fraction
    if turn_in_place:
        twist[:2] = 0.0
        lo, hi = ranges[2]
        max_rate = max(abs(lo), abs(hi))
        magnitude = (
            torch.rand((), generator=generator, device=device, dtype=dtype) * (max_rate * 0.6)
            + max_rate * 0.4
        )
        twist[2] = torch.where(
            torch.rand((), generator=generator, device=device) < 0.5, -magnitude, magnitude
        )
    return twist
