"""Explicit configuration for the upstream MicroDuck velocity task."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, MutableMapping, Sequence
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
    # BAM electrical parameters are task configuration, not backend
    # constants.  Keeping them here lets future actuator variants reuse the
    # same lifecycle while changing only their randomization policy.
    vin_range: tuple[float, float] = (6.5, 8.2)
    vin_drop_gain_range: tuple[float, float] = (0.0, 0.2)
    imu_angle_degrees: float = 6.0
    encoder_bias_range: tuple[float, float] = (-0.015, 0.015)
    base_pitch_degrees: float = 10.0
    base_roll_degrees: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "mass_inertia_range",
            "joint_friction_range",
            "armature_range",
            "foot_friction_range",
            "velocity_push_interval",
            "velocity_push_range",
            "vin_range",
            "vin_drop_gain_range",
            "encoder_bias_range",
        ):
            value = getattr(self, name)
            if len(value) != 2 or float(value[0]) > float(value[1]):
                raise ValueError(f"{name} must be a two-value ascending range")
        if any(
            float(value) < 0
            for value in (self.imu_angle_degrees, self.base_pitch_degrees, self.base_roll_degrees)
        ):
            raise ValueError("Orientation randomization degrees must be non-negative")


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


def _random_batch(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | Sequence[torch.Generator] | None,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    """Draw independent rows when a batched backend supplies per-env streams."""

    if not isinstance(generator, Sequence):
        return torch.rand(shape, generator=generator, device=device, dtype=dtype)
    if len(generator) != batch_size:
        raise ValueError("A batched random draw needs one generator per environment")
    if batch_size == 1:
        return torch.rand(shape, generator=generator[0], device=device, dtype=dtype)
    row_shape = shape[1:] if shape and shape[0] == batch_size else shape
    return torch.stack(
        [
            torch.rand(row_shape, generator=stream, device=device, dtype=dtype)
            for stream in generator
        ]
    )


def sample_uniform(
    ranges: tuple[tuple[float, float], ...],
    *,
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
) -> torch.Tensor:
    """Sample one vector or a batch of vectors without hidden global RNG."""

    shape = (len(ranges),) if batch_size == 1 else (batch_size, len(ranges))
    values = torch.empty(shape, device=device, dtype=dtype)
    for index, (low, high) in enumerate(ranges):
        random_shape = () if batch_size == 1 else (batch_size,)
        values[..., index] = (
            _random_batch(
                random_shape,
                generator=generator,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
            )
            * (high - low)
            + low
        )
    return values


def sample_twist(
    config: CommandConfig | None,
    *,
    generator: torch.Generator | Sequence[torch.Generator] | None,
    device: torch.device,
    dtype: torch.dtype,
    twist_ranges: tuple[tuple[float, float], ...] | None = None,
    turn_in_place_fraction: float | None = None,
    standing_fraction: float | None = None,
    batch_size: int = 1,
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
    twist = sample_uniform(
        ranges, generator=generator, device=device, dtype=dtype, batch_size=batch_size
    )
    random_shape = () if batch_size == 1 else (batch_size,)
    standing = (
        _random_batch(
            random_shape,
            generator=generator,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        < standing_fraction_value
    )
    twist = torch.where(standing[..., None], torch.zeros_like(twist), twist)
    turn_in_place = (
        _random_batch(
            random_shape,
            generator=generator,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        < turn_fraction
    )
    twist[..., :2] = torch.where(
        turn_in_place[..., None], torch.zeros_like(twist[..., :2]), twist[..., :2]
    )
    if bool(turn_in_place.any()):
        lo, hi = ranges[2]
        max_rate = max(abs(lo), abs(hi))
        magnitude = (
            _random_batch(
                random_shape,
                generator=generator,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
            )
            * (max_rate * 0.6)
            + max_rate * 0.4
        )
        sign = torch.where(
            _random_batch(
                random_shape,
                generator=generator,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
            )
            < 0.5,
            -magnitude,
            magnitude,
        )
        twist[..., 2] = torch.where(turn_in_place, sign, twist[..., 2])
    return twist
