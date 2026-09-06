"""Pure Torch reward and contact helpers for the MicroDuck velocity task."""

from __future__ import annotations

from typing import Any

import torch

from .config import RewardConfig
from .model import ModelBundle


def _scalar(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def contact_valid(data: Any) -> torch.Tensor:
    """Return the valid contact slots in a fixed-size ``mujoco-torch`` buffer."""

    slots = torch.arange(data.contact.geom1.shape[-1], device=data.contact.geom1.device)
    count = torch.as_tensor(data.ncon, device=slots.device).to(torch.long)
    count = count.clamp(0, data.contact.geom1.shape[-1])
    valid_slots = slots < count[..., None] if count.ndim else slots < count
    return valid_slots & (data.contact.dist <= data.contact.includemargin)


def foot_contact_mask(data: Any, bundle: ModelBundle) -> torch.Tensor:
    """Return left/right foot contact flags for robot-ground contacts."""

    foot_geom_groups = bundle.handle("foot_geom_groups")
    if not foot_geom_groups:
        shape = data.contact.geom1.shape[:-1] + (0,)
        return torch.zeros(shape, dtype=torch.bool, device=data.contact.geom1.device)
    valid = contact_valid(data)
    geom1 = data.contact.geom1
    geom2 = data.contact.geom2
    result = []
    for foot_group in foot_geom_groups:
        foot_ids = torch.as_tensor(foot_group, device=geom1.device)
        result.append(
            valid
            & (torch.isin(geom1, foot_ids) | torch.isin(geom2, foot_ids))
            & (data.contact.dist <= data.contact.includemargin)
        )
    return (
        torch.stack([mask.any(dim=-1) for mask in result], dim=-1)
        if geom1.ndim > 1
        else torch.stack([mask.any() for mask in result])
    )


def self_collision(data: Any, bundle: ModelBundle) -> torch.Tensor:
    """Detect robot self contacts while excluding contacts with world geoms."""

    valid = contact_valid(data)
    geom1 = data.contact.geom1
    geom2 = data.contact.geom2
    robot_geoms = torch.as_tensor(
        bundle.handle("collision_geom_ids"), dtype=geom1.dtype, device=geom1.device
    )
    robot_pair = torch.isin(geom1, robot_geoms) & torch.isin(geom2, robot_geoms)
    result = (valid & robot_pair).any(dim=-1) if geom1.ndim > 1 else (valid & robot_pair).any()
    return result


def _body_linear_velocity(data: Any, bundle: ModelBundle) -> torch.Tensor:
    """Return trunk-link linear velocity in the body frame.

    ``subtree_linvel`` is a valid MuJoCo quantity, but the local
    ``mujoco-torch`` data path does not currently populate it for a free-root
    body.  MuJoCo's ``cvel[..., 3:]`` is the root-link velocity used by mjlab
    and is populated consistently by both backends.
    """

    return data.cvel[..., bundle.root_body_id, 3:6]


def compute_velocity_reward_terms(
    bundle: ModelBundle,
    data: Any,
    *,
    command: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    previous_foot_positions: torch.Tensor,
    foot_air_time: torch.Tensor,
    foot_contact: torch.Tensor,
    config: RewardConfig,
    foot_touchdown: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute raw upstream velocity terms for one scalar environment.

    Keeping raw terms separate from their weights is what lets the generic
    ``RewardManager`` add, replace, or remove terms without embedding a task's
    reward policy in the physics layer.
    """

    q = data.qpos.index_select(-1, bundle.qpos_indices)
    batched = q.ndim == 2
    q_error = q - bundle.default_pose
    leg_indices = torch.tensor([0, 1, 2, 3, 4, 9, 10, 11, 12, 13], device=q.device)
    speed = torch.linalg.vector_norm(command[..., :2], dim=-1) + torch.abs(command[..., 2])
    pose_standing = torch.tensor(config.pose_standing_std, dtype=q.dtype, device=q.device)
    pose_walking = torch.tensor(config.pose_walking_std, dtype=q.dtype, device=q.device)
    leg_std = torch.where(speed[..., None] < config.walking_threshold, pose_standing, pose_walking)
    pose_values = torch.exp(-torch.square(q_error.index_select(-1, leg_indices) / leg_std))
    pose = pose_values.mean(dim=-1) if batched else pose_values.mean()

    gravity = data.xmat[..., bundle.root_body_id, :, :].transpose(-1, -2) @ torch.tensor(
        [0.0, 0.0, -1.0], dtype=q.dtype, device=q.device
    )
    upright = torch.exp(
        -torch.sum(torch.square(gravity[..., :2]), dim=-1 if batched else None)
        / config.upright_std**2
    )

    body_velocity = _body_linear_velocity(data, bundle)
    linear_error = body_velocity[..., :2] - command[..., :2]
    track_linear_velocity = torch.exp(
        -torch.sum(torch.square(linear_error), dim=-1 if batched else None) / config.velocity_std**2
    )
    angular_velocity = data.sensordata[..., bundle.sensor_slices["imu_ang_vel"]]
    angular_error = angular_velocity[..., 2] - command[..., 2]
    track_angular_velocity = torch.exp(
        -torch.square(angular_error) / config.angular_velocity_std**2
    )

    speed_active = speed >= config.walking_threshold
    air_low, air_high = config.air_time_range
    touchdown = foot_contact if foot_touchdown is None else foot_touchdown
    air_time = torch.where(
        speed_active,
        torch.sum(
            touchdown & (foot_air_time >= air_low) & (foot_air_time <= air_high),
            dim=-1 if batched else None,
        ).to(q.dtype),
        torch.zeros_like(speed),
    )

    foot_site_ids = bundle.handle("foot_site_ids")
    foot_position = data.site_xpos[..., list(foot_site_ids), :]
    foot_velocity = (foot_position - previous_foot_positions) / (
        bundle.timestep * bundle.decimation
    )
    foot_slip_values = torch.square(foot_velocity[..., :2]) * foot_contact.to(q.dtype).unsqueeze(-1)
    foot_slip = torch.sum(foot_slip_values, dim=(-2, -1) if batched else None)
    clearance_error = torch.relu(config.foot_target_height - foot_position[..., 2])
    foot_clearance = torch.sum(
        torch.square(clearance_error) * foot_contact.to(q.dtype),
        dim=-1 if batched else None,
    )
    swing_height = torch.relu(config.foot_target_height - foot_position[..., 2])
    foot_swing_height = torch.sum(
        torch.square(swing_height) * (~foot_contact).to(q.dtype),
        dim=-1 if batched else None,
    )
    foot_slip = torch.where(speed_active, foot_slip, torch.zeros_like(foot_slip))
    foot_clearance = torch.where(speed_active, foot_clearance, torch.zeros_like(foot_clearance))
    foot_swing_height = torch.where(
        speed_active, foot_swing_height, torch.zeros_like(foot_swing_height)
    )

    head_position = q[..., 5:9]
    head_backlash = data.qpos.index_select(-1, bundle.backlash_qpos_indices[5:9])
    head_position = head_position + head_backlash * bundle.backlash_mask[5:9]
    head_error = (head_position - bundle.default_pose[5:9]) - command[..., 3:7]
    head_pose_tracking = torch.exp(-torch.square(head_error / 0.5)).mean(
        dim=-1 if batched else None
    )
    body_ang_vel = torch.sum(torch.square(angular_velocity), dim=-1 if batched else None)
    angular_momentum = torch.sum(
        torch.square(data.subtree_angmom[..., bundle.root_body_id, :]), dim=-1
    )
    action_rate_l2 = torch.sum(torch.square(action - previous_action), dim=-1 if batched else None)
    self_collisions = self_collision(data, bundle).to(q.dtype)

    terms = {
        "pose": pose,
        "upright": upright,
        "track_linear_velocity": track_linear_velocity,
        "track_angular_velocity": track_angular_velocity,
        "air_time": air_time,
        "head_pose_tracking": head_pose_tracking,
        "foot_slip": foot_slip,
        "body_ang_vel": body_ang_vel,
        "angular_momentum": angular_momentum,
        "action_rate_l2": action_rate_l2,
        "foot_clearance": foot_clearance,
        "foot_swing_height": foot_swing_height,
        "self_collisions": self_collisions,
    }
    return terms


_VELOCITY_TERM_NAMES = (
    "pose",
    "upright",
    "track_linear_velocity",
    "track_angular_velocity",
    "air_time",
    "head_pose_tracking",
    "foot_slip",
    "body_ang_vel",
    "angular_momentum",
    "action_rate_l2",
    "foot_clearance",
    "foot_swing_height",
    "self_collisions",
)


def velocity_term(name: str):  # type: ignore[no-untyped-def]
    """Return one configured reward-term function for the velocity task.

    The feature calculation is shared and cached for one transition, but the
    manager still invokes and weights each named term independently. This
    preserves the current parity math while allowing another task to add,
    remove, or replace terms without supplying a monolithic evaluator.
    """

    if name not in _VELOCITY_TERM_NAMES:
        raise KeyError(f"Unknown velocity reward term {name!r}")

    def evaluate(env: Any) -> torch.Tensor:
        cached = getattr(env, "_velocity_reward_cache", None)
        if cached is None:
            transition = env.transition
            if transition is None or env.data is None:
                raise RuntimeError("Velocity reward terms require an active transition")
            if (
                transition.previous_foot_positions is None
                or transition.foot_air_time is None
                or transition.foot_contact is None
                or transition.foot_touchdown is None
            ):
                raise RuntimeError("Velocity reward terms require configured foot sensors")
            cached = compute_velocity_reward_terms(
                env.bundle,
                env.data,
                command=env.command,
                action=transition.action,
                previous_action=transition.previous_action,
                previous_foot_positions=transition.previous_foot_positions,
                foot_air_time=transition.foot_air_time,
                foot_contact=transition.foot_contact,
                config=env.config.rewards,
                foot_touchdown=transition.foot_touchdown,
            )
            env._velocity_reward_cache = cached
        return cached[name]

    evaluate.__name__ = f"velocity_{name}"
    return evaluate
