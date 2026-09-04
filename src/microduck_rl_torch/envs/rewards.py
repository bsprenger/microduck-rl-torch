"""Pure Torch reward and contact helpers for the MicroDuck velocity task."""

from __future__ import annotations

from typing import Any

import torch

from .config import RewardConfig
from .model import MicroDuckModelBundle


def _scalar(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def contact_valid(data: Any) -> torch.Tensor:
    """Return the valid contact slots in a fixed-size ``mujoco-torch`` buffer."""

    count = max(0, min(_scalar(data.ncon), data.contact.geom1.shape[-1]))
    slots = torch.arange(data.contact.geom1.shape[-1], device=data.contact.geom1.device)
    return (slots < count) & (data.contact.dist <= data.contact.includemargin)


def foot_contact_mask(data: Any, bundle: MicroDuckModelBundle) -> torch.Tensor:
    """Return left/right foot contact flags for robot-ground contacts."""

    valid = contact_valid(data)
    geom1 = data.contact.geom1
    geom2 = data.contact.geom2
    result = []
    for foot_group in bundle.foot_geom_groups:
        foot_ids = torch.as_tensor(foot_group, device=geom1.device)
        result.append(
            valid
            & (torch.isin(geom1, foot_ids) | torch.isin(geom2, foot_ids))
            & (data.contact.dist <= data.contact.includemargin)
        )
    return torch.stack([mask.any() for mask in result])


def self_collision(data: Any, bundle: MicroDuckModelBundle) -> torch.Tensor:
    """Detect robot self contacts while excluding contacts with world geoms."""

    valid = contact_valid(data)
    geom1 = data.contact.geom1
    geom2 = data.contact.geom2
    robot_geoms = torch.as_tensor(bundle.collision_geom_ids, device=geom1.device)
    robot_pair = torch.isin(geom1, robot_geoms) & torch.isin(geom2, robot_geoms)
    return (valid & robot_pair).any()


def _body_linear_velocity(data: Any, bundle: MicroDuckModelBundle) -> torch.Tensor:
    """Return trunk-link linear velocity in the body frame.

    ``subtree_linvel`` is a valid MuJoCo quantity, but the local
    ``mujoco-torch`` data path does not currently populate it for a free-root
    body.  MuJoCo's ``cvel[..., 3:]`` is the root-link velocity used by mjlab
    and is populated consistently by both backends.
    """

    return data.cvel[bundle.trunk_body_id, 3:6]


def compute_velocity_reward_terms(
    bundle: MicroDuckModelBundle,
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
    q_error = q - bundle.default_pose
    leg_indices = torch.tensor([0, 1, 2, 3, 4, 9, 10, 11, 12, 13], device=q.device)
    speed = torch.linalg.vector_norm(command[:2]) + torch.abs(command[2])
    pose_std_values = (
        config.pose_standing_std if speed < config.walking_threshold else config.pose_walking_std
    )
    leg_std = torch.tensor(
        pose_std_values,
        dtype=q.dtype,
        device=q.device,
    )
    pose = torch.exp(-torch.square(q_error.index_select(-1, leg_indices) / leg_std)).mean()

    gravity = data.xmat[bundle.trunk_body_id].transpose(-1, -2) @ torch.tensor(
        [0.0, 0.0, -1.0], dtype=q.dtype, device=q.device
    )
    upright = torch.exp(-torch.sum(torch.square(gravity[:2])) / config.upright_std**2)

    body_velocity = _body_linear_velocity(data, bundle)
    linear_error = body_velocity[:2] - command[:2]
    track_linear_velocity = torch.exp(
        -torch.sum(torch.square(linear_error)) / config.velocity_std**2
    )
    angular_velocity = data.sensordata[..., bundle.sensor_slices["imu_ang_vel"]]
    angular_error = angular_velocity[2] - command[2]
    track_angular_velocity = torch.exp(
        -torch.square(angular_error) / config.angular_velocity_std**2
    )

    speed_active = speed >= config.walking_threshold
    air_low, air_high = config.air_time_range
    touchdown = foot_contact if foot_touchdown is None else foot_touchdown
    air_time = torch.where(
        speed_active,
        torch.sum(touchdown & (foot_air_time >= air_low) & (foot_air_time <= air_high)).to(q.dtype),
        torch.zeros((), dtype=q.dtype, device=q.device),
    )

    foot_position = data.site_xpos[list(bundle.foot_site_ids)]
    foot_velocity = (foot_position - previous_foot_positions) / (
        bundle.timestep * bundle.decimation
    )
    foot_slip = torch.sum(torch.square(foot_velocity[:, :2]) * foot_contact.unsqueeze(-1))
    clearance_error = torch.relu(config.foot_target_height - foot_position[:, 2])
    foot_clearance = torch.sum(torch.square(clearance_error) * foot_contact.to(q.dtype))
    swing_height = torch.relu(config.foot_target_height - foot_position[:, 2])
    foot_swing_height = torch.sum(torch.square(swing_height) * (~foot_contact).to(q.dtype))
    if not speed_active:
        foot_slip = torch.zeros_like(foot_slip)
        foot_clearance = torch.zeros_like(foot_clearance)
        foot_swing_height = torch.zeros_like(foot_swing_height)

    head_position = q[5:9]
    head_backlash = data.qpos.index_select(-1, bundle.backlash_qpos_indices[5:9])
    head_position = head_position + head_backlash * bundle.backlash_mask[5:9]
    head_error = (head_position - bundle.default_pose[5:9]) - command[3:7]
    head_pose_tracking = torch.exp(-torch.square(head_error / 0.5)).mean()
    body_ang_vel = torch.sum(torch.square(angular_velocity))
    angular_momentum = torch.sum(torch.square(data.subtree_angmom[bundle.trunk_body_id]))
    action_rate_l2 = torch.sum(torch.square(action - previous_action))
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


def compute_reward(
    bundle: MicroDuckModelBundle,
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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compatibility wrapper returning the historical weighted reward."""

    terms = compute_velocity_reward_terms(
        bundle,
        data,
        command=command,
        action=action,
        previous_action=previous_action,
        previous_foot_positions=previous_foot_positions,
        foot_air_time=foot_air_time,
        foot_contact=foot_contact,
        config=config,
        foot_touchdown=foot_touchdown,
    )
    weights = {
        "pose": config.pose,
        "upright": config.upright,
        "track_linear_velocity": config.track_linear_velocity,
        "track_angular_velocity": config.track_angular_velocity,
        "air_time": config.air_time,
        "head_pose_tracking": config.head_pose_tracking,
        "foot_slip": config.foot_slip,
        "body_ang_vel": config.body_ang_vel,
        "angular_momentum": config.angular_momentum,
        "action_rate_l2": config.action_rate_l2,
        "foot_clearance": config.foot_clearance,
        "foot_swing_height": config.foot_swing_height,
        "self_collisions": config.self_collisions,
    }
    weighted = torch.stack([terms[name] * weight for name, weight in weights.items()]).sum()
    return weighted.to(dtype=torch.float32), terms
