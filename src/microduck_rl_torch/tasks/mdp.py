"""Task-term compatibility surface.

Upstream exposes custom task terms through ``tasks/mdp.py``.  The Torch
runtime keeps the implementation split into focused manager modules, but
re-exports the current velocity terms here so task config files can use the
same conceptual import boundary without copying the upstream monolith.
"""

from microduck_rl_torch.envs.observations import build_actor_observation, command_vector
from microduck_rl_torch.envs.rewards import (
    compute_reward,
    compute_velocity_reward_terms,
    foot_contact_mask,
    self_collision,
)

__all__ = [
    "build_actor_observation",
    "command_vector",
    "compute_reward",
    "compute_velocity_reward_terms",
    "foot_contact_mask",
    "self_collision",
]
