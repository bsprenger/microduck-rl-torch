"""Task-term exports for task configuration modules.

The environment keeps its implementation split into focused manager modules,
but exports the current terms here so task config files have one stable import
boundary.
"""

from microduck_rl_torch.envs.observations import command_term, command_vector
from microduck_rl_torch.envs.rewards import (
    compute_velocity_reward_terms,
    foot_contact_mask,
    self_collision,
)

__all__ = [
    "command_vector",
    "command_term",
    "compute_velocity_reward_terms",
    "foot_contact_mask",
    "self_collision",
]
