"""Direct task factories.

This module intentionally contains no registration side effects.  Call a
factory, inspect/mutate its returned configuration, and pass it to the Torch
manager-based environment directly.
"""

from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg
from .names import BACKLASH_TASK_NAMES, BASE_TASK_NAMES

# The package exposes the velocity task factory and the shared task identifiers
# used for lookup and reporting.

__all__ = [
    "BASE_TASK_NAMES",
    "BACKLASH_TASK_NAMES",
    "MicroduckRlCfg",
    "make_microduck_velocity_env_cfg",
]
