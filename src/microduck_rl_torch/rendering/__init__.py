"""Environment rendering and rollout artifact helpers."""

from .config import CameraConfig, RenderBackend, RenderConfig, RenderMode
from .video import VideoWriter, convert_video_to_gif

__all__ = [
    "CameraConfig",
    "RenderBackend",
    "RenderConfig",
    "RenderMode",
    "VideoWriter",
    "convert_video_to_gif",
    "render_policy_rollout",
]


def __getattr__(name: str):
    # Keep the policy-specific helper import lazy.  The environment imports
    # rendering configuration during construction, and eager importing
    # rollout.py would create an env -> rendering -> rollout -> env cycle.
    if name == "render_policy_rollout":
        from .rollout import render_policy_rollout

        return render_policy_rollout
    raise AttributeError(name)
