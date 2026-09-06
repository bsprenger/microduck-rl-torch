"""Small, task-independent rendering configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RenderBackend = Literal["mujoco", "mujoco-torch"]
RenderMode = Literal["rgb_array"]


@dataclass(frozen=True)
class CameraConfig:
    """Camera settings shared by native and Torch-backed renderers.

    ``name`` selects a fixed MuJoCo camera.  Otherwise a free camera is used;
    ``track_body`` turns that free camera into a MuJoCo tracking camera.  The
    default follows the current MicroDuck artifact workflow, where the free
    camera follows the root in the XY plane while retaining a fixed view.
    """

    name: str | None = None
    track_body: str | None = None
    follow_root: bool = True
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.10)
    distance: float = 0.65
    azimuth: float = 135.0
    elevation: float = -18.0
    fovy: float | None = None

    def __post_init__(self) -> None:
        if self.name is not None and self.track_body is not None:
            raise ValueError("CameraConfig.name and track_body are mutually exclusive")
        if len(self.lookat) != 3:
            raise ValueError("Camera lookat must contain exactly three values")
        if self.distance <= 0.0:
            raise ValueError("Camera distance must be positive")
        if self.fovy is not None and self.fovy <= 0.0:
            raise ValueError("Camera fovy must be positive")


@dataclass(frozen=True)
class RenderConfig:
    """Configuration for the environment-owned RGB renderer."""

    backend: RenderBackend = "mujoco"
    width: int = 320
    height: int = 240
    camera: CameraConfig = field(default_factory=CameraConfig)
    ray_chunk_size: int = 256

    def __post_init__(self) -> None:
        if self.backend not in ("mujoco", "mujoco-torch"):
            raise ValueError(f"Unknown render backend {self.backend!r}")
        if self.width < 1 or self.height < 1:
            raise ValueError("Render dimensions must be positive")
        if self.ray_chunk_size < 1:
            raise ValueError("Ray chunk size must be positive")
