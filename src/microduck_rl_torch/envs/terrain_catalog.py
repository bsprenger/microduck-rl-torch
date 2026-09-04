"""Additional built-in rough-terrain generators."""

from __future__ import annotations

import colorsys
import math
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import interpolate

from .scene import (
    BoxPyramidStairsTerrainCfg,
    FlatPatchSamplingCfg,
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
    _find_flat_patches_from_heightfield,
    _mujoco_geom_box,
    _safe_box_size,
    _terrain_body,
)

_BLUE = (0.20, 0.45, 0.95, 1.0)
_GREEN = (0.25, 0.80, 0.45, 1.0)
_RED = (0.90, 0.30, 0.30, 1.0)


def _brand_ramp(
    base_rgb: tuple[float, float, float], parameter: float
) -> tuple[float, float, float, float]:
    """Match the one-sample color ramp without changing RNG draw order."""

    hue, saturation, _value = colorsys.rgb_to_hsv(*base_rgb)
    value = 0.75 + 0.25 * parameter
    saturation = float(np.clip(saturation * (0.85 + 0.25 * parameter), 0.0, 1.0))
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return red, green, blue, 1.0


def _box(
    body: Any,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float, float],
    *,
    quat: tuple[float, float, float, float] | None = None,
) -> TerrainGeometry:
    geom = body.add_geom(type=_mujoco_geom_box(), size=_safe_box_size(size), pos=pos)
    if quat is not None:
        geom.quat = quat
    return TerrainGeometry(geom=geom, color=color)


def _border(
    body: Any,
    size: tuple[float, float],
    inner: tuple[float, float],
    height: float,
    z: float,
    color: tuple[float, float, float, float],
) -> list[TerrainGeometry]:
    tx = max(1.0e-6, (size[0] - inner[0]) / 2)
    ty = max(1.0e-6, (size[1] - inner[1]) / 2)
    return [
        _box(body, (size[0], ty, height), (size[0] / 2, size[1] - ty / 2, z), color),
        _box(body, (size[0], ty, height), (size[0] / 2, ty / 2, z), color),
        _box(
            body,
            (tx, max(1.0e-6, size[1] - 2 * ty), height),
            (tx / 2, size[1] / 2, z),
            color,
        ),
        _box(
            body,
            (tx, max(1.0e-6, size[1] - 2 * ty), height),
            (size[0] - tx / 2, size[1] / 2, z),
            color,
        ),
    ]


@dataclass(kw_only=True)
class BoxInvertedPyramidStairsTerrainCfg(BoxPyramidStairsTerrainCfg):
    """Four-sided staircase descending toward the center."""

    def function(self, difficulty: float, spec: Any, rng: np.random.Generator) -> TerrainOutput:
        del rng
        body = _terrain_body(spec)
        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )
        num_steps_x = int(
            (self.size[0] - 2 * self.border_width - self.platform_width) / (2 * self.step_width)
        )
        num_steps_y = int(
            (self.size[1] - 2 * self.border_width - self.platform_width) / (2 * self.step_width)
        )
        num_steps = max(0, int(min(num_steps_x, num_steps_y)))
        total_height = (num_steps + 1) * step_height
        center = (self.size[0] / 2, self.size[1] / 2)
        terrain_size = (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width)
        result = (
            _border(body, self.size, terrain_size, step_height, -step_height / 2, _RED)
            if self.border_width > 0 and not self.holes
            else []
        )
        for k in range(num_steps):
            outer = (
                (self.platform_width, self.platform_width)
                if self.holes
                else (
                    terrain_size[0] - 2 * k * self.step_width,
                    terrain_size[1] - 2 * k * self.step_width,
                )
            )
            z = -total_height / 2 - (k + 1) * step_height / 2
            h = total_height - (k + 1) * step_height
            offset = (k + 0.5) * self.step_width
            cross_y = outer[1] if self.holes else outer[1] - 2 * self.step_width
            result.extend(
                (
                    _box(
                        body,
                        (outer[0], self.step_width, h),
                        (center[0], center[1] + terrain_size[1] / 2 - offset, z),
                        _RED,
                    ),
                    _box(
                        body,
                        (outer[0], self.step_width, h),
                        (center[0], center[1] - terrain_size[1] / 2 + offset, z),
                        _RED,
                    ),
                    _box(
                        body,
                        (self.step_width, cross_y, h),
                        (center[0] + terrain_size[0] / 2 - offset, center[1], z),
                        _RED,
                    ),
                    _box(
                        body,
                        (self.step_width, cross_y, h),
                        (center[0] - terrain_size[0] / 2 + offset, center[1], z),
                        _RED,
                    ),
                )
            )
        result.append(
            _box(
                body,
                (
                    terrain_size[0] - 2 * num_steps * self.step_width,
                    terrain_size[1] - 2 * num_steps * self.step_width,
                    step_height,
                ),
                (center[0], center[1], -total_height - step_height / 2),
                _RED,
            )
        )
        return TerrainOutput(
            origin=np.asarray((center[0], center[1], -(num_steps + 1) * step_height)),
            geometries=result,
        )


@dataclass(kw_only=True)
class BoxRandomSpreadTerrainCfg(SubTerrainCfg):
    """Randomly sized and oriented boxes around a central platform."""

    num_boxes: int = 60
    box_width_range: tuple[float, float] = (0.3, 1.0)
    box_length_range: tuple[float, float] = (0.3, 1.0)
    box_height_range: tuple[float, float] = (0.05, 1.0)
    box_yaw_range: tuple[float, float] = (0.0, 360.0)
    add_floor: bool = True
    platform_width: float = 1.0
    border_width: float = 0.25

    def function(self, difficulty: float, spec: object, rng: np.random.Generator) -> TerrainOutput:
        body = _terrain_body(spec)
        result = (
            _border(
                body,
                self.size,
                (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
                1.0,
                -0.5,
                _BLUE,
            )
            if self.border_width > 0
            else []
        )
        if self.add_floor:
            result.append(
                _box(
                    body,
                    (
                        self.size[0] - 2 * self.border_width,
                        self.size[1] - 2 * self.border_width,
                        0.1,
                    ),
                    (self.size[0] / 2, self.size[1] / 2, -0.05),
                    (0.4, 0.4, 0.4, 1.0),
                )
            )
        result.append(
            _box(
                body,
                (self.platform_width, self.platform_width, 1.0),
                (self.size[0] / 2, self.size[1] / 2, -0.5),
                (0.4, 0.4, 0.4, 1.0),
            )
        )
        pmin = self.size[0] / 2 - self.platform_width / 2
        pmax = self.size[0] / 2 + self.platform_width / 2
        for _ in range(int(self.num_boxes * (0.5 + 0.5 * difficulty))):
            sx = float(rng.uniform(*self.box_width_range))
            sy = float(rng.uniform(*self.box_length_range))
            height = float(rng.uniform(*self.box_height_range)) * (0.2 + 0.8 * difficulty)
            px = float(
                rng.uniform(self.border_width + sx / 2, self.size[0] - self.border_width - sx / 2)
            )
            py = float(
                rng.uniform(self.border_width + sy / 2, self.size[1] - self.border_width - sy / 2)
            )
            if pmin - sx / 2 <= px <= pmax + sx / 2 and pmin - sy / 2 <= py <= pmax + sy / 2:
                continue
            yaw = float(np.deg2rad(rng.uniform(*self.box_yaw_range)))
            color = _brand_ramp(_BLUE[:3], float(rng.uniform(0.3, 0.8)))
            result.append(
                _box(
                    body,
                    (sx, sy, height),
                    (px, py, height / 2),
                    color,
                    quat=(float(np.cos(yaw / 2)), 0, 0, float(np.sin(yaw / 2))),
                )
            )
        return TerrainOutput(
            origin=np.asarray((self.size[0] / 2, self.size[1] / 2, 0)), geometries=result
        )


@dataclass(kw_only=True)
class BoxOpenStairsTerrainCfg(SubTerrainCfg):
    """Open bowl or pyramid staircase made from thin top plates."""

    step_height_range: tuple[float, float] = (0.1, 0.2)
    step_width_range: tuple[float, float] = (0.4, 0.8)
    platform_width: float = 1.0
    border_width: float = 0.25
    step_thickness: float = 0.05
    inverted: bool = True

    def function(self, difficulty: float, spec: object, rng: np.random.Generator) -> TerrainOutput:
        del rng
        body = _terrain_body(spec)
        height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )
        width = self.step_width_range[1] - difficulty * (
            self.step_width_range[1] - self.step_width_range[0]
        )
        center = (self.size[0] / 2, self.size[1] / 2)
        terrain_size = (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width)
        steps = max(
            0,
            int(
                min(
                    (terrain_size[0] - self.platform_width) / (2 * width),
                    (terrain_size[1] - self.platform_width) / (2 * width),
                )
            ),
        )
        result = (
            _border(body, self.size, terrain_size, height, -height / 2, _BLUE)
            if self.border_width > 0
            else []
        )
        for k in range(steps):
            box_size = (terrain_size[0] - 2 * k * width, terrain_size[1] - 2 * k * width)
            z = (
                -k * height - self.step_thickness / 2
                if self.inverted
                else (k + 1) * height - self.step_thickness / 2
            )
            offset = (k + 0.5) * width
            result.extend(
                (
                    _box(
                        body,
                        (box_size[0], width, self.step_thickness),
                        (center[0], center[1] + terrain_size[1] / 2 - offset, z),
                        _BLUE,
                    ),
                    _box(
                        body,
                        (box_size[0], width, self.step_thickness),
                        (center[0], center[1] - terrain_size[1] / 2 + offset, z),
                        _BLUE,
                    ),
                    _box(
                        body,
                        (width, box_size[1] - 2 * width, self.step_thickness),
                        (center[0] + terrain_size[0] / 2 - offset, center[1], z),
                        _BLUE,
                    ),
                    _box(
                        body,
                        (width, box_size[1] - 2 * width, self.step_thickness),
                        (center[0] - terrain_size[0] / 2 + offset, center[1], z),
                        _BLUE,
                    ),
                )
            )
        platform_z = (
            -steps * height - self.step_thickness / 2
            if self.inverted
            else (steps + 1) * height - self.step_thickness / 2
        )
        result.append(
            _box(
                body,
                (
                    terrain_size[0] - 2 * steps * width,
                    terrain_size[1] - 2 * steps * width,
                    self.step_thickness,
                ),
                (center[0], center[1], platform_z),
                _BLUE,
            )
        )
        return TerrainOutput(
            origin=np.asarray((center[0], center[1], platform_z + self.step_thickness / 2)),
            geometries=result,
        )


@dataclass(kw_only=True)
class BoxRandomStairsTerrainCfg(SubTerrainCfg):
    """Pyramid stairs whose individual rises are sampled independently."""

    step_width: float = 0.8
    step_height_range: tuple[float, float] = (0.1, 0.3)
    platform_width: float = 1.0
    border_width: float = 0.25

    def function(self, difficulty: float, spec: object, rng: np.random.Generator) -> TerrainOutput:
        body = _terrain_body(spec)
        center = (self.size[0] / 2, self.size[1] / 2)
        terrain_size = (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width)
        steps = max(
            0,
            int(
                min(
                    (terrain_size[0] - self.platform_width) / (2 * self.step_width),
                    (terrain_size[1] - self.platform_width) / (2 * self.step_width),
                )
            ),
        )
        result = (
            _border(body, self.size, terrain_size, 0.1, -0.05, _BLUE)
            if self.border_width > 0
            else []
        )
        current = 0.0
        for k in range(steps):
            current += float(rng.uniform(*self.step_height_range)) * (0.5 + 0.5 * difficulty)
            offset = (k + 0.5) * self.step_width
            box_size = (
                terrain_size[0] - 2 * k * self.step_width,
                terrain_size[1] - 2 * k * self.step_width,
            )
            result.extend(
                (
                    _box(
                        body,
                        (box_size[0], self.step_width, current),
                        (center[0], center[1] + terrain_size[1] / 2 - offset, current / 2),
                        _BLUE,
                    ),
                    _box(
                        body,
                        (box_size[0], self.step_width, current),
                        (center[0], center[1] - terrain_size[1] / 2 + offset, current / 2),
                        _BLUE,
                    ),
                    _box(
                        body,
                        (self.step_width, box_size[1] - 2 * self.step_width, current),
                        (center[0] + terrain_size[0] / 2 - offset, center[1], current / 2),
                        _BLUE,
                    ),
                    _box(
                        body,
                        (self.step_width, box_size[1] - 2 * self.step_width, current),
                        (center[0] - terrain_size[0] / 2 + offset, center[1], current / 2),
                        _BLUE,
                    ),
                )
            )
        result.append(
            _box(
                body,
                (
                    terrain_size[0] - 2 * steps * self.step_width,
                    terrain_size[1] - 2 * steps * self.step_width,
                    current,
                ),
                (center[0], center[1], current / 2),
                _BLUE,
            )
        )
        return TerrainOutput(origin=np.asarray((center[0], center[1], current)), geometries=result)


@dataclass(kw_only=True)
class BoxSteppingStonesTerrainCfg(SubTerrainCfg):
    """Separated stones over a deep floor with difficulty-scaled spacing."""

    stone_size_range: tuple[float, float] = (0.4, 0.8)
    stone_distance_range: tuple[float, float] = (0.2, 0.5)
    stone_height: float = 0.2
    stone_height_variation: float = 0.1
    stone_size_variation: float = 0.1
    floor_depth: float = 2.0
    displacement_range: float = 0.1
    platform_width: float = 1.0
    border_width: float = 0.25

    def function(self, difficulty: float, spec: object, rng: np.random.Generator) -> TerrainOutput:
        body = _terrain_body(spec)
        distance = self.stone_distance_range[0] + difficulty * (
            self.stone_distance_range[1] - self.stone_distance_range[0]
        )
        stone_size = self.stone_size_range[1] - difficulty * (
            self.stone_size_range[1] - self.stone_size_range[0]
        )
        spacing = stone_size + distance
        inner_x, inner_y = (
            self.size[0] - 2 * self.border_width,
            self.size[1] - 2 * self.border_width,
        )
        nx, ny = int(np.floor(inner_x / spacing)) + 1, int(np.floor(inner_y / spacing)) + 1
        ox, oy = (
            self.border_width + (inner_x - (nx - 1) * spacing) / 2,
            self.border_width + (inner_y - (ny - 1) * spacing) / 2,
        )
        z = (self.stone_height - self.floor_depth) / 2
        result = (
            _border(
                body, self.size, (inner_x, inner_y), self.stone_height + self.floor_depth, z, _GREEN
            )
            if self.border_width > 0
            else []
        )
        result.append(
            _box(
                body,
                (self.size[0], self.size[1], 0.1),
                (self.size[0] / 2, self.size[1] / 2, -self.floor_depth - 0.05),
                (0.1, 0.1, 0.1, 1.0),
            )
        )
        result.append(
            _box(
                body,
                (self.platform_width, self.platform_width, self.stone_height + self.floor_depth),
                (self.size[0] / 2, self.size[1] / 2, z),
                _GREEN,
            )
        )
        pmin, pmax = (
            self.size[0] / 2 - self.platform_width / 2,
            self.size[0] / 2 + self.platform_width / 2,
        )
        for i in range(nx):
            for j in range(ny):
                px = (
                    ox
                    + i * spacing
                    + rng.uniform(
                        -self.displacement_range * difficulty, self.displacement_range * difficulty
                    )
                )
                py = (
                    oy
                    + j * spacing
                    + rng.uniform(
                        -self.displacement_range * difficulty, self.displacement_range * difficulty
                    )
                )
                sx = stone_size + rng.uniform(
                    -self.stone_size_variation * difficulty, self.stone_size_variation * difficulty
                )
                sy = stone_size + rng.uniform(
                    -self.stone_size_variation * difficulty, self.stone_size_variation * difficulty
                )
                if pmin <= px <= pmax and pmin <= py <= pmax:
                    continue
                xmin, xmax = np.clip(
                    (px - sx / 2, px + sx / 2), self.border_width, self.size[0] - self.border_width
                )
                ymin, ymax = np.clip(
                    (py - sy / 2, py + sy / 2), self.border_width, self.size[1] - self.border_width
                )
                if xmax - xmin < 0.05 or ymax - ymin < 0.05:
                    continue
                height = (
                    self.floor_depth
                    + self.stone_height
                    + rng.uniform(
                        -self.stone_height_variation * difficulty,
                        self.stone_height_variation * difficulty,
                    )
                )
                color = _brand_ramp(_GREEN[:3], float(rng.uniform(0.4, 0.7)))
                result.append(
                    _box(
                        body,
                        (xmax - xmin, ymax - ymin, height),
                        ((xmin + xmax) / 2, (ymin + ymax) / 2, -self.floor_depth + height / 2),
                        color,
                    )
                )
        return TerrainOutput(
            origin=np.asarray((self.size[0] / 2, self.size[1] / 2, self.stone_height)),
            geometries=result,
        )


@dataclass(kw_only=True)
class BoxNarrowBeamsTerrainCfg(SubTerrainCfg):
    """Radial narrow beams over a deep floor."""

    num_beams: int = 16
    beam_width_range: tuple[float, float] = (0.2, 0.4)
    beam_height: float = 0.2
    spacing: float = 0.8
    platform_width: float = 1.0
    border_width: float = 0.25
    floor_depth: float = 2.0

    def function(self, difficulty: float, spec: object, rng: np.random.Generator) -> TerrainOutput:
        del rng
        body = _terrain_body(spec)
        width = self.beam_width_range[1] - difficulty * (
            self.beam_width_range[1] - self.beam_width_range[0]
        )
        z = (self.beam_height - self.floor_depth) / 2
        result = (
            _border(
                body,
                self.size,
                (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
                self.beam_height + self.floor_depth,
                z,
                _BLUE,
            )
            if self.border_width > 0
            else []
        )
        result.append(
            _box(
                body,
                (self.size[0], self.size[1], 0.1),
                (self.size[0] / 2, self.size[1] / 2, -self.floor_depth - 0.05),
                (0.1, 0.1, 0.1, 1.0),
            )
        )
        result.append(
            _box(
                body,
                (self.platform_width, self.platform_width, self.beam_height + self.floor_depth),
                (self.size[0] / 2, self.size[1] / 2, z),
                _BLUE,
            )
        )
        inner = self.size[0] - 2 * self.border_width
        radius = self.platform_width / 2
        for angle in np.linspace(0, 2 * np.pi, self.num_beams, endpoint=False):
            ca, sa = abs(np.cos(angle)), abs(np.sin(angle))
            distance = (inner / 2) / max(ca, sa)
            length = distance + (width / 2) * min(ca, sa) / max(ca, sa) - radius
            center_distance = radius + length / 2
            result.append(
                _box(
                    body,
                    (length, width, self.beam_height + self.floor_depth),
                    (
                        self.size[0] / 2 + center_distance * np.cos(angle),
                        self.size[1] / 2 + center_distance * np.sin(angle),
                        z,
                    ),
                    _BLUE,
                    quat=(float(np.cos(angle / 2)), 0, 0, float(np.sin(angle / 2))),
                )
            )
        return TerrainOutput(
            origin=np.asarray((self.size[0] / 2, self.size[1] / 2, self.beam_height)),
            geometries=result,
        )


@dataclass(kw_only=True)
class BoxTiltedGridTerrainCfg(SubTerrainCfg):
    """Independent tilted mesh tiles with a central platform."""

    grid_width: float = 1.0
    tilt_range_deg: float = 15.0
    height_range: float = 0.1
    platform_width: float = 1.0
    border_width: float = 0.25
    floor_depth: float = 2.0

    def function(self, difficulty: float, spec: Any, rng: np.random.Generator) -> TerrainOutput:
        import mujoco

        body = _terrain_body(spec)
        max_tilt = np.deg2rad(self.tilt_range_deg * difficulty)
        height_range = self.height_range * difficulty
        nx = int((self.size[0] - 2 * self.border_width) / self.grid_width)
        ny = int((self.size[1] - 2 * self.border_width) / self.grid_width)
        remainder = self.size[0] - nx * self.grid_width
        start = remainder / 2
        base_height = 0.2
        result = (
            _border(
                body,
                self.size,
                (nx * self.grid_width, ny * self.grid_width),
                base_height + self.floor_depth,
                (base_height - self.floor_depth) / 2,
                _GREEN,
            )
            if remainder > 0
            else []
        )
        result.append(
            _box(
                body,
                (self.size[0], self.size[1], 0.1),
                (self.size[0] / 2, self.size[1] / 2, -self.floor_depth - 0.05),
                (0.1, 0.1, 0.1, 1.0),
            )
        )
        result.append(
            _box(
                body,
                (self.platform_width, self.platform_width, base_height + self.floor_depth),
                (self.size[0] / 2, self.size[1] / 2, (base_height - self.floor_depth) / 2),
                _GREEN,
            )
        )
        pmin, pmax = (
            self.size[0] / 2 - self.platform_width / 2,
            self.size[0] / 2 + self.platform_width / 2,
        )
        faces = [
            4,
            6,
            7,
            4,
            7,
            5,
            0,
            1,
            3,
            0,
            3,
            2,
            0,
            2,
            6,
            0,
            6,
            4,
            1,
            5,
            7,
            1,
            7,
            3,
            0,
            4,
            5,
            0,
            5,
            1,
            2,
            3,
            7,
            2,
            7,
            6,
        ]
        for i in range(nx):
            for j in range(ny):
                x, y = start + (i + 0.5) * self.grid_width, start + (j + 0.5) * self.grid_width
                if pmin <= x <= pmax and pmin <= y <= pmax:
                    continue
                h = base_height + rng.uniform(-height_range / 2, height_range / 2)
                tx, ty = rng.uniform(-max_tilt, max_tilt), rng.uniform(-max_tilt, max_tilt)
                xmin, xmax, ymin, ymax = (
                    x - self.grid_width / 2,
                    x + self.grid_width / 2,
                    y - self.grid_width / 2,
                    y + self.grid_width / 2,
                )
                vertices = [
                    (vx, vy, -self.floor_depth) for vx in (xmin, xmax) for vy in (ymin, ymax)
                ]
                vertices += [
                    (vx, vy, h + tx * (vx - x) + ty * (vy - y))
                    for vx in (xmin, xmax)
                    for vy in (ymin, ymax)
                ]
                mesh = spec.add_mesh(
                    name=f"terrain_tile_{i}_{j}_{int(rng.integers(int(1e9)))}",
                    uservert=np.asarray(vertices).flatten().tolist(),
                    userface=faces,
                )
                color = (*rng.uniform(0.3, 0.7, 3), 1.0)
                result.append(
                    TerrainGeometry(
                        geom=body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name),
                        color=color,
                    )
                )
        return TerrainOutput(
            origin=np.asarray((self.size[0] / 2, self.size[1] / 2, base_height)), geometries=result
        )


@dataclass(kw_only=True)
class BoxNestedRingsTerrainCfg(SubTerrainCfg):
    """Concentric raised rings with a central platform."""

    num_rings: int = 5
    ring_width_range: tuple[float, float] = (0.3, 0.6)
    gap_range: tuple[float, float] = (0.0, 0.2)
    height_range: tuple[float, float] = (0.1, 0.4)
    platform_width: float = 1.0
    border_width: float = 0.25
    floor_depth: float = 2.0

    def function(self, difficulty: float, spec: object, rng: np.random.Generator) -> TerrainOutput:
        body = _terrain_body(spec)
        cx, cy = self.size[0] / 2, self.size[1] / 2
        outer = [self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width]
        gap = self.gap_range[0] + difficulty * (self.gap_range[1] - self.gap_range[0])
        width = self.ring_width_range[1] - difficulty * (
            self.ring_width_range[1] - self.ring_width_range[0]
        )
        result = (
            _border(
                body,
                self.size,
                (outer[0], outer[1]),
                0.5 + self.floor_depth,
                (0.5 - self.floor_depth) / 2,
                _BLUE,
            )
            if self.border_width > 0
            else []
        )
        result.append(
            _box(
                body,
                (self.size[0], self.size[1], 0.1),
                (cx, cy, -self.floor_depth - 0.05),
                (0.1, 0.1, 0.1, 1.0),
            )
        )
        for _index in range(self.num_rings):
            if outer[0] <= self.platform_width or outer[1] <= self.platform_width:
                break
            height = float(rng.uniform(*self.height_range)) * (1 + 0.5 * difficulty)
            z, box_height = (height - self.floor_depth) / 2, height + self.floor_depth
            result.extend(
                (
                    _box(
                        body,
                        (outer[0], width, box_height),
                        (cx, cy - (outer[1] - width) / 2, z),
                        _BLUE,
                    ),
                    _box(
                        body,
                        (outer[0], width, box_height),
                        (cx, cy + (outer[1] - width) / 2, z),
                        _BLUE,
                    ),
                    _box(
                        body,
                        (width, outer[1] - 2 * width, box_height),
                        (cx - (outer[0] - width) / 2, cy, z),
                        _BLUE,
                    ),
                    _box(
                        body,
                        (width, outer[1] - 2 * width, box_height),
                        (cx + (outer[0] - width) / 2, cy, z),
                        _BLUE,
                    ),
                )
            )
            outer[0] -= 2 * (width + gap)
            outer[1] -= 2 * (width + gap)
        platform_height = 0.2
        result.append(
            _box(
                body,
                (
                    max(0.01, outer[0] + 2 * gap),
                    max(0.01, outer[1] + 2 * gap),
                    platform_height + self.floor_depth,
                ),
                (cx, cy, (platform_height - self.floor_depth) / 2),
                _BLUE,
            )
        )
        return TerrainOutput(origin=np.asarray((cx, cy, platform_height)), geometries=result)


RAMP_DEG_MIN = 2.0
RAMP_DEG_MAX = 20.0


@dataclass(kw_only=True)
class FlatRampTerrainCfg(SubTerrainCfg):
    """Flat start, descending ramp, and flat runout for a slope task."""

    flat_length: float = 2.0
    ramp_length_range: tuple[float, float] = (3.0, 8.0)
    runout_length: float = 4.0
    spawn_on_ramp: float = 0.3
    deg_min: float = RAMP_DEG_MIN
    deg_max: float = RAMP_DEG_MAX
    thickness: float = 0.5

    def function(self, difficulty: float, spec: object, rng: np.random.Generator) -> TerrainOutput:
        total_max = self.flat_length + self.ramp_length_range[1] + self.runout_length
        if total_max > self.size[0]:
            raise ValueError(
                f"flat+ramp_max+runout ({total_max}) must fit in size[0] ({self.size[0]})"
            )
        body = _terrain_body(spec)
        clipped_difficulty = float(np.clip(difficulty, 0.0, 1.0))
        angle = math.radians(self.deg_min + clipped_difficulty * (self.deg_max - self.deg_min))
        width = self.size[1]
        ramp_length = float(rng.uniform(*self.ramp_length_range))
        drop = ramp_length * math.tan(angle)
        flat = _box(
            body,
            (self.flat_length, width, self.thickness),
            (self.flat_length / 2.0, 0.0, -self.thickness / 2.0),
            (0.5, 0.5, 0.5, 1.0),
        )
        surface_length = ramp_length / math.cos(angle)
        ramp_center = (
            self.flat_length + ramp_length / 2.0 - (self.thickness / 2.0) * math.sin(angle),
            0.0,
            -(drop / 2.0) - (self.thickness / 2.0) * math.cos(angle),
        )
        ramp = _box(
            body,
            (surface_length, width, self.thickness),
            ramp_center,
            (0.45, 0.55, 0.75, 1.0),
            quat=(math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0),
        )
        runout = _box(
            body,
            (self.runout_length, width, self.thickness),
            (
                self.flat_length + ramp_length + self.runout_length / 2.0,
                0.0,
                -drop - self.thickness / 2.0,
            ),
            (0.5, 0.5, 0.5, 1.0),
        )
        spawn_x = self.flat_length + self.spawn_on_ramp
        spawn_z = -self.spawn_on_ramp * math.tan(angle)
        return TerrainOutput(
            origin=np.asarray((spawn_x, 0.0, spawn_z)),
            geometries=[flat, ramp, runout],
        )


def _heightfield(
    spec: Any,
    noise: np.ndarray,
    *,
    size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
    base_thickness_ratio: float,
    z_offset: float,
    origin_z: float,
    rng: np.random.Generator,
    patches: dict[str, FlatPatchSamplingCfg] | None,
    physical_height: float | None = None,
) -> TerrainOutput:
    import mujoco

    if physical_height is None:
        minimum, maximum = int(noise.min()), int(noise.max())
        span = max(1, maximum - minimum)
        physical = (noise.astype(np.float64) - minimum) * vertical_scale
        max_height = span * vertical_scale
        normalized = ((noise - minimum) / span).astype(np.float32)
    else:
        normalized = np.asarray(noise, dtype=np.float32)
        max_height = float(physical_height)
        physical = normalized.astype(np.float64) * max_height
    field = spec.add_hfield(
        name=f"terrain_hfield_{uuid.uuid4().hex}",
        size=(size[0] / 2, size[1] / 2, max_height, max_height * base_thickness_ratio),
        nrow=noise.shape[0],
        ncol=noise.shape[1],
        userdata=normalized.flatten().tolist(),
    )
    # Keep physical heights exact; the current backend path has no tested
    # MjSpec texture transfer, so use the RGBA fallback.
    body = _terrain_body(spec)
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=field.name,
        pos=(size[0] / 2, size[1] / 2, z_offset),
    )
    flat_patches = {
        name: _find_flat_patches_from_heightfield(physical, horizontal_scale, z_offset, cfg, rng)
        for name, cfg in (patches or {}).items()
    }
    return TerrainOutput(
        origin=np.asarray((size[0] / 2, size[1] / 2, origin_z)),
        geometries=[TerrainGeometry(geom=geom, hfield=field)],
        flat_patches=flat_patches,
    )


def _fractal_noise(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    octaves: int,
    persistence: float,
    lacunarity: float,
    scale: float,
) -> np.ndarray:
    def fade(value: np.ndarray) -> np.ndarray:
        return value * value * value * (value * (value * 6 - 15) + 10)

    def gradient(hash_value: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        hash_value = hash_value % 4
        return np.where(
            hash_value == 0,
            x + y,
            np.where(hash_value == 1, x - y, np.where(hash_value == 2, -x + y, -x - y)),
        )

    def perlin(x: np.ndarray, y: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        xi, yi = x.astype(int) % 256, y.astype(int) % 256
        xf, yf = x - x.astype(int), y - y.astype(int)
        u, v = fade(xf), fade(yf)
        n00 = gradient(permutation[permutation[xi] + yi], xf, yf)
        n01 = gradient(permutation[permutation[xi] + yi + 1], xf, yf - 1)
        n11 = gradient(permutation[permutation[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = gradient(permutation[permutation[xi + 1] + yi], xf - 1, yf)
        first = n00 + u * (n10 - n00)
        second = n01 + u * (n11 - n01)
        return first + v * (second - first)

    permutation = np.arange(256, dtype=int)
    rng.shuffle(permutation)
    permutation = np.tile(permutation, 2)
    xx, yy = np.meshgrid(
        np.linspace(0, rows, rows, endpoint=False),
        np.linspace(0, cols, cols, endpoint=False),
        indexing="ij",
    )
    result = np.zeros((rows, cols))
    amplitude, frequency, total = 1.0, scale, 0.0
    for _ in range(octaves):
        result += amplitude * perlin(xx * frequency / rows, yy * frequency / cols, permutation)
        total += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return result / total


@dataclass(kw_only=True)
class HfRandomUniformTerrainCfg(SubTerrainCfg):
    """Interpolated random heightfield."""

    noise_range: tuple[float, float] = (-0.1, 0.1)
    noise_step: float = 0.005
    downsampled_scale: float | None = None
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 1.0
    border_width: float = 0.0

    def function(self, difficulty: float, spec: object, rng: np.random.Generator) -> TerrainOutput:
        del difficulty
        if self.border_width > 0 and self.border_width < self.horizontal_scale:
            raise ValueError("Heightfield border_width must be at least horizontal_scale")
        rows, cols = (
            int(self.size[0] / self.horizontal_scale),
            int(self.size[1] / self.horizontal_scale),
        )
        downsampled = (
            self.horizontal_scale if self.downsampled_scale is None else self.downsampled_scale
        )
        if downsampled < self.horizontal_scale:
            raise ValueError("downsampled_scale must be at least horizontal_scale")
        border = int(self.border_width / self.horizontal_scale)
        noise = np.zeros((rows, cols), dtype=np.int16)
        low = int(self.noise_range[0] / self.vertical_scale)
        high = int(self.noise_range[1] / self.vertical_scale)
        step = int(self.noise_step / self.vertical_scale)
        height_range = np.arange(low, high + step, step)
        if border:
            inner_rows, inner_cols = rows - 2 * border, cols - 2 * border
            inner_size = (
                inner_rows * self.horizontal_scale,
                inner_cols * self.horizontal_scale,
            )
            sample_rows = int(inner_size[0] / downsampled)
            sample_cols = int(inner_size[1] / downsampled)
            values = rng.choice(height_range, size=(sample_rows, sample_cols))
            interpolator = interpolate.RectBivariateSpline(
                np.linspace(0, inner_size[0], sample_rows),
                np.linspace(0, inner_size[1], sample_cols),
                values,
            )
            inner = np.rint(
                interpolator(
                    np.linspace(0, inner_size[0], inner_rows),
                    np.linspace(0, inner_size[1], inner_cols),
                )
            ).astype(np.int16)
            noise[border:-border, border:-border] = inner
        else:
            sample_rows = int(self.size[0] / downsampled)
            sample_cols = int(self.size[1] / downsampled)
            values = rng.choice(height_range, size=(sample_rows, sample_cols))
            interpolator = interpolate.RectBivariateSpline(
                np.linspace(0, self.size[0], sample_rows),
                np.linspace(0, self.size[1], sample_cols),
                values,
            )
            noise = np.rint(
                interpolator(
                    np.linspace(0, self.size[0], rows),
                    np.linspace(0, self.size[1], cols),
                )
            ).astype(np.int16)
        return _heightfield(
            spec,
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            z_offset=0.0,
            origin_z=sum(self.noise_range) / 2,
            rng=rng,
            patches=self.flat_patch_sampling,
        )


@dataclass(kw_only=True)
class HfWaveTerrainCfg(SubTerrainCfg):
    """Two-dimensional sinusoidal wave heightfield."""

    amplitude_range: tuple[float, float] = (0.05, 0.2)
    num_waves: int = 1
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 0.25
    border_width: float = 0.0

    def function(self, difficulty: float, spec: object, rng: np.random.Generator) -> TerrainOutput:
        if self.num_waves <= 0:
            raise ValueError("num_waves must be positive")
        if self.border_width > 0 and self.border_width < self.horizontal_scale:
            raise ValueError("Heightfield border_width must be at least horizontal_scale")
        amplitude = self.amplitude_range[0] + difficulty * (
            self.amplitude_range[1] - self.amplitude_range[0]
        )
        rows, cols = (
            int(self.size[0] / self.horizontal_scale),
            int(self.size[1] / self.horizontal_scale),
        )
        border = int(self.border_width / self.horizontal_scale)
        inner_rows, inner_cols = rows - 2 * border, cols - 2 * border
        pixels = int(0.5 * amplitude / self.vertical_scale)
        xx, yy = np.meshgrid(
            np.arange(max(inner_rows, 1)), np.arange(max(inner_cols, 1)), indexing="ij"
        )
        wave_number = 2 * np.pi * self.num_waves / max(inner_cols, 1)
        wave = pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))
        noise = np.zeros((rows, cols), dtype=np.int16)
        if border:
            noise[border:-border, border:-border] = np.rint(wave).astype(np.int16)
        else:
            noise = np.rint(wave).astype(np.int16)
        max_height = max(1, int(noise.max()) - int(noise.min())) * self.vertical_scale
        return _heightfield(
            spec,
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            z_offset=-max_height / 2,
            origin_z=0.0,
            rng=rng,
            patches=self.flat_patch_sampling,
        )


@dataclass(kw_only=True)
class HfDiscreteObstaclesTerrainCfg(SubTerrainCfg):
    """Heightfield with randomly placed pits and bumps."""

    obstacle_height_mode: Literal["choice", "fixed"] = "choice"
    obstacle_width_range: tuple[float, float] = (0.2, 0.5)
    obstacle_height_range: tuple[float, float] = (0.05, 0.2)
    num_obstacles: int = 20
    platform_width: float = 1.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 1.0
    border_width: float = 0.0
    square_obstacles: bool = False
    origin_z_offset: float = 0.0

    def function(self, difficulty: float, spec: object, rng: np.random.Generator) -> TerrainOutput:
        if self.border_width > 0 and self.border_width < self.horizontal_scale:
            raise ValueError("Heightfield border_width must be at least horizontal_scale")
        obs_height = self.obstacle_height_range[0] + difficulty * (
            self.obstacle_height_range[1] - self.obstacle_height_range[0]
        )
        border = int(self.border_width / self.horizontal_scale)
        rows = int(self.size[0] / self.horizontal_scale)
        cols = int(self.size[1] / self.horizontal_scale)
        obs_height_pixels = int(obs_height / self.vertical_scale)
        obs_width_min = int(self.obstacle_width_range[0] / self.horizontal_scale)
        obs_width_max = int(self.obstacle_width_range[1] / self.horizontal_scale)
        platform_pixels = int(self.platform_width / self.horizontal_scale)
        inner_rows = rows - 2 * border if border else rows
        inner_cols = cols - 2 * border if border else cols
        noise = np.zeros((inner_rows, inner_cols), dtype=np.int16)
        width_range = np.arange(obs_width_min, obs_width_max + 1, 4)
        if len(width_range) == 0:
            width_range = np.array([obs_width_min])
        for _ in range(self.num_obstacles):
            if self.obstacle_height_mode == "choice":
                amount = rng.choice(
                    np.array(
                        [
                            -obs_height_pixels,
                            -obs_height_pixels // 2,
                            obs_height_pixels // 2,
                            obs_height_pixels,
                        ]
                    )
                )
            else:
                amount = obs_height_pixels
            width = int(rng.choice(width_range))
            length = width if self.square_obstacles else int(rng.choice(width_range))
            x_values = np.arange(0, inner_rows, 4)
            y_values = np.arange(0, inner_cols, 4)
            if len(x_values) == 0 or len(y_values) == 0:
                continue
            x = int(rng.choice(x_values))
            y = int(rng.choice(y_values))
            noise[x : min(x + width, inner_rows), y : min(y + length, inner_cols)] = amount
        center_x, center_y = inner_rows // 2, inner_cols // 2
        half_platform = platform_pixels // 2
        noise[
            max(center_x - half_platform, 0) : min(center_x + half_platform, inner_rows),
            max(center_y - half_platform, 0) : min(center_y + half_platform, inner_cols),
        ] = 0
        if border:
            outer_noise = np.zeros((rows, cols), dtype=np.int16)
            outer_noise[border : border + inner_rows, border : border + inner_cols] = noise
            noise = outer_noise
        elevation_min = int(noise.min())
        z_offset = (
            elevation_min * self.vertical_scale if self.obstacle_height_mode == "choice" else 0.0
        )
        return _heightfield(
            spec,
            noise,
            size=self.size,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            base_thickness_ratio=self.base_thickness_ratio,
            z_offset=z_offset,
            origin_z=self.origin_z_offset,
            rng=rng,
            patches=self.flat_patch_sampling,
        )


@dataclass(kw_only=True)
class HfPerlinNoiseTerrainCfg(SubTerrainCfg):
    """Fractal Perlin-noise heightfield."""

    height_range: tuple[float, float] = (0.05, 0.2)
    octaves: int = 4
    persistence: float = 0.5
    lacunarity: float = 2.0
    scale: float = 10.0
    horizontal_scale: float = 0.1
    resolution: float = 0.05
    base_thickness_ratio: float = 1.0
    border_width: float = 0.0

    def function(self, difficulty: float, spec: object, rng: np.random.Generator) -> TerrainOutput:
        if self.border_width > 0 and self.border_width < self.horizontal_scale:
            raise ValueError("Heightfield border_width must be at least horizontal_scale")
        target = self.height_range[0] + difficulty * (self.height_range[1] - self.height_range[0])
        rows, cols = int(self.size[0] / self.resolution), int(self.size[1] / self.resolution)
        border = int(self.border_width / self.resolution)
        inner_rows, inner_cols = rows - 2 * border, cols - 2 * border
        raw = _fractal_noise(
            inner_rows if border else rows,
            inner_cols if border else cols,
            rng,
            self.octaves,
            self.persistence,
            self.lacunarity,
            self.scale * self.resolution / self.horizontal_scale,
        )
        normalized_inner = (raw - raw.min()) / max(float(raw.max() - raw.min()), 1.0e-12)
        normalized = np.zeros((rows, cols), dtype=np.float32)
        if border:
            normalized[border:-border, border:-border] = normalized_inner
        else:
            normalized = normalized_inner.astype(np.float32)
        return _heightfield(
            spec,
            normalized,
            size=self.size,
            horizontal_scale=self.resolution,
            vertical_scale=1.0,
            base_thickness_ratio=self.base_thickness_ratio,
            z_offset=0.0,
            origin_z=target,
            rng=rng,
            patches=self.flat_patch_sampling,
            physical_height=target,
        )


__all__ = [
    "BoxInvertedPyramidStairsTerrainCfg",
    "BoxRandomSpreadTerrainCfg",
    "BoxOpenStairsTerrainCfg",
    "BoxRandomStairsTerrainCfg",
    "BoxSteppingStonesTerrainCfg",
    "BoxNarrowBeamsTerrainCfg",
    "BoxTiltedGridTerrainCfg",
    "BoxNestedRingsTerrainCfg",
    "FlatRampTerrainCfg",
    "HfRandomUniformTerrainCfg",
    "HfWaveTerrainCfg",
    "HfDiscreteObstaclesTerrainCfg",
    "HfPerlinNoiseTerrainCfg",
]
