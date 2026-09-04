from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import pytest

from microduck_rl_torch.envs import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxNarrowBeamsTerrainCfg,
    BoxNestedRingsTerrainCfg,
    BoxOpenStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    BoxRandomSpreadTerrainCfg,
    BoxRandomStairsTerrainCfg,
    BoxSteppingStonesTerrainCfg,
    BoxTiltedGridTerrainCfg,
    FlatPatchSamplingCfg,
    FlatRampTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfPerlinNoiseTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
)


def _compile_single(
    sub_terrain: object,
    *,
    size: tuple[float, float] = (4.0, 4.0),
    difficulty: float = 0.5,
    seed: int = 17,
) -> tuple[TerrainGenerator, mujoco.MjSpec]:
    cfg = TerrainGeneratorCfg(
        size=size,
        num_rows=1,
        num_cols=1,
        difficulty_range=(difficulty, difficulty),
        seed=seed,
        sub_terrains={"terrain": sub_terrain},
    )
    generator = TerrainGenerator(cfg)
    spec = mujoco.MjSpec()
    generator.compile(spec)
    return generator, spec


@dataclass(frozen=True)
class _HFieldSnapshot:
    nrow: int
    ncol: int
    size: np.ndarray


def _single_hfield(
    sub_terrain: object,
    *,
    size: tuple[float, float] = (4.0, 4.0),
    difficulty: float = 0.5,
    seed: int = 17,
) -> tuple[TerrainGenerator, _HFieldSnapshot, np.ndarray]:
    generator, spec = _compile_single(sub_terrain, size=size, difficulty=difficulty, seed=seed)
    fields = tuple(spec.hfields)
    assert len(fields) == 1
    field = fields[0]
    grid = np.asarray(field.userdata, dtype=np.float64).reshape(field.nrow, field.ncol)
    snapshot = _HFieldSnapshot(
        nrow=field.nrow,
        ncol=field.ncol,
        size=np.asarray(field.size, dtype=np.float64).copy(),
    )
    return generator, snapshot, grid.copy()


@pytest.mark.parametrize(
    ("name", "sub_terrain", "expected_geoms", "expected_hfields"),
    (
        ("flat", BoxFlatTerrainCfg(), 1, 0),
        (
            "pyramid_stairs",
            BoxPyramidStairsTerrainCfg(
                border_width=1.0,
                step_height_range=(0.0, 0.015),
                step_width=0.15,
                platform_width=2.0,
            ),
            57,
            0,
        ),
        (
            "random_grid",
            BoxRandomGridTerrainCfg(
                grid_width=0.45,
                grid_height_range=(0.0, 0.01),
                platform_width=1.5,
            ),
            261,
            0,
        ),
        (
            "pyramid_slope",
            HfPyramidSlopedTerrainCfg(
                slope_range=(0.03, 0.10),
                platform_width=2.0,
                vertical_scale=0.001,
            ),
            1,
            1,
        ),
    ),
)
def test_each_microduck_rough_family_has_a_concrete_generator_output(
    name: str,
    sub_terrain: object,
    expected_geoms: int,
    expected_hfields: int,
) -> None:
    """Keep all four rough-terrain families on the typed generator path."""

    cfg = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        num_rows=1,
        num_cols=1,
        difficulty_range=(1.0, 1.0),
        seed=17,
        sub_terrains={name: sub_terrain},
    )
    generator = TerrainGenerator(cfg)
    spec = mujoco.MjSpec()
    generator.compile(spec)

    assert len(tuple(spec.geoms)) == expected_geoms
    assert len(tuple(spec.hfields)) == expected_hfields
    assert np.isfinite(generator.terrain_origins).all()
    assert generator.terrain_types == [[name]]


@pytest.mark.parametrize(
    "sub_terrain",
    (
        BoxInvertedPyramidStairsTerrainCfg(step_height_range=(0.05, 0.10), step_width=0.5),
        BoxRandomSpreadTerrainCfg(num_boxes=8),
        BoxOpenStairsTerrainCfg(),
        BoxRandomStairsTerrainCfg(),
        BoxSteppingStonesTerrainCfg(),
        BoxNarrowBeamsTerrainCfg(num_beams=8),
        BoxTiltedGridTerrainCfg(),
        BoxNestedRingsTerrainCfg(num_rings=3),
        HfRandomUniformTerrainCfg(),
        HfWaveTerrainCfg(),
        HfDiscreteObstaclesTerrainCfg(num_obstacles=8),
        HfPerlinNoiseTerrainCfg(),
        FlatRampTerrainCfg(ramp_length_range=(1.0, 1.0)),
    ),
)
def test_full_terrain_catalog_materializes(sub_terrain: object) -> None:
    """Every rough generator has a real Torch scene boundary."""

    cfg = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        num_rows=1,
        num_cols=1,
        difficulty_range=(0.75, 0.75),
        seed=17,
        sub_terrains={"terrain": sub_terrain},
    )
    generator = TerrainGenerator(cfg)
    spec = mujoco.MjSpec()
    generator.compile(spec)

    assert len(tuple(spec.geoms)) > 0
    assert np.isfinite(generator.terrain_origins).all()
    mujoco.MjModel.from_xml_string(spec.to_xml())


@pytest.mark.parametrize(
    "factory",
    (
        lambda: HfRandomUniformTerrainCfg(noise_range=(-0.1, 0.1)),
        lambda: HfWaveTerrainCfg(amplitude_range=(0.05, 0.2)),
        lambda: HfDiscreteObstaclesTerrainCfg(num_obstacles=8),
        lambda: HfPerlinNoiseTerrainCfg(resolution=0.1),
    ),
)
def test_heightfield_generation_is_seeded_and_has_expected_origin_semantics(factory):
    """Same seed/config reproduces the height samples and the spawn table."""

    first, field_first, grid_first = _single_hfield(factory(), seed=31)
    second, field_second, grid_second = _single_hfield(factory(), seed=31)

    np.testing.assert_array_equal(grid_first, grid_second)
    np.testing.assert_allclose(field_first.size, field_second.size)
    np.testing.assert_allclose(first.terrain_origins, second.terrain_origins)
    assert np.isfinite(grid_first).all()
    assert first.terrain_origins.shape == (1, 1, 3)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: HfRandomUniformTerrainCfg(
            noise_range=(-0.1, 0.1), horizontal_scale=0.1, border_width=0.2
        ),
        lambda: HfWaveTerrainCfg(
            amplitude_range=(0.05, 0.2), horizontal_scale=0.1, border_width=0.2
        ),
        lambda: HfDiscreteObstaclesTerrainCfg(
            num_obstacles=8, horizontal_scale=0.1, border_width=0.2
        ),
    ),
)
def test_bordered_heightfields_keep_a_constant_flat_perimeter(factory):
    _generator, field, grid = _single_hfield(factory(), size=(4.0, 4.0), seed=23)

    border = 2
    assert field.nrow == 40 and field.ncol == 40
    perimeter = np.concatenate(
        (
            grid[:border].ravel(),
            grid[-border:].ravel(),
            grid[:, :border].ravel(),
            grid[:, -border:].ravel(),
        )
    )
    assert np.allclose(perimeter, perimeter[0])


def test_perlin_border_is_zero_and_target_height_is_preserved():
    generator, field, grid = _single_hfield(
        HfPerlinNoiseTerrainCfg(height_range=(0.03, 0.17), resolution=0.1, border_width=0.2),
        size=(4.0, 4.0),
        difficulty=0.5,
        seed=23,
    )

    np.testing.assert_allclose(grid[:2], 0.0)
    np.testing.assert_allclose(grid[-2:], 0.0)
    np.testing.assert_allclose(grid[:, :2], 0.0)
    np.testing.assert_allclose(grid[:, -2:], 0.0)
    assert np.any(grid[2:-2, 2:-2] > 0.0)
    np.testing.assert_allclose(field.size[2], 0.1)
    np.testing.assert_allclose(generator.terrain_origins[0, 0], (0.0, 0.0, 0.1))


def test_wave_difficulty_changes_physical_amplitude():
    _low, low_field, _ = _single_hfield(
        HfWaveTerrainCfg(amplitude_range=(0.05, 0.2)), difficulty=0.0, seed=5
    )
    _high, high_field, _ = _single_hfield(
        HfWaveTerrainCfg(amplitude_range=(0.05, 0.2)), difficulty=1.0, seed=5
    )

    assert float(high_field.size[2]) > float(low_field.size[2])


@pytest.mark.parametrize(
    "factory",
    (
        lambda: HfRandomUniformTerrainCfg(noise_range=(-0.1, 0.1), border_width=0.05),
        lambda: HfWaveTerrainCfg(amplitude_range=(0.05, 0.2), border_width=0.05),
        lambda: HfDiscreteObstaclesTerrainCfg(num_obstacles=2, border_width=0.05),
        lambda: HfPerlinNoiseTerrainCfg(resolution=0.1, border_width=0.05),
    ),
)
def test_heightfield_border_must_match_resolution(factory):
    with pytest.raises(ValueError, match="border_width"):
        _compile_single(factory())


def test_random_uniform_downsampled_scale_has_a_valid_interpolation_grid():
    _generator, field, grid = _single_hfield(
        HfRandomUniformTerrainCfg(
            noise_range=(-0.1, 0.1), horizontal_scale=0.1, downsampled_scale=0.5
        ),
        size=(4.0, 4.0),
        seed=41,
    )

    assert (field.nrow, field.ncol) == (40, 40)
    assert grid.shape == (40, 40)
    for downsampled_scale in (0.05, 0.0):
        with pytest.raises(ValueError, match="downsampled_scale"):
            _compile_single(
                HfRandomUniformTerrainCfg(
                    noise_range=(-0.1, 0.1),
                    horizontal_scale=0.1,
                    downsampled_scale=downsampled_scale,
                )
            )


def test_heightfield_flat_patch_sampling_is_retained_in_generator_grid():
    patch_cfg = FlatPatchSamplingCfg(num_patches=3, patch_radius=0.1, max_height_diff=1.0)
    generator, _field, _grid = _single_hfield(
        HfWaveTerrainCfg(
            amplitude_range=(0.05, 0.2),
            flat_patch_sampling={"spawn": patch_cfg},
        ),
        seed=29,
    )

    patches = generator.flat_patches["spawn"]
    assert patches.shape == (1, 1, 3, 3)
    assert np.isfinite(patches).all()


def test_primitive_catalog_is_seeded_across_geometry_and_difficulty_changes_outputs():
    def geometry_arrays(
        config: object, difficulty: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _generator, spec = _compile_single(config, size=(8.0, 8.0), difficulty=difficulty, seed=7)
        geoms = tuple(spec.body("terrain").geoms)
        return (
            np.asarray([geom.pos for geom in geoms]),
            np.asarray([geom.size for geom in geoms]),
            np.asarray([geom.quat for geom in geoms]),
        )

    config = BoxRandomStairsTerrainCfg(step_width=0.8)
    first = geometry_arrays(config, 0.6)
    second = geometry_arrays(BoxRandomStairsTerrainCfg(step_width=0.8), 0.6)
    for first_array, second_array in zip(first, second, strict=True):
        np.testing.assert_allclose(first_array, second_array)

    low = geometry_arrays(BoxRandomStairsTerrainCfg(step_width=0.8), 0.0)[1]
    high = geometry_arrays(BoxRandomStairsTerrainCfg(step_width=0.8), 1.0)[1]
    assert float(high[:, 2].max()) > float(low[:, 2].max())
