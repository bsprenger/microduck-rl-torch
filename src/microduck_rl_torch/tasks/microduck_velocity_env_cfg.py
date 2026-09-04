"""Microduck velocity task configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from microduck_rl_torch.envs.scene import (
    BoxFlatTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    TerrainCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
)
from microduck_rl_torch.envs.task_config import TaskEnvCfg
from microduck_rl_torch.robot import MICRODUCK_WALK_ROBOT_CFG

from .common_env_cfg import MicroduckRlCfg, make_velocity_env_cfg
from .names import MJLAB_VELOCITY_FLAT_MICRODUCK, MJLAB_VELOCITY_ROUGH_MICRODUCK

MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.25),
        "pyramid_stairs": BoxPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.015),
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
        "random_grid": BoxRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),
            platform_width=1.5,
        ),
        "pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.03, 0.10),
            platform_width=2.0,
            vertical_scale=0.001,
        ),
    },
    add_lights=False,
)


def _soften_terrain_contacts(spec: Any, _cfg: object | None = None) -> None:
    """Apply the Microduck rough-terrain contact policy at task composition.

    The terrain generator remains generic: other tasks may use the same
    primitive families with their own contact settings.  This is the direct
    task-specific scene ``spec_fn`` hook.
    """

    body = spec.body("terrain")
    for geom in body.geoms:
        geom.solref = (0.04, 1.0)
        geom.solimp = (0.85, 0.95, 0.001, 0.5, 2.0)


def make_microduck_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
    *,
    terrain_rows: int | None = None,
    terrain_cols: int | None = None,
) -> TaskEnvCfg:
    """Compose the Microduck velocity task from the generic velocity base."""

    if not rough and (terrain_rows is not None or terrain_cols is not None):
        raise ValueError("terrain_rows and terrain_cols require rough=True")
    if terrain_rows is not None and terrain_rows < 1:
        raise ValueError("terrain_rows must be positive")
    if terrain_cols is not None and terrain_cols < 1:
        raise ValueError("terrain_cols must be positive")

    cfg = make_velocity_env_cfg(play=play)
    cfg.task_name = MJLAB_VELOCITY_ROUGH_MICRODUCK if rough else MJLAB_VELOCITY_FLAT_MICRODUCK
    cfg.scene.entities["robot"] = MICRODUCK_WALK_ROBOT_CFG
    # Keep the task canonical: SceneBuilder composes robot_walk.xml with the
    # configured terrain generator or flat plane.
    cfg.scene.scene_xml = None
    terrain_generator = None
    if rough:
        terrain_cfg = deepcopy(MICRODUCK_ROUGH_TERRAINS_CFG)
        if play:
            # Use a compact 5x5 sample for interactive rendering so it does
            # not compile 200 cells.
            terrain_cfg.num_rows = 5
            terrain_cfg.num_cols = 5
            terrain_cfg.curriculum = False
        if terrain_rows is not None:
            terrain_cfg.num_rows = terrain_rows
        if terrain_cols is not None:
            terrain_cfg.num_cols = terrain_cols
        terrain_generator = TerrainGenerator(terrain_cfg)
    cfg.scene.terrain = TerrainCfg(
        kind="generator" if rough else "plane",
        generator=terrain_generator,
    )
    if rough:
        # Use the task's bounded solver policy as a composition hook after
        # generic
        # terrain generation has produced the terrain body.
        cfg.scene.spec_fn = _soften_terrain_contacts
        cfg.fixed_iterations = True
        cfg.solver_iterations = 30
        cfg.line_search_iterations = 50
        # Use a contact arena of 200 for rough terrain. This is a simulator
        # contract, not a terrain-generator or candidate-only setting.
        cfg.nconmax = 200
        # The backend has explicit hfield paths for every geometry pair used
        # by this task. Do not silently flatten an unsupported rough shape.
        cfg.collision_policy = "error"
    cfg.play = play
    cfg.metadata.update(
        {
            "family": "microduck_velocity",
            "rough": rough,
            # Keep authored mesh contact masks intact.  Disabling all
            # mesh-mesh pairs is a performance shortcut that removes valid
            # self/obstacle contacts and therefore changes rough-terrain
            # semantics.
            "disable_mesh_mesh_contacts": False,
            "rl_cfg": MicroduckRlCfg(),
            "domain_randomization": False if play else cfg.metadata["domain_randomization"],
        }
    )
    return cfg


__all__ = [
    "MICRODUCK_ROUGH_TERRAINS_CFG",
    "MicroduckRlCfg",
    "make_microduck_velocity_env_cfg",
]
