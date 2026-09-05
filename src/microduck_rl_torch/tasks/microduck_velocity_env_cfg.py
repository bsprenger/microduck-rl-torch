"""Upstream-compatible MicroDuck velocity task configuration."""

from __future__ import annotations

from microduck_rl_torch.envs.scene import TerrainCfg, make_microduck_rough_scene
from microduck_rl_torch.envs.task_config import TaskEnvCfg
from microduck_rl_torch.robot import MICRODUCK_WALK_ROBOT_CFG

from .common_env_cfg import MicroduckRlCfg, make_velocity_env_cfg
from .names import MJLAB_VELOCITY_FLAT_MICRODUCK, MJLAB_VELOCITY_ROUGH_MICRODUCK


def make_microduck_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> TaskEnvCfg:
    """Compose the MicroDuck velocity task from the generic velocity base."""

    cfg = make_velocity_env_cfg(play=play)
    cfg.task_name = MJLAB_VELOCITY_ROUGH_MICRODUCK if rough else MJLAB_VELOCITY_FLAT_MICRODUCK
    cfg.scene.entities["robot"] = MICRODUCK_WALK_ROBOT_CFG
    cfg.scene.scene_xml = MICRODUCK_WALK_ROBOT_CFG.load_path
    cfg.scene.terrain = TerrainCfg(
        kind="generator" if rough else "plane",
        generator=make_microduck_rough_scene if rough else None,
        params={
            "source": "MICRODUCK_ROUGH_TERRAINS_CFG",
            "rows": 10,
            "cols": 20,
        }
        if rough
        else {},
    )
    cfg.play = play
    cfg.metadata.update(
        {
            "family": "microduck_velocity",
            "rough": rough,
            "rl_cfg": MicroduckRlCfg(),
            "domain_randomization": False if play else cfg.metadata["domain_randomization"],
        }
    )
    return cfg


__all__ = ["MicroduckRlCfg", "make_microduck_velocity_env_cfg"]
