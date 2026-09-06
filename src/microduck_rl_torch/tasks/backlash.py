"""Backlash task overlay matching upstream task-id and model conventions."""

from __future__ import annotations

from microduck_rl_torch.envs.observations import (
    joint_position_rel_backlash,
    joint_velocity_rel_backlash,
)
from microduck_rl_torch.envs.scene import EntityCfg
from microduck_rl_torch.envs.task_config import TaskEnvCfg
from microduck_rl_torch.robot import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)


def make_backlash_variant(cfg: TaskEnvCfg, robot_cfg: EntityCfg) -> TaskEnvCfg:
    """Return a mutated copy with output-side encoder/backlash semantics.

    The upstream helper keeps the base task's manager graph and swaps only the
    robot entity plus the actuator/encoder behavior.  We preserve that
    property here: all term collections are copied, while the model variant
    and task name are the only required changes.
    """

    result = cfg.clone()
    result.task_name = result.task_name.replace("-MicroDuck", "-Backlash-MicroDuck", 1)
    result.scene.entities["robot"] = robot_cfg
    result.scene.scene_xml = robot_cfg.load_path
    result.actions.actuator_mode = "bam"
    for group_name in ("actor", "critic"):
        group = result.observations.groups.get(group_name)
        if group is None:
            continue
        position = group.terms.get("joint_position")
        if position is not None:
            position.func = joint_position_rel_backlash
        velocity = group.terms.get("joint_velocity")
        if velocity is not None:
            velocity.func = joint_velocity_rel_backlash
    result.metadata.update(
        {
            "family": f"{result.metadata.get('family', 'microduck')}_backlash",
            "backlash": True,
            "base_task_name": cfg.task_name,
            "robot_variant": robot_cfg.name,
        }
    )
    return result


def make_microduck_velocity_backlash_env_cfg(
    cfg: TaskEnvCfg,
) -> TaskEnvCfg:
    """Apply the walk backlash model to a velocity task configuration."""

    return make_backlash_variant(cfg, MICRODUCK_WALK_BACKLASH_ROBOT_CFG)


__all__ = [
    "MICRODUCK_BACKLASH_ROBOT_CFG",
    "MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG",
    "MICRODUCK_WALK_BACKLASH_ROBOT_CFG",
    "make_backlash_variant",
    "make_microduck_velocity_backlash_env_cfg",
]
