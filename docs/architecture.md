# MicroDuck Torch task architecture

The Torch task layer mirrors the upstream `mjlab_microduck.tasks` layout while
deliberately avoiding registration side effects. A task is a direct factory
call that returns a fresh, mutable configuration:

```python
from microduck_rl_torch.envs import ManagerBasedTaskEnv
from microduck_rl_torch.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)

cfg = make_microduck_velocity_env_cfg(play=False, rough=False)
env = ManagerBasedTaskEnv(cfg)
```

The configuration is the composition root. It contains named scene entities,
terrain, action and command contracts, actor/critic observation groups,
ordered reward/termination/event/curriculum terms, and task metadata. Task
factories start from a fresh base factory and mutate only the pieces that make
the task different. `TermCollection.add`, `replace`, and `remove` are the
supported mutation operations.

The runtime separates four concerns:

```text
TaskEnvCfg
  ├── SceneCfg / EntityCfg        model and scene composition
  ├── manager term collections    task behavior and mutation surface
  └── runtime settings             device-independent task parameters

ManagerBasedTaskEnv
  ├── ModelBundle                  MuJoCo/Torch state and semantic handles
  ├── ActionManager                delay, scaling, actuator target
  ├── CommandManager               command resampling
  ├── ObservationManager           actor/critic groups
  ├── RewardManager                raw terms → configured weighted total
  ├── TerminationManager           terminal vs truncation terms
  ├── EventManager                reset/pre/post physics lifecycle
  └── CurriculumManager            optional task progression
```

`NominalMicroDuckEnv` remains available as the compatibility path used by
existing callers. The manager-based velocity task is checked against it by a
fixed action trace, and the native MuJoCo golden fixture remains independent
of both implementations.

## Model variants

`microduck_rl_torch.robot.model_variants` mirrors the upstream entity names:

- `MICRODUCK_WALK_ROBOT_CFG`
- `MICRODUCK_STANDUP_ROBOT_CFG`
- `MICRODUCK_GROUND_PICK_ROBOT_CFG`
- `MICRODUCK_WALK_ROLLERS_ROBOT_CFG`
- `MICRODUCK_WALK_BACKLASH_ROBOT_CFG`
- `MICRODUCK_BACKLASH_ROBOT_CFG`
- `MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG`

An entity declares its robot XML, optional compatibility scene wrapper,
keyframe policy, sensor requirements, and semantic contact selectors. Named
foot geoms and roller ankle subtrees are both supported. Models without a
`STAND` keyframe can use `qpos0` or a task-owned reset callback.

The XML and mesh assets are not copied or transformed. The model layer only
resolves them and records their capabilities (`nq`, `nv`, `nu`, backlash,
passive joints, wheels, and semantic handles).

## Naming

The first task uses the exact upstream identifiers and factory names:

```text
Mjlab-Velocity-Flat-MicroDuck
make_microduck_velocity_env_cfg(play=False, rough=False)
MicroduckRlCfg
```

Backlash is an overlay, not a separate environment implementation:

```python
backlash_cfg = make_backlash_variant(
    make_microduck_velocity_env_cfg(),
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)
```

This produces `Mjlab-Velocity-Flat-Backlash-MicroDuck` while preserving the
base task's manager graph and observation/action dimensions.
