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

## Ownership

`ManagerBasedTaskEnv` is the actual environment and lifecycle owner. There is
no task runtime object or runtime factory between the environment and its
managers.

```text
TaskEnvCfg
  ├── SceneCfg / EntityCfg        model and scene composition
  ├── manager term collections    task behavior and mutation surface
  └── task configuration          task-term parameters

ManagerBasedTaskEnv
  ├── PhysicsBackend               model/data and low-level stepping
  ├── EnvironmentState              sensors, transition, manager/task data
  ├── ActionManager                delay, scaling, actuator target
  ├── CommandManager               ordered command terms and resampling
  ├── ObservationManager           ordered actor/critic observation terms
  ├── RewardManager                independently evaluated weighted terms
  ├── TerminationManager           terminal vs truncation terms
  ├── EventManager                 startup/reset/interval/step/physics callbacks
  └── CurriculumManager            optional task progression
```

Managers receive the actual `ManagerBasedTaskEnv` directly. The environment
owns the single RNG, step counter, reset/step sequencing, and all manager
instances. `PhysicsBackend` owns only simulation mechanics and does not know
about commands, observations, rewards, terminations, or task curricula.

## Lifecycle

The current scalar environment executes this order:

```text
reset:
  seed RNG → restore model defaults → sample configured randomization
  → PhysicsBackend.reset → initialize state histories
  → reset events → forward and refresh state baselines → observation

step:
  pre-physics events → action preparation → decimated physics
  → sensor/contact/history update → post-physics events
  → reward → termination → curriculum → command update
  → step events → interval events → actor observation

`env.observations()` also computes every enabled configured group. The current
velocity configuration exposes a 61D noisy actor group and a 64D privileged
critic group; the policy-facing `env.observation()` default remains the actor
group.
```

The command used to calculate a transition's reward is therefore the command
that produced that transition; command resampling happens afterward.

## Composition and mutation

Task factories start from a fresh base factory and mutate only the pieces that
make the task different. `TermCollection.add`, `replace`, and `remove` are the
supported mutation operations for observations, rewards, events, terminations,
and curricula. `CommandConfig` has the same ordered mapping/mutation surface
for command terms.

The environment does not interpret task-specific command dimensions, reward
slices, or observation offsets. Each manager derives its layout from the
configured ordered terms and validates the resulting tensor shapes. A future
task can therefore replace a velocity command with a phase or prop command,
remove velocity rewards, and add new state under `EnvironmentState.task_data`
without introducing another environment lifecycle or runtime class.

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

## Comparison with upstream

The structural correspondence is:

| Upstream | Torch | Status |
|---|---|---|
| `ManagerBasedRlEnv` | `ManagerBasedTaskEnv` | Same role; direct lifecycle owner |
| `ManagerBasedRlEnvCfg` | `TaskEnvCfg` | Same composition role; dataclass implementation |
| `SceneCfg` / entities | `SceneCfg` / entities | Same task-owned model selection |
| action/command/observation/reward/etc. managers | same manager names | Same manager graph and mutation concept |
| `tasks/*_env_cfg.py` factories | `tasks/*_env_cfg.py` factories | Same direct factory layout |
| `tasks/mdp.py` terms | `tasks/mdp.py` plus focused Torch modules | Same term boundary and ordered composition |
| registered task IDs | explicit task-name constants | Intentionally registration-free |
| MuJoCo/Warp simulation | `PhysicsBackend` with `mujoco-torch` | Intentional backend difference |

The following differences remain deliberate or incomplete and should not be
mistaken for upstream parity:

1. The Torch backend currently supports one scalar environment. Upstream is
   vectorized and manages per-environment buffers and partial resets.
2. `TaskEnvCfg` uses lightweight dataclasses rather than mjlab's config and
   manager-term classes.
3. Upstream's managers are built on richer mjlab runtime/config base classes;
   Torch uses direct dataclasses and callbacks while preserving the same
   composition boundary and lifecycle phases.
4. The velocity reward terms share a cached feature computation because their
   raw features overlap. They are still configured and weighted as independent
   manager terms; this is an implementation optimization, not a velocity-only
   reward-manager interface.
5. Torch compiles one explicit MuJoCo scene XML selected by `SceneCfg`, whereas
   upstream's scene/entity system can compose and replicate richer scene
   layouts. The Torch model layer now resolves every configured entity into a
   named `EntityView` (bodies, geoms, joints, and qpos/qvel addresses), so task
   terms do not need robot-global indices. XML composition itself remains a
   backend-specific boundary. Non-flat terrain is now a concrete scene
   generator; the scalar backend materializes one bounded representative
   obstacle, while upstream materializes a vectorized terrain grid.
6. The Torch environment does not yet implement upstream's automatic
   vectorized reset semantics or full Gym space contract.
7. Reward scaling by control `dt` is an explicit `TaskEnvCfg` option and is
   disabled for the current golden-policy contract. Upstream tasks that need
   physical timestep scaling can enable it without changing manager code.
8. The physics backend supports fixed or sampled actuator command lag and
   output-side backlash feedback. The current golden configuration keeps
   actuator lag at zero because its tracked fixture was generated with zero
   lag; future upstream-style BAM task configs can set a `(3, 6)` lag range.
9. The current repository implements the complete velocity task composition and
   the shared extension points, not placeholder bodies for every upstream task
   factory. Future task families still need their actual task terms, assets, and
   task-specific state before they can be truthfully executable.

These are the explicit remaining parity boundaries. None requires restoring a
task runtime class; they belong in the environment, managers, task terms, or
the backend state contract. In particular, adding the next scalar task variant
does not wait on vectorization: it can reuse the lifecycle and add/mutate terms,
provided its model assets and task-specific state terms are implemented.

## Model variants

`microduck_rl_torch.robot.model_variants` mirrors the upstream entity names:

- `MICRODUCK_WALK_ROBOT_CFG`
- `MICRODUCK_STANDUP_ROBOT_CFG`
- `MICRODUCK_GROUND_PICK_ROBOT_CFG`
- `MICRODUCK_WALK_ROLLERS_ROBOT_CFG`
- `MICRODUCK_WALK_BACKLASH_ROBOT_CFG`
- `MICRODUCK_BACKLASH_ROBOT_CFG`
- `MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG`

The XML and mesh assets are not copied or transformed. The model layer only
resolves them and records their capabilities (`nq`, `nv`, `nu`, backlash,
passive joints, wheels, and semantic handles).
