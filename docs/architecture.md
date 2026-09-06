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
  ├── SensorManager                named semantic sensors and histories
  ├── EnvironmentState              sensors, transition, manager/task data
  ├── TaskStateManager              persistent task components and reset hooks
  ├── ActionManager                composed action terms and actuator targets
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

The environment supports one or many independent environment rows and executes
the same order for both:

```text
reset:
  seed RNG → restore model defaults → sample configured randomization
  → PhysicsBackend.reset → initialize state histories
  → SensorManager.reset → task-state/action/command manager reset
  → reset events → forward and refresh all sensor/state baselines → observation

step:
  task-state pre-physics → pre-physics events → action-term processing
  → composed actuator target → decimated physics
  → SensorManager update → task-state post-physics → post-physics events
  → reward → termination → curriculum → command update
  → step events → interval events → actor observation → automatic per-row reset

`env.observations()` also computes every enabled configured group. The current
velocity configuration exposes a 61D noisy actor group and a 64D privileged
critic group; the policy-facing `env.observation()` default remains the actor
group. With `num_envs > 1`, observations, commands, rewards, terminations,
manager state, model randomization, and partial resets carry a leading
environment dimension.
```

The command used to calculate a transition's reward is therefore the command
that produced that transition; command resampling happens afterward.

## Rendering ownership and lifecycle

Rendering is an optional environment capability, not a second simulation path.
The environment exposes the small Gymnasium-shaped contract
`render_mode=None | "rgb_array"`, `render()`, `close()`, and `metadata`. A
`RenderConfig` in `TaskEnvCfg.viewer` selects the backend, output size, and
camera. The renderer is created lazily on the first `render()` call and is
released by `env.close()`.

```python
from microduck_rl_torch.envs import ManagerBasedTaskEnv
from microduck_rl_torch.rendering import CameraConfig, RenderConfig

env = ManagerBasedTaskEnv(
    cfg,
    render_mode="rgb_array",
    render_config=RenderConfig(
        backend="mujoco",
        width=640,
        height=480,
        camera=CameraConfig(track_body="trunk_base"),
    ),
)
try:
    env.reset()
    frame = env.render()  # uint8 array shaped (height, width, 3)
finally:
    env.close()
```

The native renderer mirrors the current Torch state into a scratch native
`MjData`, runs `mj_forward`, and rasterizes the complete CAD model. The pure
Torch renderer is available for named-camera ray rendering and shares the same
environment contract, but it is intentionally narrower and does not replace
the native renderer for high-fidelity rollout artifacts. Renderers never step
physics, update observations, or own policy execution.

`render_policy_rollout` is a thin policy-and-recording adapter around this
contract. It captures post-step frames at an explicit `render_every` cadence,
requires the requested output FPS to match simulated time, streams frames to
ffmpeg, and closes the environment on success or failure. This keeps rendering
semantics deterministic and prevents video timing or encoder lifetime from
leaking into the task lifecycle. An interactive viewer can be added on the same
environment API later without changing task or policy code.

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

Actions, sensors, and persistent task state use the same composition principle:

- `ActionCfg` is an ordered mapping of `ActionTermCfg` objects. Each term owns
  its action dimension, semantic entity, processing, clipping, and actuator
  contribution. `ActionManager` concatenates policy slices and composes the
  contributions into one backend target.
- `SceneCfg.sensors` is an ordered named sensor contract. `SensorManager`
  resolves MuJoCo sensor names and semantic body/site/joint/contact selectors
  once at construction, then exposes values through `env.sensors[name]`.
  Custom stateful readers use the same reset/update contract, which is the
  extension point for backend-native raycast sensors.
- `TaskEnvCfg.task_state` contains persistent task components. A
  `TaskStateTermCfg` class is constructed once and receives explicit
  `reset(env_ids)`, `pre_physics`, `post_physics`, and `step` callbacks from
  `TaskStateManager`. Task phases and prop bookkeeping therefore do not leak
  into `PhysicsBackend` or module globals.

`SceneBuilder` now materializes a real composed MuJoCo wrapper when a task has
multiple entities and no hand-written scene XML. It includes each entity's
source XML, normalizes relative mesh roots, copies the world/terrain template,
applies entity spawn transforms, and overlays entity keyframes into the
compiled qpos vector. `EntityView` resolves per-entity bodies, geoms, sites,
joints, actuators, and qpos/qvel addresses after compilation.

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
| action/command/observation/reward/etc. managers | same manager names | Same manager graph and mutation concept; action terms are now composed |
| `tasks/*_env_cfg.py` factories | `tasks/*_env_cfg.py` factories | Same direct factory layout |
| `tasks/mdp.py` terms | `tasks/mdp.py` plus focused Torch modules | Same term boundary and ordered composition |
| registered task IDs | explicit task-name constants | Intentionally registration-free |
| MuJoCo/Warp simulation | `PhysicsBackend` with `mujoco-torch` | Intentional backend difference |

The following differences remain deliberate or incomplete and should not be
mistaken for upstream parity:

1. `TaskEnvCfg` uses lightweight dataclasses rather than mjlab's config and
   manager-term classes.
2. Upstream's managers are built on richer mjlab runtime/config base classes;
   Torch uses direct dataclasses and callbacks while preserving the same
   composition boundary and lifecycle phases.
3. The velocity reward terms share a cached feature computation because their
   raw features overlap. They are still configured and weighted as independent
   manager terms; this is an implementation optimization, not a velocity-only
   reward-manager interface.
4. Torch composes one compiled MuJoCo scene wrapper from the configured entity
   XML sources, while upstream's scene importer composes and replicates richer
   vectorized layouts. The Torch model layer resolves every configured entity
   into a named `EntityView` (bodies, geoms, sites, joints, actuators, and
   qpos/qvel addresses), and `SensorManager` uses those handles. Terrain
   generators use a typed `TerrainOutput` boundary and can be applied to both
   single- and multi-entity scenes. The current CPU Torch driver still uses an
   explicitly selectable bounded collision approximation for authored
   cylinder/ellipsoid/heightfield primitives; `collision_policy="error"`
   rejects those scenes when exact native support is required.
5. Reward scaling by control `dt` is an explicit `TaskEnvCfg` option and is
   disabled for the current golden-policy contract. Upstream tasks that need
   physical timestep scaling can enable it without changing manager code.
6. The physics backend supports fixed or sampled actuator command lag and
   output-side backlash feedback. The current golden configuration keeps
   actuator lag at zero because its tracked fixture was generated with zero
   lag; future upstream-style BAM task configs can set a `(3, 6)` lag range.
7. The current repository implements the complete velocity task composition and
   the shared extension points, not placeholder bodies for every upstream task
   factory. Future task families still need their actual task terms, assets, and
   task-specific state before they can be truthfully executable. The built-in
   velocity sensor histories and MicroDuck domain-randomization recipe are
   opt-in task state (`metadata["velocity_state"]`); generic tasks do not inherit
   those assumptions.

These are the explicit remaining parity boundaries. None requires restoring a
task runtime class; they belong in the environment, managers, task terms, or
the backend state contract. Adding the next task variant can reuse the
lifecycle and add/mutate terms, provided its model assets and task-specific
state terms are implemented.

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
