# MicroDuck RL Torch: Train MicroDuck Without an NVIDIA GPU

<p align="center">
  A PyTorch-native reinforcement-learning stack for <a href="https://www.pollen-robotics.com/">Pollen Robotics</a>' MicroDuck,
  powered by <a href="https://github.com/vmoens/mujoco-torch">mujoco-torch</a>.
</p>

<p align="center">
  <a href="https://github.com/bsprenger/microduck-rl-torch/actions/workflows/ci.yml"><img src="https://github.com/bsprenger/microduck-rl-torch/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status"></a>
  <a href="https://app.codecov.io/gh/bsprenger/microduck-rl-torch"><img src="https://codecov.io/gh/bsprenger/microduck-rl-torch/branch/main/graph/badge.svg" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-orange" alt="Apache 2.0 license">
</p>

<p align="center">
  <strong>No NVIDIA GPU? No problem.</strong><br>
  <em>🎬 MicroDuck walking demo coming soon</em>
</p>

## The problem: MicroDuck RL is CUDA-first

The official MicroDuck reinforcement-learning stack is built on
[`mjlab`](https://github.com/mujocolab/mjlab), which uses
[MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html) (`mjwarp`) for
GPU-accelerated physics. That is a powerful stack—but its practical training path is built for
NVIDIA CUDA.

If you have a MacBook, Mac Studio, or a machine without an NVIDIA GPU, CUDA is not an optional
speedup. It is the missing prerequisite. You cannot simply clone the upstream project and train
your MicroDuck locally using its intended workflow.

MJX is not a straightforward solution either. MJX is a JAX implementation of MuJoCo physics,
so a PyTorch policy and trainer must interoperate with a separate JAX/XLA runtime. Even when the
JAX backend can run on a Mac, it is a poor drop-in replacement for the CUDA-first MicroDuck
training path: the backend, performance, tensor types, compilation tools, and debugging workflow
are all different.

**Most MicroDuck users should not need to buy or rent an NVIDIA GPU just to start training.**

## The solution: MicroDuck physics in PyTorch

This project brings the canonical MicroDuck model and policy interface into a PyTorch-native
workflow. It uses [`mujoco-torch`](https://github.com/vmoens/mujoco-torch), a PyTorch
implementation of the MuJoCo/MJX physics path, so simulation state and training tensors can live
in the same framework.

That makes CPU-first development possible today, including on macOS, and provides a foundation
for training at modest scale without NVIDIA hardware. CUDA remains available as an acceleration
path when you have it—but it is no longer the admission ticket.

This is not a toy replacement robot or a new model format. The project uses the official
MicroDuck MuJoCo XML, meshes, actuator ordering, observation contract, and deployment policy so
that experiments remain connected to the real robot workflow.

## Why the PyTorch-native boundary matters

MuJoCo Warp does have a PyTorch interface, and `mjlab` can share GPU memory with PyTorch without
blindly copying every tensor. The important distinction is deeper: the physics still executes
inside Warp. Torch can consume the results, but TorchInductor cannot see inside `mjwarp.step` to
fuse its physics kernels with the policy, observations, rewards, or rollout logic.

With `mujoco-torch`, the physics is expressed as PyTorch operations. That creates a path toward a
single compiler-visible rollout graph:

```text
policy → action processing → physics → observations → rewards → next state
```

When shapes and control flow are made static, the whole rollout step can potentially participate
in `torch.compile`, `torch.vmap`, and PyTorch automatic differentiation. In other words, the
goal is not merely to pass tensors between PyTorch and a simulator. The goal is to let PyTorch
own the simulation/training path end to end.

That opens the door to:

- **Training without an NVIDIA GPU:** run the development and small-scale training workflow on
  CPU, including macOS.
- **One framework:** use PyTorch tensors, compilation, device placement, profiling, and
  debugging across the policy and physics.
- **A single graph boundary:** compile physics, observations, rewards, and policy execution
  together instead of handing physics to an opaque external runtime.
- **Differentiable simulation:** differentiate simulated behavior with respect to actions,
  controls, or model parameters for system identification, trajectory optimization, and
  model-based control.
- **MuJoCo compatibility:** retain the model format and reference simulator used by the official
  MicroDuck assets and deployment workflow.

The distinction is simple:

| | Upstream `mjwarp` / `mjlab` | MicroDuck RL Torch |
| --- | --- | --- |
| Physics runtime | NVIDIA Warp kernels | PyTorch operations through `mujoco-torch` |
| Practical training path | NVIDIA CUDA GPU | CPU-first; CUDA optional for acceleration |
| PyTorch integration | Zero-copy bridge to a separate physics runtime | Native tensors throughout the physics path |
| Compiler visibility | Torch sees the boundary, not the Warp kernels | Physics can participate in a PyTorch graph |
| Graph mechanism | Warp/CUDA graph capture | Potential `torch.compile` graph across the rollout |

`mjwarp` can absolutely capture and replay its own CUDA graph. The advantage here is different:
**PyTorch can potentially compile across the physics boundary because the physics is PyTorch code.**

## What this repository provides today

The current repository establishes the trustworthy MicroDuck simulation and policy boundary
needed for the full training stack:

- the official MuJoCo XML, meshes, keyframes, and actuator ordering;
- the `new_cmd_obs` 61-element observation contract;
- the official 14-output ONNX deployment policy;
- the 50 Hz policy loop, 0.005 s physics timestep, and four-step decimation;
- native MuJoCo versus `mujoco-torch` rollout comparisons; and
- reproducible policy provenance with a resolved Hugging Face revision and SHA-256 digest.

The full upstream trainer is not claimed yet. Rewards, termination conditions, actuator backlash,
delayed response, and raw trainer checkpoint loading are separate parity layers still to be added.
The public ONNX deployment policy is the initial golden artifact because it can be executed and
verified without reconstructing a private training checkpoint.

## Installation

Install [Python 3.12](https://www.python.org/) and
[`uv`](https://docs.astral.sh/uv/). The development configuration currently uses a local
`mujoco-torch` checkout; make sure the direct dependency in `pyproject.toml` resolves to that
checkout for your machine.

From the repository root, synchronize the environment and development tools:

```bash
make install
```

This is equivalent to:

```bash
uv sync --all-groups
```

After changing the sibling simulator checkout, rebuild the installed local wheel with:

```bash
uv sync --reinstall-package mujoco-torch --all-groups
```

The optional `training` dependency group includes [TorchRL](https://pytorch.org/rl/) for the
training layer as it is added.

## Quick start

Download the official deployment policy, validate its dimensions and provenance, then run a
short native MuJoCo versus `mujoco-torch` rollout:

```bash
make verify-quick
```

The policy and manifest are written to `artifacts/hf/`, which is intentionally ignored by Git.
The default validation is a bounded CPU smoke test with contacts disabled so the high-detail mesh
scene can initialize predictably. To attempt the full contact path:

```bash
CONTACTS=enabled make validate-env
```

Render the golden policy in the Torch environment and write both a full-CAD video and a looping GIF:

```bash
make render-golden-gif
```

This writes `artifacts/render/microduck-alpha-walking.mp4` and
`artifacts/render/microduck-alpha-walking.gif`. The rollout dynamics and observations come from
`mujoco-torch`; the default renderer uses native MuJoCo only to rasterize that Torch state with
the complete CAD visual model. `make render-golden-torch` runs the same Torch environment and
writes `-torch.mp4`/`-torch.gif` outputs.

The rollout duration can be selected directly from the Makefile. Ten simulated seconds at the
50 Hz policy rate are 500 control steps:

```bash
make render-golden-gif RENDER_SECONDS=10
# or use the convenience target:
make render-golden-10s
```

`RENDER_SECONDS` is converted using the model timestep and decimation, so it remains correct if
those values change. `RENDER_STEPS` remains available for exact step-count experiments.
Rendering defaults to contacts with the floor enabled and detailed mesh-to-mesh contacts disabled;
the latter can make a fallen CAD model spend an unbounded amount of time in the convex SAT solver.
Use `RENDER_MESH_MESH_CONTACTS=enabled` when that self-contact path is specifically under test.

To exercise the pure Torch ray renderer instead, run:

```bash
make render-golden-ray RENDER_WIDTH=64 RENDER_HEIGHT=48
```

The pure Torch renderer uses the attached head camera and is substantially more demanding for the
detailed CAD meshes; ray processing is chunked to avoid an all-pixels/all-triangles allocation.
The native renderer is launched through the macOS `mjpython` compatibility wrapper. `ffmpeg` must
be installed and available on `PATH` for either video or GIF output.

For a direct CPU physics smoke test:

```python
import torch

from microduck_rl_torch.envs import NominalMicroDuckEnv
from microduck_rl_torch.envs.model import load_microduck_model

bundle = load_microduck_model(device="cpu", disable_contacts=True)
env = NominalMicroDuckEnv(bundle)
observation = env.reset()
result = env.step(torch.zeros(bundle.action_size, dtype=bundle.dtype))

print(observation.shape)         # torch.Size([61])
print(result.observation.shape)  # torch.Size([61])
```

To validate a different policy from the official manifest, set `POLICY` to its filename stem:

```bash
POLICY=alpha_stand make validate-policy
```

## Development

The `Makefile` collects repeatable commands for setup, checks, tests, artifact validation, and
packaging:

```bash
make check                         # lint, type checking, dependency hygiene
make test                          # unit and integration tests
make coverage                      # test suite with coverage reporting
make reinstall-mujoco-torch        # rebuild the local sibling simulator dependency
make validate-policy               # inspect the official ONNX policy and manifest
make validate-env STEPS=8          # short policy-driven environment rollout
make verify-quick                  # policy validation plus environment validation
make render-golden                 # golden Torch rollout to MP4
make render-golden-gif             # golden Torch rollout to MP4 and GIF
make render-golden-torch           # full-CAD Torch rollout to MP4 and GIF
make render-golden-ray             # pure mujoco-torch ray-rendered rollout
make render-golden-10s              # 10-second golden rollout to MP4 and GIF
make build                         # build a wheel and source distribution
```

## Layout

```text
assets/robot/microduck/       Pinned MuJoCo XML and robot meshes
src/microduck_rl_torch/       Environment, model, observation, and policy code
src/microduck_rl_torch_verification/
                              Native-vs-torch validation harness
scripts/                      Reproducible command-line entry points
tests/                        Contract and short-rollout tests
docs/                         Policy parity and artifact notes
```

## Credits and licensing

The robot model and simulation assets are derived from
[`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for asset attribution and redistribution terms.
Project code is released under the [Apache-2.0 license](LICENSE).
