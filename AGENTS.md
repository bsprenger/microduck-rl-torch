# AGENTS.md

## Purpose

MicroDuck RL Torch reimplements the policy-facing MicroDuck environment with PyTorch and
`mujoco-torch`. The project exists to make the MicroDuck RL workflow practical on non-CUDA
hardware, including macOS machines using CPU or MPS, while keeping experiments connected to the
official robot model, deployment policy, and upstream environment.

The main engineering goals are:

- fast execution and a PyTorch-native path that can benefit from `torch.compile`, batching, and
  other PyTorch tooling across policy, physics, observations, and rewards; and
- near-exact behavioral parity with the Warp-based `microduck_rl` environment used upstream.

Parity includes the model and actuator ordering, BAM control behavior, timing and decimation,
reset/randomization behavior, the 61-element actor observation, rewards, termination, and the
14-output deployment-policy interface. When a backend difference cannot be made identical (for
example, contact solver details), measure and document it rather than silently changing the
contract.

Read `README.md` for the project overview and `docs/policy-parity.md` for the parity assumptions
and validation methodology.

## Repository map

- `assets/robot/microduck/`: official MicroDuck MuJoCo XML, meshes, scenes, and configuration.
- `src/microduck_rl_torch/envs/`: the core policy-facing environment. `core.py` orchestrates the
  rollout; `model.py` loads/fingerprints MuJoCo models; `actuation.py` implements BAM behavior;
  `observations.py`, `rewards.py`, and `config.py` hold the corresponding task contracts.
- `src/microduck_rl_torch/policies/`: official Hugging Face ONNX policy download, provenance,
  validation, and execution helpers.
- `src/microduck_rl_torch/rendering/`: optional rollout visualization and MP4/GIF helpers. It
  should consume environment state, not define different dynamics.
- `src/microduck_rl_torch_verification/`: native MuJoCo, Torch, golden-trajectory, and upstream
  Warp comparison utilities.
- `scripts/`: command-line workflows for fetching policies, validation, rendering, fixture
  generation, and Warp parity.
- `tests/`: unit tests for contracts plus short native/Torch/integration rollouts.
- `docs/`: focused notes on policy and environment parity.

The full upstream trainer and private training checkpoints are out of scope for the current
milestone. The optional `training` dependency group provides TorchRL for future training work.
The optional `upstream` group provides the `mjlab`/MuJoCo-Warp reference stack for direct parity
checks; the normal development path should not require it. CUDA is useful for the upstream Warp
reference and acceleration, but it is not a requirement for this project’s non-CUDA workflow.

## Working agreements

- Keep compatibility-sensitive changes explicit, focused, and covered by tests or parity reports.
- Preserve the official asset and policy interfaces unless the change is intentionally updating
  the compatibility target.
- Use `uv` and the repeatable `Makefile` commands for development.
- Keep generated downloads, renders, and parity reports in `artifacts/`; add fixtures to
  `tests/fixtures/` only when they are intentional, reproducible test inputs.

## Benchmarking scope

- The default `microduck_rl_torch` model dtype is `torch.float32` so the local environment can
  target both CPU and Apple MPS when the installed `mujoco-torch` backend supports the device.
- The initial throughput benchmark measures physics/environment stepping only, using a single
  environment and a fixed action tape. Do not include policy inference in the initial benchmark.
- The official ONNX policy adapter currently runs through CPU ONNX Runtime. Add a device-resident
  policy implementation and validate it against the official policy before reporting MPS or CUDA
  policy-inference throughput; this is a required future task.
- Keep the physics benchmark and the policy benchmark as separate measurements and artifacts so
  policy/device transfers cannot be mistaken for simulator throughput.

## Common checks

```bash
make check
make test
make verify-quick
```

For direct upstream comparison, install the optional reference group and run
`make warp-parity UPSTREAM_ROOT=/path/to/microduck_rl`. The default comparison is a deterministic
500-policy-step rollout and reports action, observation, `qpos`, and `qvel` differences.
