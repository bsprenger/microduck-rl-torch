# AGENTS.md

This file contains repository-specific guidance for coding agents and contributors.
Use `README.md` for installation and usage, and `docs/policy-parity.md` and
`docs/known-limitations.md` for compatibility behavior.

## Project contract

This repository provides a PyTorch-based Microduck environment backed by `mujoco-torch`.

Preserve compatibility-sensitive behavior unless a change is intentional and documented.
This includes:

- official robot assets, model names, and model/actuator ordering;
- BAM actuation behavior;
- simulation timing and decimation;
- reset, randomization, observation, reward, and termination semantics;
- the deployment-policy contract: 61 actor observations, 14 actions, and 50 Hz control.

When backend behavior cannot be numerically identical to `microduck_rl`, document the
difference and update the relevant parity tests or report.

## Repository layout

- `assets/robot/microduck/`: robot MuJoCo models, meshes, and scenes.
- `src/microduck_rl_torch/envs/`: environment lifecycle, physics, sensors, and managers.
- `src/microduck_rl_torch/tasks/`: task configurations and task terms.
- `src/microduck_rl_torch/robot/`: robot configurations and model variants.
- `src/microduck_rl_torch/policies/`: policy download, provenance, validation, and execution.
- `src/microduck_rl_torch/rendering/`: optional environment rendering and rollout video support.
- `src/microduck_rl_torch_verification/`: native, Torch, trajectory, benchmark, and parity checks.
- `scripts/`: command-line workflows.
- `tests/`: unit, integration, and contract tests.
- `docs/`: architecture, parity, and limitation documentation.

## Working rules

- Use `uv` and the repository `Makefile` for development commands.
- Keep changes focused and preserve existing public interfaces.
- Add or update tests for compatibility-sensitive changes.
- Update documentation when behavior, supported backends, or limitations change.
- Keep generated downloads, renders, benchmarks, and reports under `artifacts/`.
- Add files under `tests/fixtures/` only when they are intentional and reproducible.

## Validation

Run before submitting changes:

```bash
make check
make test
```

For packaging changes, also run:

```bash
make archive-check
```

When network access is available:

```bash
make verify-quick
```

The optional Warp comparison requires a separate `microduck_rl` checkout:

```bash
make warp-parity MICRODUCK_RL_ROOT=/path/to/microduck_rl
```
