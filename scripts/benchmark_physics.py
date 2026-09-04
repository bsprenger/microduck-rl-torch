#!/usr/bin/env python3
"""Benchmark single-environment physics throughput across MicroDuck backends.

The timed region contains only direct simulator physics steps. It intentionally
does not run policy inference, observations, rewards, termination checks,
rendering, logging, or host-side state copies. Each result is a repeated block
after an untimed warmup, with accelerator synchronization around the timer.

Examples::

    uv run --group benchmark python scripts/benchmark_physics.py \
        --backend both --devices cpu,mps --upstream-root ../microduck_rl \
        --steps 500 --warmup-steps 50 --repeats 7 \
        --output artifacts/benchmarks/physics-single-env \
        --readme-graph docs/assets/microduck-physics-throughput.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from microduck_rl_torch_verification.benchmark import measure_repeated, summarize_timings


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _device_type(device: str) -> str:
    return device.split(":", 1)[0].lower()


def _torch_synchronize(device: Any) -> None:
    import torch

    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "mps":
        torch.mps.synchronize()


def _unsupported(backend: str, device: str, reason: str) -> dict[str, Any]:
    labels = {
        "microduck_rl_torch": "mujoco-torch",
        "upstream_mujoco_warp": "upstream MuJoCo-Warp",
    }
    return {
        "backend": backend,
        "backend_label": labels.get(backend, backend),
        "device": device,
        "status": "unsupported",
        "reason": reason,
    }


def _failed(backend: str, device: str, error: BaseException) -> dict[str, Any]:
    labels = {
        "microduck_rl_torch": "mujoco-torch",
        "upstream_mujoco_warp": "upstream MuJoCo-Warp",
    }
    return {
        "backend": backend,
        "backend_label": labels.get(backend, backend),
        "device": device,
        "status": "failed",
        "reason": f"{type(error).__name__}: {error}",
    }


def _make_control_tape(
    *, steps: int, seed: int, action_size: int = 14, amplitude: float = 0.0
) -> np.ndarray:
    # The default is a zero-load tape. It avoids making throughput depend on a
    # particular policy-like trajectory and keeps float32 accelerator runs
    # numerically stable while the robot settles into contact. A bounded
    # deterministic torque tape remains available for stress experiments.
    if amplitude < 0:
        raise ValueError("control amplitude must be non-negative")
    rng = np.random.default_rng(seed)
    return rng.uniform(-amplitude, amplitude, size=(steps, action_size)).astype(np.float32)


def _run_local(
    *,
    device: str,
    controls_np: np.ndarray,
    warmup_steps: int,
    repeats: int,
    solver_iterations: int,
    line_search_iterations: int,
    mesh_mesh_contacts: bool,
) -> dict[str, Any]:
    import mujoco
    import torch

    mujoco_api: Any = mujoco

    if _device_type(device) == "mps" and not torch.backends.mps.is_available():
        return _unsupported(
            "microduck_rl_torch",
            device,
            "torch.backends.mps.is_available() is false on this host",
        )
    if _device_type(device) == "cuda" and not torch.cuda.is_available():
        return _unsupported("microduck_rl_torch", device, "torch.cuda.is_available() is false")

    from microduck_rl_torch.envs.model import load_microduck_model

    bundle = load_microduck_model(
        device=device,
        dtype=torch.float32,
        fixed_iterations=True,
        solver_iterations=solver_iterations,
        line_search_iterations=line_search_iterations,
        disable_mesh_mesh_contacts=not mesh_mesh_contacts,
    )
    controls = torch.as_tensor(controls_np, device=bundle.device, dtype=bundle.dtype)
    steps = int(controls.shape[0])
    data: Any = None

    def prepare_sample() -> None:
        nonlocal data
        # Allocation and the initial forward are deliberately outside the
        # timed block. This measures steady-state stepping, not reset/setup.
        data = bundle.new_data()
        for index in range(warmup_steps):
            data.ctrl.copy_(controls[index % steps])
            data = _mujoco_torch_step(bundle.torch_model, data, bundle.fixed_iterations)
        # Warmup is for backend caches, not for changing the measured initial
        # condition. Recreate the standing state before every timed sample;
        # this also avoids carrying an MPS float32 contact state into timing.
        data = bundle.new_data()

    def run_sample() -> None:
        nonlocal data
        for index in range(steps):
            data.ctrl.copy_(controls[(warmup_steps + index) % steps])
            data = _mujoco_torch_step(bundle.torch_model, data, bundle.fixed_iterations)

    with torch.inference_mode():
        measured = measure_repeated(
            run_sample,
            prepare_sample=prepare_sample,
            synchronize=lambda: _torch_synchronize(bundle.device),
            repeats=repeats,
        )
    active = (bundle.native_model.geom_contype != 0) & (bundle.native_model.geom_conaffinity != 0)
    mesh_type = int(mujoco_api.mjtGeom.mjGEOM_MESH)
    return {
        "backend": "microduck_rl_torch",
        "backend_label": "mujoco-torch",
        "device": str(bundle.device),
        "status": "ok",
        "semantics": "direct mujoco_torch.step; one single-environment physics timestep",
        "collision": {
            "active_geom_count": int(active.sum()),
            "active_mesh_geom_count": int(
                ((bundle.native_model.geom_type == mesh_type) & active).sum()
            ),
            "allocated_contact_count": int(bundle.torch_model.collision_total_contacts_py),
            "mesh_mesh_contacts": mesh_mesh_contacts,
        },
        "model": bundle.fingerprint(),
        "stats": summarize_timings(measured["seconds"], steps=steps),
    }


def _mujoco_torch_step(model: Any, data: Any, fixed_iterations: bool) -> Any:
    import mujoco_torch

    return mujoco_torch.step(model, data, fixed_iterations=fixed_iterations)


def _make_upstream_config(
    *,
    steps: int,
    solver_iterations: int,
    line_search_iterations: int,
    mesh_mesh_contacts: bool,
) -> Any:
    """Build the same deterministic flat single-env setup used by parity."""

    config_module = importlib.import_module("mjlab_microduck.tasks.microduck_velocity_env_cfg")
    config = config_module.make_microduck_velocity_env_cfg(play=False, rough=False)
    config.scene.num_envs = 1
    config.episode_length_s = max(float(steps) * 0.005 + 1.0, 10.0)
    config.observations["actor"].enable_corruption = False
    config.sim.mujoco.iterations = solver_iterations
    config.sim.mujoco.ls_iterations = line_search_iterations

    for name in tuple(config.events):
        if name not in {"expand_bam_friction_fields", "reset_action_history", "reset_base"}:
            config.events.pop(name)
    for name in tuple(config.curriculum):
        config.curriculum.pop(name)
    reset_base = config.events.get("reset_base")
    if reset_base is not None:
        pose_range = reset_base.params.get("pose_range", {})
        for key in tuple(pose_range):
            pose_range[key] = (0.12, 0.12) if key == "z" else (0.0, 0.0)
        velocity_range = reset_base.params.get("velocity_range", {})
        for key in tuple(velocity_range):
            velocity_range[key] = (0.0, 0.0)

    robot = deepcopy(config.scene.entities["robot"])
    if not mesh_mesh_contacts:
        # The walk XML uses mask (2, 2) for the three anonymous self-collision
        # mesh geoms and (1, 1) for the two named foot meshes. Disable only the
        # former so the benchmark retains plane-foot contacts while removing
        # mesh-mesh narrowphase work in the Warp model.
        import mujoco

        mujoco_api: Any = mujoco
        base_spec_fn = robot.spec_fn

        def spec_without_mesh_mesh_contacts() -> Any:
            spec = base_spec_fn()
            for geom in spec.geoms:
                if (
                    geom.type == mujoco_api.mjtGeom.mjGEOM_MESH
                    and int(geom.contype) == 2
                    and int(geom.conaffinity) == 2
                ):
                    geom.contype = 0
                    geom.conaffinity = 0
            return spec

        robot.spec_fn = spec_without_mesh_mesh_contacts
    for actuator_group in robot.articulation.actuators:
        actuator_group.delay_min_lag = 0
        actuator_group.delay_max_lag = 0
        if hasattr(actuator_group, "vin_range"):
            actuator_group.vin_range = (7.4, 7.4)
        if hasattr(actuator_group, "vin_drop_gain_range"):
            actuator_group.vin_drop_gain_range = (0.1, 0.1)
    config.scene.entities["robot"] = robot
    for command in config.commands.values():
        if hasattr(command, "resampling_time_range"):
            command.resampling_time_range = (1.0e6, 1.0e6)
    return config


def _warp_synchronize(device: Any) -> None:
    import warp as wp

    if getattr(device, "is_cuda", False):
        wp.synchronize_device(device)


def _run_upstream(
    *,
    device: str,
    controls_np: np.ndarray,
    warmup_steps: int,
    repeats: int,
    solver_iterations: int,
    line_search_iterations: int,
    mesh_mesh_contacts: bool,
    seed: int,
    upstream_root: Path,
) -> dict[str, Any]:
    if _device_type(device) == "mps":
        return _unsupported(
            "upstream_mujoco_warp",
            device,
            "MuJoCo-Warp uses Warp, which has no Apple MPS backend",
        )
    if not upstream_root.is_dir():
        return _failed(
            "upstream_mujoco_warp",
            device,
            FileNotFoundError(upstream_root),
        )

    import mujoco
    import torch

    mujoco_api: Any = mujoco

    source_root = upstream_root.resolve() / "src"
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    manager_module = importlib.import_module("mjlab.envs")
    env_class = manager_module.ManagerBasedRlEnv
    config = _make_upstream_config(
        steps=int(controls_np.shape[0]),
        solver_iterations=solver_iterations,
        line_search_iterations=line_search_iterations,
        mesh_mesh_contacts=mesh_mesh_contacts,
    )
    env = env_class(cfg=config, device=device)
    controls = torch.as_tensor(controls_np, device=env.device, dtype=torch.float32)
    steps = int(controls.shape[0])
    initial_qpos: torch.Tensor
    initial_qvel: torch.Tensor
    data = env.sim.data

    try:
        env.reset(seed=seed)
        initial_qpos = data.qpos.clone()
        initial_qvel = data.qvel.clone()

        def prepare_sample() -> None:
            env.sim.reset()
            data.qpos.copy_(initial_qpos)
            data.qvel.copy_(initial_qvel)
            data.ctrl.zero_()
            env.sim.forward()
            for index in range(warmup_steps):
                data.ctrl[0].copy_(controls[index % steps])
                env.sim.step()
            env.sim.reset()
            data.qpos.copy_(initial_qpos)
            data.qvel.copy_(initial_qvel)
            data.ctrl.zero_()
            env.sim.forward()

        def run_sample() -> None:
            for index in range(steps):
                data.ctrl[0].copy_(controls[(warmup_steps + index) % steps])
                env.sim.step()

        measured = measure_repeated(
            run_sample,
            prepare_sample=prepare_sample,
            synchronize=lambda: _warp_synchronize(env.sim.wp_device),
            repeats=repeats,
        )
        return {
            "backend": "upstream_mujoco_warp",
            "backend_label": "upstream MuJoCo-Warp",
            "device": device,
            "status": "ok",
            "semantics": "direct env.sim.step; one single-environment physics timestep",
            "collision": {
                "active_geom_count": int(
                    (
                        (env.sim.mj_model.geom_contype != 0)
                        & (env.sim.mj_model.geom_conaffinity != 0)
                    ).sum()
                ),
                "active_mesh_geom_count": int(
                    (
                        (env.sim.mj_model.geom_type == int(mujoco_api.mjtGeom.mjGEOM_MESH))
                        & (env.sim.mj_model.geom_contype != 0)
                        & (env.sim.mj_model.geom_conaffinity != 0)
                    ).sum()
                ),
                "configured_contact_capacity": int(config.sim.nconmax),
                "mesh_mesh_contacts": mesh_mesh_contacts,
            },
            "model": {
                "timestep": float(env.physics_dt),
                "decimation": int(getattr(env.cfg, "decimation", 4)),
                "solver_iterations": solver_iterations,
                "line_search_iterations": line_search_iterations,
                "num_envs": 1,
            },
            "stats": summarize_timings(measured["seconds"], steps=steps),
        }
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def _plot_benchmark(payload: dict[str, Any], png_path: Path, svg_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = payload["results"]
    labels = [f"{row.get('backend_label', row['backend'])}\n{row['device']}" for row in results]
    x = np.arange(len(results), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(7.0, 2.0 * len(results)), 4.8))
    successful = False
    for position, row in zip(x, results, strict=True):
        if row["status"] != "ok":
            ax.text(
                position,
                0.02,
                row["status"],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
                color="0.35",
            )
            continue
        successful = True
        stats = row["stats"]
        median = float(stats["median_steps_per_second"])
        low = float(stats["p05_steps_per_second"])
        high = float(stats["p95_steps_per_second"])
        ax.errorbar(
            position,
            median,
            yerr=[[median - low], [high - median]],
            fmt="o",
            capsize=5,
            color="#2166ac",
            markersize=8,
            label="median; p05–p95" if position == x[0] else None,
        )
        samples = np.asarray(stats["seconds"], dtype=np.float64)
        sample_speeds = payload["config"]["steps"] / samples
        jitter = np.linspace(-0.10, 0.10, len(sample_speeds))
        ax.scatter(position + jitter, sample_speeds, s=18, color="#67a9cf", alpha=0.8)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Physics steps / second")
    ax.set_title("MicroDuck physics throughput — one environment")
    ax.grid(axis="y", alpha=0.25)
    if successful:
        speeds = [
            float(row["stats"]["median_steps_per_second"])
            for row in results
            if row["status"] == "ok"
        ]
        # Device comparisons can span orders of magnitude (as CPU vs MPS does
        # on this host), so a log axis keeps every successful cell legible.
        if max(speeds) / min(speeds) > 20:
            ax.set_yscale("log")
            ax.set_ylabel("Physics steps / second (log scale)")
        else:
            ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "No supported benchmark results",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    fig.text(
        0.01,
        0.01,
        "Timed region: direct backend physics only; points are repeats; whiskers are p05–p95.",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)


def _write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    with (output_dir / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "backend",
                "device",
                "status",
                "repeat",
                "seconds",
                "steps_per_second",
                "reason",
            ],
        )
        writer.writeheader()
        for row in payload["results"]:
            stats = row.get("stats")
            if not stats:
                writer.writerow(
                    {
                        **{key: row.get(key, "") for key in writer.fieldnames},
                        "reason": row.get("reason", ""),
                    }
                )
                continue
            for repeat, seconds in enumerate(stats["seconds"], start=1):
                writer.writerow(
                    {
                        "backend": row["backend"],
                        "device": row["device"],
                        "status": row["status"],
                        "repeat": repeat,
                        "seconds": seconds,
                        "steps_per_second": payload["config"]["steps"] / seconds,
                        "reason": "",
                    }
                )


def _print_summary(payload: dict[str, Any]) -> None:
    print("\nPhysics benchmark summary (median; p05–p95 steps/s)")
    print("backend                    device   status        result")
    print("-------------------------  -------  ------------  ----------------")
    for row in payload["results"]:
        if row["status"] == "ok":
            stats = row["stats"]
            result = (
                f"{stats['median_steps_per_second']:.2f}; "
                f"{stats['p05_steps_per_second']:.2f}–{stats['p95_steps_per_second']:.2f}"
            )
        else:
            result = row["reason"]
        print(f"{row['backend']:<27} {row['device']:<8} {row['status']:<12} {result}")


def _run_one(
    *, backend: str, device: str, args: argparse.Namespace, controls_np: np.ndarray
) -> dict[str, Any]:
    try:
        if backend == "local":
            return _run_local(
                device=device,
                controls_np=controls_np,
                warmup_steps=args.warmup_steps,
                repeats=args.repeats,
                solver_iterations=args.solver_iterations,
                line_search_iterations=args.line_search_iterations,
                mesh_mesh_contacts=args.mesh_mesh_contacts == "enabled",
            )
        return _run_upstream(
            device=device,
            controls_np=controls_np,
            warmup_steps=args.warmup_steps,
            repeats=args.repeats,
            solver_iterations=args.solver_iterations,
            line_search_iterations=args.line_search_iterations,
            mesh_mesh_contacts=args.mesh_mesh_contacts == "enabled",
            seed=args.seed,
            upstream_root=args.upstream_root,
        )
    except Exception as error:  # Keep other platform/backend cells runnable.
        return _failed(
            "microduck_rl_torch" if backend == "local" else "upstream_mujoco_warp",
            device,
            error,
        )


def _run_isolated(
    *,
    backend: str,
    device: str,
    args: argparse.Namespace,
    worker_output: Path,
) -> dict[str, Any]:
    """Run one backend/device cell in a fresh process.

    ``mujoco-torch`` caches constants by device globally. Running CPU and MPS
    in one interpreter can therefore contaminate the second device's cache;
    process isolation is both safer and closer to how a cross-platform CI
    benchmark will be collected.
    """

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-output",
        str(worker_output),
        "--backend",
        backend,
        "--devices",
        device,
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--control-amplitude",
        str(args.control_amplitude),
        "--solver-iterations",
        str(args.solver_iterations),
        "--line-search-iterations",
        str(args.line_search_iterations),
        "--mesh-mesh-contacts",
        args.mesh_mesh_contacts,
        "--upstream-root",
        str(args.upstream_root),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if worker_output.is_file():
        return json.loads(worker_output.read_text())
    stderr_lines = completed.stderr.strip().splitlines()
    if stderr_lines:
        detail = (
            f"worker exited with status {completed.returncode}; "
            f"stderr tail: {' | '.join(stderr_lines[-4:])}"
        )
    else:
        detail = f"worker exited with status {completed.returncode}"
    return _failed(
        "microduck_rl_torch" if backend == "local" else "upstream_mujoco_warp",
        device,
        RuntimeError(detail),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "upstream", "both"), default="both")
    parser.add_argument(
        "--devices", default="cpu", help="Comma-separated devices, e.g. cpu,mps,cuda:0"
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--control-amplitude",
        type=float,
        default=0.0,
        help="Uniform direct-motor control half-range; zero is the stable default",
    )
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--line-search-iterations", type=int, default=4)
    parser.add_argument(
        "--mesh-mesh-contacts",
        choices=("enabled", "disabled"),
        default="disabled",
        help="Enable or disable detailed mesh-mesh collision work in both backends",
    )
    parser.add_argument("--upstream-root", type=Path, default=Path("../microduck_rl"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmarks/physics-single-env")
    )
    parser.add_argument(
        "--readme-graph",
        type=Path,
        help="Also write the PNG/SVG graph at this tracked README asset path",
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.warmup_steps < 0 or args.repeats < 1:
        parser.error("steps and repeats must be positive; warmup-steps cannot be negative")
    if args.solver_iterations < 1 or args.line_search_iterations < 1:
        parser.error("solver iteration counts must be positive")

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        parser.error("--devices must contain at least one device")
    if args.control_amplitude < 0:
        parser.error("control-amplitude must be non-negative")
    controls_np = _make_control_tape(
        steps=args.steps,
        seed=args.seed,
        amplitude=args.control_amplitude,
    )
    backends = ("local", "upstream") if args.backend == "both" else (args.backend,)

    if args.worker_output is not None:
        result = _run_one(
            backend=backends[0],
            device=devices[0],
            args=args,
            controls_np=controls_np,
        )
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(json.dumps(result, sort_keys=True, default=str) + "\n")
        return 0 if result["status"] in {"ok", "unsupported"} else 1

    results: list[dict[str, Any]] = []
    worker_dir = args.output / ".workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_index = 0
    for backend in backends:
        for device in devices:
            worker_output = (
                worker_dir / f"{worker_index:03d}-{backend}-{device.replace(':', '_')}.json"
            )
            results.append(
                _run_isolated(
                    backend=backend,
                    device=device,
                    args=args,
                    worker_output=worker_output,
                )
            )
            worker_index += 1

    payload = {
        "schema_version": 1,
        "benchmark": "microduck-physics-single-environment",
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "repeats": args.repeats,
            "seed": args.seed,
            "control_amplitude": args.control_amplitude,
            "dtype": "float32",
            "solver_iterations": args.solver_iterations,
            "line_search_iterations": args.line_search_iterations,
            "mesh_mesh_contacts": args.mesh_mesh_contacts,
            "num_envs": 1,
            "policy_included": False,
            "timed_work": "direct physics step only",
        },
        "action_tape": {
            "kind": "bounded_uniform_direct_motor_control",
            "shape": list(controls_np.shape),
            "sha256": hashlib.sha256(controls_np.tobytes()).hexdigest(),
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": _package_version("torch"),
            "mujoco": _package_version("mujoco"),
            "mujoco_torch": _package_version("mujoco-torch"),
            "warp": _package_version("warp-lang"),
        },
        "upstream_root": str(args.upstream_root.resolve()),
        "results": results,
    }
    _write_outputs(payload, args.output)
    _plot_benchmark(payload, args.output / "throughput.png", args.output / "throughput.svg")
    if args.readme_graph is not None:
        _plot_benchmark(payload, args.readme_graph, args.readme_graph.with_suffix(".svg"))
    _print_summary(payload)
    print(f"\nArtifacts: {args.output.resolve()}")
    return 0 if any(row["status"] == "ok" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
