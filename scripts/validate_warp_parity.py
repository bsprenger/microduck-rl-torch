"""Run a paired 500-step Torch versus upstream MuJoCo-Warp parity check."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from microduck_rl_torch.envs import NominalMicroDuckEnv
from microduck_rl_torch.envs.model import load_microduck_model
from microduck_rl_torch.envs.observations import command_vector
from microduck_rl_torch.policies.huggingface import OnnxPolicy, fetch_policy
from microduck_rl_torch_verification.warp_parity import (
    WarpParityTrace,
    format_parity_table,
    interval_metrics,
    parity_passed,
    write_parity_report,
)


def _local_trace(
    *,
    policy_path: Path,
    manifest_path: Path,
    steps: int,
    seed: int,
    vx: float,
    vy: float,
    vtheta: float,
    device: str,
) -> WarpParityTrace:
    # The upstream walk task's FULL_COLLISION expression matches only the two
    # named sole geoms in robot_walk.xml. Its remaining CAD meshes are visual
    # or self-collision-only, so the mesh-to-mesh contact kernel is not part of
    # this reference rollout and should not be initialized locally.
    bundle = load_microduck_model(
        device=device,
        dtype=torch.float32,
        disable_mesh_mesh_contacts=True,
    )
    environment = NominalMicroDuckEnv(
        bundle,
        command=command_vector(
            vx=vx,
            vy=vy,
            vtheta=vtheta,
            device=bundle.device,
            dtype=bundle.dtype,
        ),
        action_delay_lag=0,
        domain_randomization=False,
    )
    policy = OnnxPolicy(policy_path, manifest_path)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    terminated: list[bool] = []
    truncated: list[bool] = []

    # Parity is a forward-only diagnostic. Inference mode is important here:
    # mujoco-torch exposes the differentiable simulator, and retaining a graph
    # across 500 policy steps would turn a small trace into an OOM failure.
    with torch.inference_mode():
        observation = environment.reset(seed=seed)
        for step in range(steps):
            observation_array = observation.detach().cpu().numpy().astype(np.float32, copy=True)
            action = policy(observation).detach().cpu().numpy().astype(np.float32, copy=True)
            result = environment.step(torch.as_tensor(action, device=bundle.device))
            snapshot = environment.snapshot()
            observations.append(observation_array)
            actions.append(action)
            qpos.append(snapshot["qpos"].detach().cpu().numpy().astype(np.float64, copy=True))
            qvel.append(snapshot["qvel"].detach().cpu().numpy().astype(np.float64, copy=True))
            terminated.append(result.terminated)
            truncated.append(result.truncated)
            if result.terminated or result.truncated:
                raise RuntimeError(f"Torch environment ended before requested step {step + 1}")
            observation = result.observation

    metadata = {
        "schema_version": 1,
        "backend": "local-microduck-rl-torch",
        "model": bundle.fingerprint(),
        "policy_path": str(policy_path.resolve()),
        "seed": seed,
        "steps": steps,
        "command": [vx, vy, vtheta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "device": str(bundle.device),
        "deterministic_events": True,
        "observation_corruption": False,
        "actuator_delay_lag": 0,
        "joint_velocity_observation_delay_lag": 1,
        "domain_randomization": False,
    }
    return WarpParityTrace(
        metadata=metadata,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        qpos=np.asarray(qpos, dtype=np.float64),
        qvel=np.asarray(qvel, dtype=np.float64),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
    )


def _run_upstream(
    *,
    upstream_python: Path,
    upstream_root: Path,
    policy_path: Path,
    steps: int,
    seed: int,
    vx: float,
    vy: float,
    vtheta: float,
    device: str,
    output_path: Path,
) -> tuple[WarpParityTrace, str]:
    runner = Path(__file__).with_name("run_upstream_warp_rollout.py").resolve()
    source_root = upstream_root.resolve() / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(source_root), environment.get("PYTHONPATH", "")) if path
    )
    command = [
        str(upstream_python.absolute()),
        str(runner),
        "--policy-path",
        str(policy_path.resolve()),
        "--output",
        str(output_path.resolve()),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--vx",
        str(vx),
        "--vy",
        str(vy),
        "--vtheta",
        str(vtheta),
        "--device",
        device,
        "--upstream-root",
        str(upstream_root.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=upstream_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if completed.returncode != 0:
        detail = output[-4000:] if output else "upstream runner exited without diagnostics"
        raise RuntimeError(
            f"Upstream Warp rollout failed with exit code {completed.returncode}:\n{detail}"
        )
    return WarpParityTrace.load(output_path), output


def _policy_paths(policy_dir: Path, policy: str, download: bool) -> tuple[Path, Path]:
    filename = policy if policy.endswith(".onnx") else f"{policy}.onnx"
    policy_path = policy_dir / filename
    manifest_path = policy_dir / "manifest.json"
    if download or not policy_path.is_file() or not manifest_path.is_file():
        artifact = fetch_policy(policy, output_dir=policy_dir)
        policy_path = artifact.policy_path
        manifest_path = artifact.manifest_path
    if not policy_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Expected policy and manifest under {policy_dir}")
    return policy_path, manifest_path


def run_check(args: argparse.Namespace) -> int:
    policy_path, manifest_path = _policy_paths(args.policy_dir, args.policy, args.download)
    thresholds = {
        "action": args.atol_action,
        "observation": args.atol_observation,
        "qpos": args.atol_qpos,
        "qvel": args.atol_qvel,
    }
    output_path = args.output.resolve()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "policy": str(policy_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "upstream_root": str(args.upstream_root.resolve()),
        "upstream_python": str(args.upstream_python.absolute()),
        "steps_requested": args.steps,
        "checkpoints": [step for step in (1, 5, 10, 25, 50, 100, 250, 500) if step <= args.steps],
        "seed": args.seed,
        "command": [
            args.vx,
            args.vy,
            args.vtheta,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "local_device": args.device,
        "upstream_device": args.upstream_device,
        "comparison": (
            "policy-driven paired rollout; observations are policy inputs, states are post-step"
        ),
    }
    metrics = []
    error: str | None = None
    passed: bool | None = None
    try:
        local_trace = _local_trace(
            policy_path=policy_path,
            manifest_path=manifest_path,
            steps=args.steps,
            seed=args.seed,
            vx=args.vx,
            vy=args.vy,
            vtheta=args.vtheta,
            device=args.device,
        )
        with tempfile.TemporaryDirectory(prefix="microduck-warp-parity-") as temporary:
            upstream_path = Path(temporary) / "warp-trace.npz"
            upstream_trace, upstream_output = _run_upstream(
                upstream_python=args.upstream_python,
                upstream_root=args.upstream_root,
                policy_path=policy_path,
                steps=args.steps,
                seed=args.seed,
                vx=args.vx,
                vy=args.vy,
                vtheta=args.vtheta,
                device=args.upstream_device,
                output_path=upstream_path,
            )
        metadata.update(
            {
                "local_trace": local_trace.metadata,
                "upstream_trace": upstream_trace.metadata,
                "upstream_runner_output": upstream_output,
            }
        )
        metrics = interval_metrics(
            local_trace,
            upstream_trace,
            checkpoints=metadata["checkpoints"],
        )
        passed = parity_passed(metrics, thresholds=thresholds, fail_on=args.fail_on)
    except Exception as exc:  # noqa: BLE001 - report the runnable failure in the artifact.
        error = str(exc)
        metadata["failure_type"] = type(exc).__name__

    write_parity_report(
        output_path,
        metadata=metadata,
        metrics=metrics,
        passed=passed,
        thresholds=thresholds,
        error=error,
    )
    if metrics:
        print(format_parity_table(metrics, passed=passed))
    if error:
        print(f"Warp parity did not complete: {error}", file=sys.stderr)
        return 2
    print(f"Report: {output_path}")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="alpha_walking")
    parser.add_argument("--policy-dir", type=Path, default=Path("artifacts/hf"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--upstream-python", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/parity/microduck-warp-500.md")
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vx", type=float, default=0.15)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vtheta", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--upstream-device", default="cpu")
    parser.add_argument("--atol-action", type=float, default=1e-5)
    parser.add_argument("--atol-observation", type=float, default=1e-3)
    parser.add_argument("--atol-qpos", type=float, default=2e-3)
    parser.add_argument("--atol-qvel", type=float, default=5e-2)
    parser.add_argument("--fail-on", choices=("none", "action", "state", "all"), default="state")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    return run_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
