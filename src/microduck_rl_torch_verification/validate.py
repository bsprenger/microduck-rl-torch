"""Run structural, policy, and short native-vs-torch environment validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from microduck_rl_torch.envs import ManagerBasedTaskEnv, default_scene_path
from microduck_rl_torch.envs.model import load_model_bundle
from microduck_rl_torch.envs.observations import command_vector
from microduck_rl_torch.policies.huggingface import (
    OFFICIAL_POLICY_REPO,
    OnnxPolicy,
    PolicyArtifact,
    fetch_policy,
    load_policy,
    resolve_policy_filename,
)
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg

from .native import NativeMicroDuckEnv


def _local_artifact(policy_dir: Path, policy: str) -> tuple[Path, Path]:
    manifest_path = policy_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected {manifest_path}; run make fetch-golden-policy first")
    manifest = json.loads(manifest_path.read_text())
    filename = resolve_policy_filename(manifest, policy)
    policy_path = policy_dir / filename
    if not policy_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Expected {policy_path} and {manifest_path}; run make fetch-golden-policy first"
        )
    return policy_path, manifest_path


def _artifact_from_local(policy_dir: Path, policy: str) -> PolicyArtifact:
    policy_path, manifest_path = _local_artifact(policy_dir, policy)
    provenance_path = policy_dir / "download.json"
    if not provenance_path.is_file():
        # Read the old sidecar format for artifacts produced before policy-set
        # downloads were introduced.
        provenance_path = policy_dir / "artifact.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    artifact = load_policy(
        policy_path,
        manifest_path,
        repo_id=provenance.get("repo_id", OFFICIAL_POLICY_REPO),
        revision=provenance.get("revision", "local-artifact"),
    )
    policy_records = provenance.get("policies")
    if isinstance(policy_records, dict):
        if policy_path.name not in policy_records:
            raise ValueError(
                f"Policy {policy_path.name} is not part of the downloaded set in {provenance_path}"
            )
        record = policy_records[policy_path.name]
        expected_sha256 = record.get("sha256") if isinstance(record, dict) else None
        if expected_sha256 and expected_sha256 != artifact.sha256:
            raise ValueError(
                f"Policy digest does not match {provenance_path}: "
                f"expected {expected_sha256}, got {artifact.sha256}"
            )
    return artifact


def _max_abs(left: Any, right: Any) -> float:
    values = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(values))) if values.size else 0.0


def validate(
    artifact: PolicyArtifact,
    *,
    steps: int = 8,
    device: str = "cpu",
    xml_path: Path | None = None,
    fixed_iterations: bool = False,
    solver_iterations: int | None = None,
    line_search_iterations: int | None = None,
    disable_contacts: bool = False,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if artifact.policy_name != "alpha_walking.onnx":
        raise ValueError(
            "This validation entry point constructs the velocity task and only supports "
            "alpha_walking.onnx; construct the matching task environment and wire "
            "OnnxPolicy explicitly for other policies"
        )
    task_cfg = make_microduck_velocity_env_cfg()
    bundle = load_model_bundle(
        xml_path,
        entity_cfg=task_cfg.scene.entities["robot"],
        device=device,
        fixed_iterations=fixed_iterations,
        solver_iterations=solver_iterations,
        line_search_iterations=line_search_iterations,
        disable_contacts=disable_contacts,
    )
    torch_env = ManagerBasedTaskEnv(
        task_cfg,
        bundle=bundle,
        command=command_vector(vx=0.15, device=bundle.device),
        domain_randomization=False,
    )
    native_env = NativeMicroDuckEnv(
        xml_path,
        bundle=bundle,
        timestep=bundle.timestep,
        decimation=bundle.decimation,
        solver_iterations=bundle.solver_iterations,
        line_search_iterations=bundle.line_search_iterations,
        disable_contacts=not bundle.contacts_enabled,
    )
    native_env.command[:] = torch_env.command.detach().cpu().numpy()
    torch_observation = torch_env.reset().detach().cpu()
    native_observation = torch.from_numpy(native_env.reset())
    initial_observation_error = _max_abs(torch_observation, native_observation)

    policy = OnnxPolicy(artifact)
    max_observation_error = initial_observation_error
    max_state_error = 0.0
    actions: list[np.ndarray] = []
    finite = bool(torch.isfinite(torch_observation).all())
    for _ in range(steps):
        action = policy(torch_observation)
        if action.shape != (14,):
            raise RuntimeError(f"Policy returned shape {tuple(action.shape)}, expected (14,)")
        if not torch.isfinite(action).all():
            raise RuntimeError("Policy returned non-finite action")
        action_np = action.detach().cpu().numpy().astype(np.float64)
        actions.append(action_np)
        native_observation = torch.from_numpy(native_env.step(action_np))
        result = torch_env.step(action)
        torch_observation = result.observation.detach().cpu()
        max_observation_error = max(
            max_observation_error, _max_abs(torch_observation, native_observation)
        )
        native_snapshot = native_env.snapshot()
        torch_snapshot = torch_env.snapshot()
        max_state_error = max(
            max_state_error,
            _max_abs(torch_snapshot["qpos"], native_snapshot["qpos"]),
            _max_abs(torch_snapshot["qvel"], native_snapshot["qvel"]),
            _max_abs(torch_snapshot["sensordata"], native_snapshot["sensordata"]),
        )
        finite = finite and bool(torch.isfinite(torch_observation).all()) and result.info["finite"]
        if result.terminated:
            break

    result = {
        "policy_repo": artifact.repo_id,
        "policy_revision": artifact.revision,
        "policy_name": artifact.policy_name,
        "policy_sha256": artifact.sha256,
        "policy_obs_len": 61,
        "policy_action_len": 14,
        "model": bundle.fingerprint(),
        "steps_requested": steps,
        "steps_completed": len(actions),
        "finite": finite,
        "initial_observation_max_abs": initial_observation_error,
        "rollout_observation_max_abs": max_observation_error,
        "rollout_state_max_abs": max_state_error,
        "action_max_abs": max(float(np.max(np.abs(action))) for action in actions)
        if actions
        else 0.0,
        "status": "pass" if finite else "fail",
    }
    if not finite:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="alpha_walking")
    parser.add_argument("--policy-dir", type=Path, default=Path("artifacts/hf"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--xml", type=Path, default=default_scene_path())
    parser.add_argument("--fixed-iterations", action="store_true")
    parser.add_argument("--solver-iterations", type=int)
    parser.add_argument("--line-search-iterations", type=int)
    parser.add_argument(
        "--disable-contacts",
        action="store_true",
        help="Disable contacts for a fast target-engine smoke test; not a physics-parity run",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the policy if it is not already present in --policy-dir",
    )
    args = parser.parse_args(argv)
    if args.download:
        artifact = fetch_policy(args.policy, output_dir=args.policy_dir)
    else:
        artifact = _artifact_from_local(args.policy_dir, args.policy)
    print(
        json.dumps(
            validate(
                artifact,
                steps=args.steps,
                device=args.device,
                xml_path=args.xml,
                fixed_iterations=args.fixed_iterations,
                solver_iterations=args.solver_iterations,
                line_search_iterations=args.line_search_iterations,
                disable_contacts=args.disable_contacts,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
