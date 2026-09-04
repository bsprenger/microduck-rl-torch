"""Fetch, inspect, and execute official MicroDuck ONNX policy artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import HfApi, hf_hub_download

OFFICIAL_POLICY_REPO = "pollen-robotics/microduck-policies"
EXPECTED_ROBOT = "microduck"
EXPECTED_OBS_LEN = 61
EXPECTED_ACTION_LEN = 14
EXPECTED_CONTROL_HZ = 50
OFFICIAL_GOLDEN_SHA256 = {
    "alpha_walking.onnx": "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_filename(policy: str) -> str:
    filename = policy if policy.endswith(".onnx") else f"{policy}.onnx"
    if Path(filename).name != filename:
        raise ValueError(f"Policy must be a repository filename, got {policy!r}")
    return filename


def _policy_entry(manifest: dict[str, Any], filename: str) -> dict[str, Any]:
    """Find a policy record across the manifest's supported list/dict shapes."""

    stem = Path(filename).stem
    candidates = manifest.get("policies", manifest.get("policy", []))
    if isinstance(candidates, dict):
        candidates = [candidates.get(stem, candidates.get(filename, candidates))]
    if not isinstance(candidates, list):
        candidates = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        names = {
            str(candidate.get(key, "")) for key in ("name", "id", "filename", "file", "policy")
        }
        if filename in names or stem in names:
            return candidate
    raise ValueError(f"{filename} is not declared in the Hugging Face policy manifest")


def _declared(entry: dict[str, Any], manifest: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in entry:
            return entry[key]
        if key in manifest:
            return manifest[key]
    return None


def _graph_shape(value: onnx.ValueInfoProto) -> list[int | None]:
    dimensions = value.type.tensor_type.shape.dim
    result: list[int | None] = []
    for dimension in dimensions:
        result.append(dimension.dim_value if dimension.HasField("dim_value") else None)
    return result


def validate_policy_artifact(
    policy_path: Path,
    manifest_path: Path,
    *,
    filename: str | None = None,
) -> dict[str, Any]:
    """Validate manifest metadata and the ONNX graph dimensions."""

    policy_path = policy_path.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Policy manifest must contain a JSON object")
    filename = filename or policy_path.name
    entry = _policy_entry(manifest, filename)

    robot = _declared(entry, manifest, "robot", "robot_name")
    robot_name = robot.get("model") if isinstance(robot, dict) else robot
    if robot_name is not None and robot_name != EXPECTED_ROBOT:
        raise ValueError(f"Expected robot {EXPECTED_ROBOT!r}, got {robot_name!r}")
    obs_len = _declared(entry, manifest, "obs_len", "observation_dim", "obs_dim")
    action_len = _declared(entry, manifest, "action_len", "action_dim")
    control_hz = _declared(entry, manifest, "control_hz", "frequency_hz", "hz")
    if control_hz is None and isinstance(robot, dict):
        control_hz = robot.get("control_hz")
    if obs_len is not None and int(obs_len) != EXPECTED_OBS_LEN:
        raise ValueError(f"Expected {EXPECTED_OBS_LEN} observations, got {obs_len}")
    if action_len is not None and int(action_len) != EXPECTED_ACTION_LEN:
        raise ValueError(f"Expected {EXPECTED_ACTION_LEN} actions, got {action_len}")
    if control_hz is not None and int(control_hz) != EXPECTED_CONTROL_HZ:
        raise ValueError(f"Expected {EXPECTED_CONTROL_HZ} Hz, got {control_hz}")

    model = onnx.load(str(policy_path))
    onnx.checker.check_model(model)
    if not model.graph.input or not model.graph.output:
        raise ValueError("ONNX graph has no input or output")
    input_shape = _graph_shape(model.graph.input[0])
    output_shape = _graph_shape(model.graph.output[0])
    if not input_shape or input_shape[-1] != EXPECTED_OBS_LEN:
        raise ValueError(f"Expected ONNX input [..., {EXPECTED_OBS_LEN}], got {input_shape}")
    if not output_shape or output_shape[-1] != EXPECTED_ACTION_LEN:
        raise ValueError(f"Expected ONNX output [..., {EXPECTED_ACTION_LEN}], got {output_shape}")
    return {
        "filename": filename,
        "robot": robot_name,
        "obs_len": EXPECTED_OBS_LEN,
        "action_len": EXPECTED_ACTION_LEN,
        "control_hz": EXPECTED_CONTROL_HZ,
        "input_name": model.graph.input[0].name,
        "output_name": model.graph.output[0].name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "sha256": sha256_file(policy_path),
    }


@dataclass(frozen=True)
class PolicyArtifact:
    """A downloaded and validated policy plus its provenance."""

    repo_id: str
    revision: str
    policy_name: str
    policy_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    sha256: str
    input_name: str
    output_name: str

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy_path"] = str(self.policy_path)
        data["manifest_path"] = str(self.manifest_path)
        data["manifest"] = str(self.manifest_path)
        data["sha256"] = self.sha256
        return data


def fetch_policy(
    policy: str = "alpha_walking",
    *,
    repo_id: str = OFFICIAL_POLICY_REPO,
    revision: str | None = None,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
) -> PolicyArtifact:
    """Download one official policy at a resolved revision and validate it."""

    filename = _policy_filename(policy)
    api = HfApi()
    model_info = api.model_info(repo_id=repo_id, revision=revision)
    resolved_revision = model_info.sha
    if not resolved_revision:
        raise RuntimeError(f"Hugging Face did not return a commit revision for {repo_id}")
    download_kwargs: dict[str, Any] = {"repo_id": repo_id, "revision": resolved_revision}
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(cache_dir)
    manifest_cache = Path(hf_hub_download(filename="manifest.json", **download_kwargs))
    policy_cache = Path(hf_hub_download(filename=filename, **download_kwargs))

    destination = output_dir or policy_cache.parent
    destination.mkdir(parents=True, exist_ok=True)
    if output_dir is not None:
        policy_path = destination / filename
        manifest_path = destination / "manifest.json"
        shutil.copy2(policy_cache, policy_path)
        shutil.copy2(manifest_cache, manifest_path)
    else:
        policy_path = policy_cache
        manifest_path = manifest_cache
    manifest = json.loads(manifest_path.read_text())
    graph_metadata = validate_policy_artifact(policy_path, manifest_path, filename=filename)
    expected_sha256 = (
        OFFICIAL_GOLDEN_SHA256.get(filename) if repo_id == OFFICIAL_POLICY_REPO else None
    )
    if expected_sha256 is not None and graph_metadata["sha256"] != expected_sha256:
        raise ValueError(
            f"Golden digest drift for {filename}: expected {expected_sha256}, "
            f"got {graph_metadata['sha256']}"
        )
    artifact = PolicyArtifact(
        repo_id=repo_id,
        revision=resolved_revision,
        policy_name=filename,
        policy_path=policy_path,
        manifest_path=manifest_path,
        manifest=manifest,
        sha256=graph_metadata["sha256"],
        input_name=graph_metadata["input_name"],
        output_name=graph_metadata["output_name"],
    )
    if output_dir is not None:
        (destination / "artifact.json").write_text(json.dumps(artifact.metadata(), indent=2) + "\n")
    return artifact


class OnnxPolicy:
    """CPU ONNX Runtime adapter with a torch-friendly call interface."""

    def __init__(self, artifact: PolicyArtifact | Path, manifest_path: Path | None = None):
        if isinstance(artifact, PolicyArtifact):
            self.artifact = artifact
            policy_path = artifact.policy_path
            input_name = artifact.input_name
            output_name = artifact.output_name
        else:
            policy_path = Path(artifact)
            if manifest_path is None:
                raise ValueError("manifest_path is required for a local policy file")
            metadata = validate_policy_artifact(policy_path, manifest_path)
            self.artifact = None
            input_name = metadata["input_name"]
            output_name = metadata["output_name"]
        self.input_name = input_name
        self.output_name = output_name
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])

    def __call__(self, observation: torch.Tensor | np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(observation)
        single = tensor.ndim == 1
        if single:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != EXPECTED_OBS_LEN:
            raise ValueError(
                f"Expected observation shape [batch, {EXPECTED_OBS_LEN}], got {tuple(tensor.shape)}"
            )
        values = np.asarray(tensor.detach().cpu(), dtype=np.float32)
        output = self.session.run([self.output_name], {self.input_name: values})[0]
        result = torch.from_numpy(np.asarray(output, dtype=np.float32))
        return result[0] if single else result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="alpha_walking")
    parser.add_argument("--repo-id", default=OFFICIAL_POLICY_REPO)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hf"))
    args = parser.parse_args(argv)
    artifact = fetch_policy(
        args.policy,
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(artifact.metadata(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
