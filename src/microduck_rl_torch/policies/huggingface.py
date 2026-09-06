"""Fetch, inspect, and execute official MicroDuck ONNX policy artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable
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
    candidates = _manifest_candidates(manifest)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        names = {
            str(candidate.get(key, ""))
            for key in ("name", "id", "filename", "file", "path", "policy")
        }
        if filename in names or stem in names:
            return candidate
    raise ValueError(f"{filename} is not declared in the Hugging Face policy manifest")


def _manifest_candidates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return policy records from the supported manifest list/dict shapes."""

    candidates = manifest.get("policies", manifest.get("policy", []))
    if isinstance(candidates, list):
        return [candidate for candidate in candidates if isinstance(candidate, dict)]
    if not isinstance(candidates, dict):
        return []

    # The official manifest is a list today, but accepting a mapping makes the
    # downloader useful for small private repos without adding a registry.
    result: list[dict[str, Any]] = []
    for name, value in candidates.items():
        if isinstance(value, dict):
            result.append({"name": name, **value})
        else:
            result.append({"name": name})
    return result


def _manifest_policy_filenames(manifest: dict[str, Any]) -> list[str]:
    """Return every ONNX filename declared by a policy manifest."""

    filenames: list[str] = []
    for entry in _manifest_candidates(manifest):
        filename = _entry_filename(entry)
        if filename not in filenames:
            filenames.append(filename)
        else:
            raise ValueError(f"Policy manifest declares {filename} more than once")
    if not filenames:
        raise ValueError("Policy manifest does not declare any policies")
    return filenames


def _entry_filename(entry: dict[str, Any]) -> str:
    """Return the repository filename declared by one manifest entry."""

    # ``file`` is the official manifest field. The other path-like fields are
    # accepted for small compatible manifests; aliases are only a fallback.
    for key in ("file", "filename", "path"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return _policy_filename(value)
    for key in ("name", "id", "policy"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return _policy_filename(value)
    raise ValueError("Policy manifest entry does not declare a policy filename")


def _resolve_policy(manifest: dict[str, Any], policy: str) -> tuple[str, dict[str, Any]]:
    """Resolve a logical policy name to its canonical repository filename."""

    requested = _policy_filename(policy)
    entry = _policy_entry(manifest, requested)
    return _entry_filename(entry), entry


def resolve_policy_filename(manifest: dict[str, Any], policy: str) -> str:
    """Resolve a logical policy name to the ONNX filename in a manifest."""

    filename, _ = _resolve_policy(manifest, policy)
    return filename


def _entry_name(entry: dict[str, Any], filename: str) -> str:
    """Return the stable user-facing key for a manifest policy."""

    for key in ("name", "id", "policy"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return Path(value).stem
    return Path(filename).stem


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


def _validate_batched_shape(shape: list[int | None], expected_width: int, *, label: str) -> None:
    if len(shape) != 2 or shape[-1] != expected_width or shape[0] not in (None, 1):
        raise ValueError(
            f"Expected {label} shape [batch, {expected_width}] with dynamic or unit batch, "
            f"got {shape}"
        )


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
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("ONNX graph must have exactly one input and one output")
    input_shape = _graph_shape(model.graph.input[0])
    output_shape = _graph_shape(model.graph.output[0])
    _validate_batched_shape(input_shape, EXPECTED_OBS_LEN, label="ONNX input")
    _validate_batched_shape(output_shape, EXPECTED_ACTION_LEN, label="ONNX output")
    expected_dtype = onnx.TensorProto.FLOAT
    for label, value in (("input", model.graph.input[0]), ("output", model.graph.output[0])):
        dtype = value.type.tensor_type.elem_type
        if dtype != expected_dtype:
            raise ValueError(f"Expected ONNX {label} dtype float32, got {dtype}")
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


def _resolved_revision(api: HfApi, repo_id: str, revision: str | None) -> str:
    model_info = api.model_info(repo_id=repo_id, revision=revision)
    resolved_revision = model_info.sha
    if not resolved_revision:
        raise RuntimeError(f"Hugging Face did not return a commit revision for {repo_id}")
    return resolved_revision


def _download_manifest(*, repo_id: str, revision: str, cache_dir: Path | None) -> Path:
    download_kwargs: dict[str, Any] = {"repo_id": repo_id, "revision": revision}
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(cache_dir)
    return Path(hf_hub_download(filename="manifest.json", **download_kwargs))


def _artifact_from_download(
    filename: str,
    *,
    repo_id: str,
    revision: str,
    manifest: dict[str, Any],
    manifest_cache: Path,
    cache_dir: Path | None,
    output_dir: Path | None,
) -> PolicyArtifact:
    download_kwargs: dict[str, Any] = {"repo_id": repo_id, "revision": revision}
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(cache_dir)
    policy_cache = Path(hf_hub_download(filename=filename, **download_kwargs))

    destination = output_dir or policy_cache.parent
    destination.mkdir(parents=True, exist_ok=True)
    if output_dir is not None:
        policy_path = destination / filename
        manifest_path = destination / "manifest.json"
        shutil.copy2(policy_cache, policy_path)
        # All policies in a set share this one manifest. Copying it is
        # idempotent and avoids one artifact.json per policy.
        shutil.copy2(manifest_cache, manifest_path)
    else:
        policy_path = policy_cache
        manifest_path = manifest_cache

    graph_metadata = validate_policy_artifact(policy_path, manifest_path, filename=filename)
    expected_sha256 = (
        OFFICIAL_GOLDEN_SHA256.get(filename) if repo_id == OFFICIAL_POLICY_REPO else None
    )
    if expected_sha256 is not None and graph_metadata["sha256"] != expected_sha256:
        raise ValueError(
            f"Golden digest drift for {filename}: expected {expected_sha256}, "
            f"got {graph_metadata['sha256']}"
        )
    return PolicyArtifact(
        repo_id=repo_id,
        revision=revision,
        policy_name=filename,
        policy_path=policy_path,
        manifest_path=manifest_path,
        manifest=manifest,
        sha256=graph_metadata["sha256"],
        input_name=graph_metadata["input_name"],
        output_name=graph_metadata["output_name"],
    )


def _write_download_metadata(output_dir: Path, artifacts: Iterable[PolicyArtifact]) -> None:
    """Write one small set-level provenance record after successful downloads."""

    artifacts = tuple(artifacts)
    if not artifacts:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "repo_id": artifacts[0].repo_id,
        "revision": artifacts[0].revision,
        "policies": {artifact.policy_name: {"sha256": artifact.sha256} for artifact in artifacts},
    }
    (output_dir / "download.json").write_text(json.dumps(metadata, indent=2) + "\n")


def fetch_policy(
    policy: str = "alpha_walking",
    *,
    repo_id: str = OFFICIAL_POLICY_REPO,
    revision: str | None = None,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
) -> PolicyArtifact:
    """Download one official policy at a resolved revision and validate it."""

    api = HfApi()
    resolved_revision = _resolved_revision(api, repo_id, revision)
    manifest_cache = _download_manifest(
        repo_id=repo_id, revision=resolved_revision, cache_dir=cache_dir
    )
    manifest = json.loads(manifest_cache.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Policy manifest must contain a JSON object")
    filename, _ = _resolve_policy(manifest, policy)
    artifact = _artifact_from_download(
        filename,
        repo_id=repo_id,
        revision=resolved_revision,
        manifest=manifest,
        manifest_cache=manifest_cache,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    if output_dir is not None:
        _write_download_metadata(output_dir, (artifact,))
    return artifact


def fetch_policy_set(
    policies: str | Iterable[str] | None = None,
    *,
    repo_id: str = OFFICIAL_POLICY_REPO,
    revision: str | None = None,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, PolicyArtifact]:
    """Download and validate a manifest-selected set of ONNX policies.

    With ``policies=None`` every ONNX policy declared by ``manifest.json`` is
    fetched. The returned mapping is keyed by the manifest's logical policy
    name, so callers can explicitly choose and wire a policy into whichever
    task environment they constructed.
    """

    api = HfApi()
    resolved_revision = _resolved_revision(api, repo_id, revision)
    manifest_cache = _download_manifest(
        repo_id=repo_id, revision=resolved_revision, cache_dir=cache_dir
    )
    manifest = json.loads(manifest_cache.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Policy manifest must contain a JSON object")
    if policies is None:
        requested = _manifest_policy_filenames(manifest)
    elif isinstance(policies, str):
        requested = [policies]
    else:
        requested = list(policies)

    resolved: list[tuple[str, str]] = []
    seen_filenames: set[str] = set()
    seen_names: set[str] = set()
    for policy in requested:
        filename, entry = _resolve_policy(manifest, policy)
        if filename in seen_filenames:
            raise ValueError(f"Policy {filename} was requested more than once")
        name = _entry_name(entry, filename)
        if name in seen_names:
            raise ValueError(f"Policy manifest uses the name {name!r} more than once")
        seen_filenames.add(filename)
        seen_names.add(name)
        resolved.append((filename, name))

    artifacts: dict[str, PolicyArtifact] = {}
    for filename, name in resolved:
        artifact = _artifact_from_download(
            filename,
            repo_id=repo_id,
            revision=resolved_revision,
            manifest=manifest,
            manifest_cache=manifest_cache,
            cache_dir=cache_dir,
            output_dir=output_dir,
        )
        artifacts[name] = artifact
    if output_dir is not None:
        _write_download_metadata(output_dir, artifacts.values())
    return artifacts


def load_policy(
    policy_path: Path,
    manifest_path: Path,
    *,
    repo_id: str = OFFICIAL_POLICY_REPO,
    revision: str = "local-artifact",
) -> PolicyArtifact:
    """Validate a policy already present on disk and return its build record."""

    policy_path = Path(policy_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Policy manifest must contain a JSON object")
    metadata = validate_policy_artifact(policy_path, manifest_path)
    return PolicyArtifact(
        repo_id=repo_id,
        revision=revision,
        policy_name=policy_path.name,
        policy_path=policy_path,
        manifest_path=manifest_path,
        manifest=manifest,
        sha256=metadata["sha256"],
        input_name=metadata["input_name"],
        output_name=metadata["output_name"],
    )


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
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every ONNX policy declared by the repository manifest",
    )
    args = parser.parse_args(argv)
    if args.all:
        artifacts = fetch_policy_set(
            repo_id=args.repo_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {name: artifact.metadata() for name, artifact in artifacts.items()},
                indent=2,
                default=str,
            )
        )
    else:
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
