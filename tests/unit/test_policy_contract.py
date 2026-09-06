import json

import numpy as np
import onnx
import pytest

from microduck_rl_torch.policies.huggingface import (
    _manifest_policy_filenames,
    _policy_entry,
    resolve_policy_filename,
    validate_policy_artifact,
)


def test_missing_policy_is_not_silently_accepted(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"robot": "microduck", "policies": [{"name": "alpha_walking.onnx"}]})
    )
    with pytest.raises(FileNotFoundError):
        validate_policy_artifact(tmp_path / "alpha_walking.onnx", tmp_path / "manifest.json")


def test_policy_filename_rejects_path_traversal():
    from microduck_rl_torch.policies.huggingface import _policy_filename

    with pytest.raises(ValueError):
        _policy_filename("../alpha_walking")


def test_manifest_policy_filenames_support_official_list_shape():
    manifest = {
        "policies": [
            {"name": "alpha_walking.onnx"},
            {"name": "ball_kick_left.onnx"},
        ]
    }

    assert _manifest_policy_filenames(manifest) == [
        "alpha_walking.onnx",
        "ball_kick_left.onnx",
    ]
    assert _policy_entry(manifest, "ball_kick_left.onnx")["name"] == "ball_kick_left.onnx"


def test_manifest_policy_filenames_support_mapping_shape():
    manifest = {"policies": {"roller": {"file": "roller.onnx"}}}

    assert _manifest_policy_filenames(manifest) == ["roller.onnx"]


def test_logical_policy_name_resolves_to_declared_file():
    manifest = {"policies": [{"name": "sitstand", "file": "alpha_sitstand.onnx"}]}

    assert resolve_policy_filename(manifest, "sitstand") == "alpha_sitstand.onnx"


def test_policy_set_downloads_each_policy_without_shared_metadata_file(monkeypatch, tmp_path):
    from microduck_rl_torch.policies import huggingface

    manifest = {
        "robot": "microduck",
        "policies": [
            {"name": "first", "file": "alpha_first.onnx"},
            {"name": "second", "file": "alpha_second.onnx"},
        ],
    }
    manifest_cache = tmp_path / "manifest.json"
    manifest_cache.write_text(json.dumps(manifest))
    downloads = {"manifest.json": manifest_cache}

    class FakeApi:
        def model_info(self, **_kwargs):
            return type("Info", (), {"sha": "commit"})()

    monkeypatch.setattr(huggingface, "HfApi", FakeApi)

    def fake_download(*, filename, **_kwargs):
        path = downloads.get(filename)
        if path is None:
            path = tmp_path / filename
            path.write_bytes(b"onnx")
            downloads[filename] = path
        return str(path)

    monkeypatch.setattr(huggingface, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        huggingface,
        "validate_policy_artifact",
        lambda policy_path, manifest_path, **_kwargs: {
            "sha256": "sha",
            "input_name": "obs",
            "output_name": "action",
        },
    )

    artifacts = huggingface.fetch_policy_set(output_dir=tmp_path / "out")

    assert set(artifacts) == {"first", "second"}
    assert artifacts["first"].policy_name == "alpha_first.onnx"
    assert artifacts["second"].policy_name == "alpha_second.onnx"
    assert (tmp_path / "out" / "alpha_first.onnx").is_file()
    assert (tmp_path / "out" / "alpha_second.onnx").is_file()
    assert (tmp_path / "out" / "manifest.json").is_file()
    download_metadata = json.loads((tmp_path / "out" / "download.json").read_text())
    assert download_metadata["revision"] == "commit"
    assert set(download_metadata["policies"]) == {"alpha_first.onnx", "alpha_second.onnx"}
    assert not (tmp_path / "out" / "artifact.json").exists()


def _write_linear_policy(path, *, input_shape, output_shape):
    observation = onnx.helper.make_tensor_value_info(
        "observation", onnx.TensorProto.FLOAT, input_shape
    )
    action = onnx.helper.make_tensor_value_info("action", onnx.TensorProto.FLOAT, output_shape)
    weights = onnx.numpy_helper.from_array(np.zeros((61, 14), dtype=np.float32), name="weights")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["observation", "weights"], ["action"])],
        "policy",
        [observation],
        [action],
        [weights],
    )
    path.write_bytes(onnx.helper.make_model(graph).SerializeToString())


def test_policy_graph_requires_batched_interface(tmp_path):
    policy_path = tmp_path / "policy.onnx"
    manifest_path = tmp_path / "manifest.json"
    _write_linear_policy(policy_path, input_shape=[None, 61], output_shape=[None, 14])
    manifest_path.write_text(json.dumps({"policies": [{"file": "policy.onnx"}]}))

    metadata = validate_policy_artifact(policy_path, manifest_path)

    assert metadata["input_shape"] == [None, 61]
    assert metadata["output_shape"] == [None, 14]


@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    [([61], [14]), ([None, 61], [14]), ([2, 61], [None, 14])],
)
def test_policy_graph_rejects_non_runtime_shapes(tmp_path, input_shape, output_shape):
    policy_path = tmp_path / "policy.onnx"
    manifest_path = tmp_path / "manifest.json"
    _write_linear_policy(policy_path, input_shape=input_shape, output_shape=output_shape)
    manifest_path.write_text(json.dumps({"policies": [{"file": "policy.onnx"}]}))

    with pytest.raises(ValueError, match="shape"):
        validate_policy_artifact(policy_path, manifest_path)
