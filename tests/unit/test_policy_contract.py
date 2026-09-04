import json

import pytest

from microduck_rl_torch.policies.huggingface import validate_policy_artifact


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
