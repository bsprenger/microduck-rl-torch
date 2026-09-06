import pytest

from microduck_rl_torch.rendering.config import CameraConfig, RenderConfig


def test_render_config_has_independent_camera_defaults():
    first = RenderConfig()
    second = RenderConfig()

    assert first.camera == second.camera
    assert first.camera is not second.camera


def test_camera_config_rejects_conflicting_sources():
    with pytest.raises(ValueError, match="mutually exclusive"):
        CameraConfig(name="head_camera", track_body="trunk_base")


def test_render_config_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        RenderConfig(width=0)
