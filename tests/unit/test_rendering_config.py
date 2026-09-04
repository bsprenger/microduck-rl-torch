import mujoco
import pytest

from microduck_rl_torch.envs import SceneBuilder
from microduck_rl_torch.rendering.camera import resolve_named_camera
from microduck_rl_torch.rendering.config import CameraConfig, RenderConfig
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg


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


def test_composed_camera_resolves_entity_qualified_name():
    scene_build = SceneBuilder().build(make_microduck_velocity_env_cfg().scene)
    model = mujoco.MjModel.from_xml_path(str(scene_build.xml_path))

    camera_id, camera_name = resolve_named_camera(model, "head_camera", entity_name="robot")

    assert camera_id >= 0
    assert camera_name == "robot/head_camera"
