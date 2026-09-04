import pytest
import torch

from microduck_rl_torch.envs import ManagerBasedTaskEnv, SceneBuilder
from microduck_rl_torch.envs.model import load_model_bundle
from microduck_rl_torch.envs.rewards import foot_contact_mask
from microduck_rl_torch.robot import MICRODUCK_WALK_ROBOT_CFG
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg


@pytest.mark.integration
def test_environment_reset_and_short_rollout():
    bundle = load_model_bundle(
        entity_cfg=MICRODUCK_WALK_ROBOT_CFG,
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    environment = ManagerBasedTaskEnv(make_microduck_velocity_env_cfg(), bundle=bundle)
    observation = environment.reset()
    assert isinstance(observation, torch.Tensor)
    assert observation.shape == (61,)
    assert torch.isfinite(observation).all()
    result = environment.step(torch.zeros(14, dtype=bundle.dtype))
    assert isinstance(result.observation, torch.Tensor)
    assert result.observation.shape == (61,)
    assert result.info["finite"]


@pytest.mark.integration
def test_environment_contact_path_is_finite():
    bundle = load_model_bundle(
        entity_cfg=MICRODUCK_WALK_ROBOT_CFG,
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=False,
    )
    environment = ManagerBasedTaskEnv(make_microduck_velocity_env_cfg(), bundle=bundle)
    observation = environment.reset()
    assert isinstance(observation, torch.Tensor)
    result = environment.step(torch.zeros(14, dtype=bundle.dtype))
    assert isinstance(result.observation, torch.Tensor)
    assert bundle.contacts_enabled
    assert torch.isfinite(observation).all()
    assert torch.isfinite(result.observation).all()
    assert result.info["finite"]

    assert environment.data is not None
    assert environment.data.contact.geom1.ndim == 1
    assert environment.data.contact.geom1.shape == environment.data.contact.geom2.shape
    assert torch.isfinite(environment.data.contact.dist).all()
    assert torch.isfinite(environment.data.contact.pos).all()
    assert torch.isfinite(environment.data.contact.frame).all()
    assert foot_contact_mask(environment.data, bundle).dtype == torch.bool
    assert foot_contact_mask(environment.data, bundle).shape == (2,)


@pytest.mark.integration
def test_injected_bundle_rebuild_preserves_mesh_mesh_policy():
    cfg = make_microduck_velocity_env_cfg().clone()
    cfg.metadata["disable_mesh_mesh_contacts"] = True
    source_bundle = load_model_bundle(
        entity_cfg=MICRODUCK_WALK_ROBOT_CFG,
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
        disable_mesh_mesh_contacts=False,
    )

    environment = ManagerBasedTaskEnv(cfg, bundle=source_bundle)

    # The injected robot-only bundle does not include the task-owned world
    # wrapper, so ManagerBasedTaskEnv rebuilds it from SceneBuilder. The
    # rebuild must retain the task's explicit collision policy.
    assert environment.bundle.xml_path != source_bundle.xml_path
    assert environment.bundle.torch_model._device_precomp["skip_mesh_mesh_contacts"] is True


@pytest.mark.integration
def test_same_path_injected_bundle_rebuilds_wrong_contact_capacity():
    cfg = make_microduck_velocity_env_cfg().clone()
    cfg.nconmax = 200
    scene_build = SceneBuilder().build(cfg.scene)
    source_bundle = load_model_bundle(
        scene_build.xml_path,
        entity_cfg=cfg.scene.entities["robot"],
        entities=cfg.scene.entities,
        nconmax=17,
        disable_contacts=True,
    )

    environment = ManagerBasedTaskEnv(cfg, bundle=source_bundle)

    assert environment.bundle.xml_path.resolve() == scene_build.xml_path.resolve()
    assert environment.bundle.nconmax == 200


@pytest.mark.integration
def test_same_path_injected_bundle_rebuilds_mesh_mesh_policy():
    cfg = make_microduck_velocity_env_cfg().clone()
    cfg.metadata["disable_mesh_mesh_contacts"] = True
    scene_build = SceneBuilder().build(cfg.scene)
    source_bundle = load_model_bundle(
        scene_build.xml_path,
        entity_cfg=cfg.scene.entities["robot"],
        entities=cfg.scene.entities,
        disable_contacts=True,
        disable_mesh_mesh_contacts=False,
    )

    environment = ManagerBasedTaskEnv(cfg, bundle=source_bundle)

    assert environment.bundle.xml_path.resolve() == scene_build.xml_path.resolve()
    assert environment.bundle.disable_mesh_mesh_contacts is True
    assert environment.bundle.torch_model._device_precomp["skip_mesh_mesh_contacts"] is True
