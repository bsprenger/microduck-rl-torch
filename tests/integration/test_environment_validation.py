import pytest
import torch

from microduck_rl_torch.envs import ManagerBasedTaskEnv
from microduck_rl_torch.envs.model import load_microduck_model
from microduck_rl_torch.envs.rewards import foot_contact_mask
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg


@pytest.mark.integration
def test_environment_reset_and_short_rollout():
    bundle = load_microduck_model(
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    environment = ManagerBasedTaskEnv(make_microduck_velocity_env_cfg(), bundle=bundle)
    observation = environment.reset()
    assert observation.shape == (61,)
    assert torch.isfinite(observation).all()
    result = environment.step(torch.zeros(14, dtype=bundle.dtype))
    assert result.observation.shape == (61,)
    assert result.info["finite"]


@pytest.mark.integration
def test_environment_contact_path_is_finite():
    bundle = load_microduck_model(
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=False,
    )
    environment = ManagerBasedTaskEnv(make_microduck_velocity_env_cfg(), bundle=bundle)
    observation = environment.reset()
    result = environment.step(torch.zeros(14, dtype=bundle.dtype))
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
