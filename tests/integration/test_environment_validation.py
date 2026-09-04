import pytest
import torch

from microduck_rl_torch.envs import NominalMicroDuckEnv
from microduck_rl_torch.envs.model import load_microduck_model


@pytest.mark.integration
def test_environment_reset_and_short_rollout():
    bundle = load_microduck_model(
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    environment = NominalMicroDuckEnv(bundle)
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
    environment = NominalMicroDuckEnv(bundle)
    observation = environment.reset()
    result = environment.step(torch.zeros(14, dtype=bundle.dtype))
    assert bundle.contacts_enabled
    assert torch.isfinite(observation).all()
    assert torch.isfinite(result.observation).all()
    assert result.info["finite"]
