import pytest
import torch

from microduck_rl_torch.envs import NominalMicroDuckEnv
from microduck_rl_torch.envs.model import load_microduck_model


@pytest.mark.integration
def test_reset_randomization_is_bounded_and_non_accumulating():
    bundle = load_microduck_model(
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    environment = NominalMicroDuckEnv(bundle, domain_randomization=True)

    environment.reset(seed=123)
    trunk = bundle.trunk_body_id
    trunk_delta = bundle.torch_model.body_ipos[trunk] - environment._base_body_ipos[trunk]

    first_body_ipos = bundle.torch_model.body_ipos.clone()
    first_body_mass = bundle.torch_model.body_mass.clone()
    first_armature = bundle.torch_model.dof_armature.clone()
    first_geom_friction = bundle.torch_model.geom_friction.clone()
    first_vin = environment._bam_vin
    first_drop_gain = environment._bam_drop_gain
    assert first_vin is not None
    assert first_drop_gain is not None

    environment.reset(seed=123)
    assert torch.equal(bundle.torch_model.body_ipos, first_body_ipos)
    assert torch.equal(bundle.torch_model.body_mass, first_body_mass)
    assert torch.equal(bundle.torch_model.dof_armature, first_armature)
    assert torch.equal(bundle.torch_model.geom_friction, first_geom_friction)
    vin = environment._bam_vin
    drop_gain = environment._bam_drop_gain
    assert vin is not None
    assert drop_gain is not None
    assert torch.equal(vin, first_vin)
    assert torch.equal(drop_gain, first_drop_gain)

    environment.reset(seed=456)
    vin = environment._bam_vin
    drop_gain = environment._bam_drop_gain
    assert vin is not None
    assert drop_gain is not None
    assert torch.equal(vin, first_vin)
    assert torch.equal(drop_gain, first_drop_gain)

    randomization = environment.config.randomization
    assert torch.all(torch.abs(trunk_delta) <= randomization.com_range)
    mass_ratio = bundle.torch_model.body_mass[trunk] / environment._base_body_mass[trunk]
    assert randomization.mass_inertia_range[0] <= float(mass_ratio)
    assert float(mass_ratio) <= randomization.mass_inertia_range[1]
