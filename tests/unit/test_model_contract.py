import pytest

from microduck_rl_torch.envs.model import SERVO_JOINT_NAMES, load_model_bundle


@pytest.mark.integration
def test_microduck_model_contract():
    bundle = load_model_bundle(
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    assert bundle.native_model.nq == 21
    assert bundle.native_model.nv == 20
    assert bundle.native_model.nu == 14
    assert bundle.actuator_joint_names == SERVO_JOINT_NAMES
    assert bundle.timestep == 0.005
    assert bundle.decimation == 4
    assert bundle.solver_iterations == 2
    assert not bundle.contacts_enabled
    assert bundle.default_pose.shape == (14,)
