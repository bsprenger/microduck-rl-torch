from types import SimpleNamespace

import mujoco
import mujoco_torch
import numpy as np

from microduck_rl_torch.envs.kinematics import site_linear_velocity


def test_site_linear_velocity_matches_mujoco_jacobian_for_scalar_and_batch():
    xml = (
        "<mujoco><worldbody><body name='body' pos='0 0 0'>"
        "<joint name='free' type='free'/><geom type='sphere' size='0.1' mass='1'/>"
        "<site name='probe' pos='0.2 0.1 0.3'/></body></worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    native = mujoco.MjData(model)
    native.qpos[:] = [0.2, -0.1, 0.4, 0.9238795, 0.0, 0.3826834, 0.0]
    native.qvel[:] = np.arange(model.nv, dtype=np.float64) + 1.0
    mujoco.mj_forward(model, native)
    torch_data = mujoco_torch.forward(
        mujoco_torch.device_put(model), mujoco_torch.device_put(native)
    )
    bundle = SimpleNamespace(native_model=model)

    jacobian = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, native, jacobian, None, 0)
    expected = jacobian @ native.qvel
    actual = site_linear_velocity(torch_data, bundle, [0])[0].detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    batched = SimpleNamespace(
        site_xpos=torch_data.site_xpos.unsqueeze(0).repeat(2, 1, 1),
        xipos=torch_data.xipos.unsqueeze(0).repeat(2, 1, 1),
        cvel=torch_data.cvel.unsqueeze(0).repeat(2, 1, 1),
    )
    actual_batch = site_linear_velocity(batched, bundle, [0]).detach().numpy()[:, 0]
    np.testing.assert_allclose(actual_batch, np.stack((expected, expected)), atol=1e-10)
