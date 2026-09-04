import mujoco
import pytest

from microduck_rl_torch.envs.model import (
    SERVO_JOINT_NAMES,
    _disable_mesh_mesh_contact_candidates,
    load_model_bundle,
)
from microduck_rl_torch.robot import MICRODUCK_WALK_ROBOT_CFG


@pytest.mark.integration
def test_microduck_model_contract():
    bundle = load_model_bundle(
        entity_cfg=MICRODUCK_WALK_ROBOT_CFG,
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


@pytest.mark.integration
def test_explicit_native_contact_capacity_is_part_of_model_contract():
    bundle = load_model_bundle(
        entity_cfg=MICRODUCK_WALK_ROBOT_CFG,
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        nconmax=200,
        disable_contacts=True,
    )
    assert bundle.native_model.nconmax == 200
    assert bundle.nconmax == 200


def test_mesh_mesh_disabling_removes_explicit_pairs():
    spec = mujoco.MjSpec()
    body_a = spec.worldbody.add_body()
    body_a.name = "a"
    geom_a = body_a.add_geom()
    geom_a.name = "mesh_a"
    geom_a.type = mujoco.mjtGeom.mjGEOM_MESH
    body_b = spec.worldbody.add_body()
    body_b.name = "b"
    geom_b = body_b.add_geom()
    geom_b.name = "mesh_b"
    geom_b.type = mujoco.mjtGeom.mjGEOM_MESH
    pair = spec.add_pair()
    pair.geomname1 = "mesh_a"
    pair.geomname2 = "mesh_b"

    _disable_mesh_mesh_contact_candidates(spec)

    assert not spec.pairs
