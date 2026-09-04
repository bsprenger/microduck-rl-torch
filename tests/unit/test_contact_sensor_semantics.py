from __future__ import annotations

from types import SimpleNamespace

import torch

from microduck_rl_torch.envs import (
    PhysicsBackend,
    SceneBuilder,
    SceneCfg,
    SemanticSelector,
    SensorCfg,
    SensorManager,
    TerrainCfg,
)
from microduck_rl_torch.envs.model import load_model_bundle
from microduck_rl_torch.robot import MICRODUCK_BALL_CFG


def test_native_contact_reader_returns_primary_frame_force() -> None:
    """Contact force fields must not be shape-correct copies of ``found``."""

    scene = SceneCfg(
        entities={"ball": MICRODUCK_BALL_CFG},
        terrain=TerrainCfg(kind="plane"),
    )
    build = SceneBuilder().build(scene)
    bundle = load_model_bundle(
        build.xml_path,
        entity_cfg=MICRODUCK_BALL_CFG,
        entities=scene.entities,
        actuator_mode="xml",
    )
    physics = PhysicsBackend(bundle, actuator_mode="xml")
    qpos = bundle.default_qpos.clone()
    qpos[2] = 0.02
    physics.reset(
        qpos=qpos,
        qvel=bundle.default_qvel,
        ctrl=bundle.default_ctrl,
    )
    manager = SensorManager(
        {
            "probe": SensorCfg(
                "probe",
                kind="contact",
                primary=SemanticSelector(names=("ball_geom",)),
                secondary=SemanticSelector(names=("floor",)),
                fields=("found", "force", "normal"),
                reduce="netforce",
                expected_dim=7,
            )
        },
        bundle,
    )
    env = SimpleNamespace(data=physics.data, physics=physics, bundle=bundle, num_envs=1)
    manager.update(env)
    contact = manager.get_sensor("probe").data

    assert contact.found is not None
    assert contact.force is not None
    assert contact.normal is not None
    torch.testing.assert_close(contact.found, torch.ones(1))
    assert float(torch.linalg.vector_norm(contact.force)) > 0.0
    force = contact.force[0]
    normal = contact.normal[0]
    assert abs(float(force[0])) < 1.0e-6
    assert abs(float(force[1])) < 1.0e-6
    assert float(force[2]) < 0.0
    # The netforce dataspec emits a fixed basis marker for normal;
    # physical normal direction is covered by the non-reduced test below.
    torch.testing.assert_close(normal, torch.tensor([1.0, 0.0, 0.0]))


def test_native_contact_reader_matches_primary_direction() -> None:
    """Primary geom2 reverses normal/tangent, not the full local wrench."""

    scene = SceneCfg(
        entities={"ball": MICRODUCK_BALL_CFG},
        terrain=TerrainCfg(kind="plane"),
    )
    build = SceneBuilder().build(scene)
    bundle = load_model_bundle(
        build.xml_path,
        entity_cfg=MICRODUCK_BALL_CFG,
        entities=scene.entities,
        actuator_mode="xml",
    )
    physics = PhysicsBackend(bundle, actuator_mode="xml")
    qpos = bundle.default_qpos.clone()
    qpos[2] = 0.02
    physics.reset(qpos=qpos, qvel=bundle.default_qvel, ctrl=bundle.default_ctrl)
    manager = SensorManager(
        {
            "probe": SensorCfg(
                "probe",
                kind="contact",
                primary=SemanticSelector(names=("ball_geom",)),
                secondary=SemanticSelector(names=("floor",)),
                fields=("found", "force", "torque", "normal", "tangent"),
                reduce="none",
                expected_dim=13,
            )
        },
        bundle,
    )
    env = SimpleNamespace(data=physics.data, physics=physics, bundle=bundle, num_envs=1)
    manager.update(env)
    contact = manager.get_sensor("probe").data

    assert contact.found is not None
    assert contact.force is not None
    assert contact.normal is not None
    assert contact.tangent is not None
    torch.testing.assert_close(contact.found, torch.ones((1, 1)))
    torch.testing.assert_close(
        contact.force[0, 0], torch.tensor([0.7022925, 0.0, 0.0]), atol=2e-5, rtol=0.0
    )
    torch.testing.assert_close(
        contact.normal[0, 0], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        contact.tangent[0, 0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6, rtol=0.0
    )
