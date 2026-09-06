from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import torch

from microduck_rl_torch.envs import (
    ActionCfg,
    JointPositionActionTermCfg,
    ManagerBasedTaskEnv,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    SceneBuilder,
    SceneCfg,
    SemanticSelector,
    SensorCfg,
    TaskStateTermCfg,
    TermCollection,
    TerrainCfg,
    TerrainHeightSensorCfg,
    TerrainManager,
)
from microduck_rl_torch.envs.model import load_model_bundle
from microduck_rl_torch.robot import (
    MICRODUCK_BALL_CFG,
    MICRODUCK_STANDUP_ROBOT_CFG,
)
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg


def test_scene_composer_builds_entities_and_overlays_entity_keyframes():
    ball = replace(MICRODUCK_BALL_CFG, spawn_pos=(0.5, 0.1, 0.035))
    scene = SceneCfg(
        entities={"robot": MICRODUCK_STANDUP_ROBOT_CFG, "ball": ball},
        terrain=TerrainCfg(kind="plane"),
    )
    build = SceneBuilder().build(scene)
    assert build.composed
    assert build.entity_names == ("robot", "ball")
    bundle = load_model_bundle(
        build.xml_path,
        entity_cfg=MICRODUCK_STANDUP_ROBOT_CFG,
        entities=scene.entities,
        actuator_mode="xml",
        disable_contacts=True,
    )
    assert set(bundle.entities) == {"robot", "ball"}
    assert bundle.entity("ball").kind == "prop"
    torch.testing.assert_close(
        bundle.default_qpos[bundle.entity("ball").qpos_indices[:3]],
        torch.tensor([0.5, 0.1, 0.035]),
    )


def test_sensor_manager_resolves_builtin_and_custom_sensor_contracts():
    cfg = make_microduck_velocity_env_cfg()
    cfg.scene.sensors.update(
        {
            "trunk_pose": SensorCfg("trunk_pose", kind="body_pose", expected_dim=7),
            "feet": SensorCfg(
                "feet",
                kind="site_position",
                selector=SemanticSelector(names=("left_foot", "right_foot")),
                expected_dim=6,
            ),
            "leg_positions": SensorCfg(
                "leg_positions",
                kind="joint_position",
                joint_names=(r".*(hip|knee|ankle).*",),
                expected_dim=10,
            ),
        }
    )

    class CustomSensor:
        def __init__(self, cfg, env):
            del cfg, env
            self.value = 0.0

        def reset(self, env_ids):
            del env_ids
            self.value = 1.0

        def read(self, env):
            del env
            self.value += 1.0
            return torch.tensor([self.value])

    cfg.scene.sensors["custom"] = SensorCfg(
        "custom", kind="custom", reader=CustomSensor, expected_dim=1
    )
    env = ManagerBasedTaskEnv(cfg)
    env.reset(seed=3)
    assert env.sensors.get_handle("trunk_pose").dimension == 7
    assert env.sensors.read("feet").shape == (6,)
    assert env.sensors.read("leg_positions").shape == (10,)
    assert env.sensors.read("custom").shape == (1,)
    before = env.sensors.read("custom").clone()
    env.step(torch.zeros(14))
    assert float(env.sensors.read("custom")) > float(before)


def test_action_manager_composes_disjoint_upstream_style_terms():
    cfg = make_microduck_velocity_env_cfg()
    cfg.actions = ActionCfg(
        terms=OrderedDict(
            (
                (
                    "legs",
                    JointPositionActionTermCfg(
                        entity_name="robot",
                        joint_names=(r".*(hip|knee|ankle).*",),
                        size=10,
                        scale=0.5,
                        offset="default",
                    ),
                ),
                (
                    "head",
                    JointPositionActionTermCfg(
                        entity_name="robot",
                        joint_names=(r"neck_pitch|head_.*",),
                        size=4,
                        scale=0.25,
                        offset="default",
                    ),
                ),
            )
        )
    )
    env = ManagerBasedTaskEnv(cfg)
    env.reset(seed=4)
    action = torch.ones(14)
    applied, target = env.action_manager.prepare(env, action)
    assert applied.shape == (14,)
    expected = env.bundle.default_pose.clone()
    expected[env.action_manager.get_term("legs").actuator_ids] += 0.5
    expected[env.action_manager.get_term("head").actuator_ids] += 0.25
    torch.testing.assert_close(
        target,
        expected,
    )
    assert env.action_manager.active_terms == ["legs", "head"]


def test_task_state_manager_has_explicit_reset_and_lifecycle():
    cfg = make_microduck_velocity_env_cfg()

    class PhaseState:
        def __init__(self, cfg, env):
            del cfg
            self.env = env
            self.reset_count = 0
            self.pre_count = 0
            self.post_count = 0
            self.step_count = 0

        def reset(self, env_ids):
            assert env_ids is not None
            self.reset_count += 1

        def pre_physics(self):
            self.pre_count += 1

        def post_physics(self):
            self.post_count += 1

        def step(self):
            self.step_count += 1

    cfg.task_state = TermCollection(OrderedDict((("phase", TaskStateTermCfg(func=PhaseState)),)))
    env = ManagerBasedTaskEnv(cfg)
    env.reset(seed=5)
    state = env.task_state_manager.get_term("phase")
    env.step(torch.zeros(14))
    assert state.reset_count == 1
    assert state.pre_count == 1
    assert state.post_count == 1
    assert state.step_count == 1


def test_scene_supports_repeated_entity_instances_and_explicit_world_template():
    first = replace(MICRODUCK_BALL_CFG, name="ball_a", spawn_pos=(0.3, 0.0, 0.04))
    second = replace(MICRODUCK_BALL_CFG, name="ball_b", spawn_pos=(0.6, 0.0, 0.04))
    scene = SceneCfg(
        entities={first.name: first, second.name: second},
        terrain=TerrainCfg(kind="plane"),
        scene_xml=MICRODUCK_STANDUP_ROBOT_CFG.scene_xml_path,
    )
    build = SceneBuilder().build(scene)
    model = load_model_bundle(
        build.xml_path,
        entity_cfg=first,
        entities=scene.entities,
        actuator_mode="xml",
        disable_contacts=True,
    )
    # The explicit scene XML is a world template and contains the standup
    # robot's authored actuators.  It is removed from the worldbody and then
    # reattached only when declared as an entity; the two ball entities do not
    # add actuators of their own.
    assert model.native_model.nu == 0
    assert model.entity("ball_a").root_body_id != model.entity("ball_b").root_body_id
    torch.testing.assert_close(
        model.default_qpos[model.entity("ball_b").qpos_indices[:3]],
        torch.tensor([0.6, 0.0, 0.04]),
    )


def test_contact_data_preserves_fields_and_per_primary_slots():
    cfg = make_microduck_velocity_env_cfg()
    cfg.scene.sensors["feet_contact_data"] = SensorCfg(
        "feet_contact_data",
        kind="contact",
        primary=SemanticSelector(mode="regex", pattern=r"^(left|right)_foot_collision$"),
        secondary=SemanticSelector(mode="regex", pattern=r"^(floor|terrain.*)$"),
        fields=("found", "force", "dist"),
        reduce="none",
        num_slots=2,
        expected_dim=2 * 2 * (1 + 3 + 1),
    )
    env = ManagerBasedTaskEnv(cfg)
    env.reset(seed=6)
    data = env.scene["feet_contact_data"].data
    assert data.found is not None and data.found.shape == (2, 2)
    assert data.force is not None and data.force.shape == (2, 2, 3)
    assert data.dist is not None and data.dist.shape == (2, 2)
    assert env.sensors.read("feet_contact_data").shape == (20,)


def test_raycast_and_terrain_height_are_first_class_typed_sensors():
    cfg = make_microduck_velocity_env_cfg()
    frames = (ObjRef("site", "left_foot", "robot"), ObjRef("site", "right_foot", "robot"))
    pattern = RingPatternCfg.single_ring(radius=0.01, num_samples=2)
    cfg.scene.sensors["rays"] = RayCastSensorCfg(
        name="rays", frame=frames, pattern=pattern, max_distance=1.0
    )
    cfg.scene.sensors["height"] = TerrainHeightSensorCfg(
        name="height", frame=frames, pattern=pattern, max_distance=1.0
    )
    env = ManagerBasedTaskEnv(cfg)
    env.reset(seed=7)
    rays = env.scene["rays"].data
    heights = env.scene["height"].data
    assert rays.distances.shape == (6,)
    assert rays.hit_pos_w.shape == (6, 3)
    assert heights.heights is not None and heights.heights.shape == (2,)


def test_batched_partial_reset_preserves_sibling_state_and_rng():
    cfg = make_microduck_velocity_env_cfg()
    cfg.scene.num_envs = 2
    env = ManagerBasedTaskEnv(
        cfg,
        num_envs=2,
        fixed_iterations=True,
        solver_iterations=2,
        line_search_iterations=2,
        disable_contacts=True,
    )
    env.reset(seed=12)
    first_command = env.command[0].clone()
    env.step(torch.zeros((2, cfg.action_size)))
    first_qpos = env.data.qpos[0].clone()
    env.reset(env_ids=torch.tensor([1]), seed=99)
    torch.testing.assert_close(env.command[0], first_command)
    torch.testing.assert_close(env.data.qpos[0], first_qpos)
    assert env.step_counts.tolist() == [1, 0]


def test_cached_generated_terrain_restores_origins_and_reaches_top_level():
    cfg = make_microduck_velocity_env_cfg(rough=True)
    first = SceneBuilder().build(cfg.scene)
    second = SceneBuilder().build(cfg.scene)
    assert first.xml_path == second.xml_path
    assert cfg.scene.terrain.generated_origins is not None
    manager = TerrainManager(cfg.scene.terrain, num_envs=1)
    manager.reset(seed=1)
    manager.levels[0] = manager.origins.shape[0] - 2
    manager.advance(torch.tensor([0]), delta=1)
    assert int(manager.levels[0]) == manager.origins.shape[0] - 1
    assert float(manager.env_origins[0, 2]) > 0.0
