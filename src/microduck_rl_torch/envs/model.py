"""Load and fingerprint the MicroDuck MuJoCo model for torch execution."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco_torch
import numpy as np
import torch

from ..robot.model_variants import SERVO_JOINT_NAMES
from .actuation import BamM6Parameters
from .scene import EntityCfg, SemanticSelector

mujoco_api: Any = mujoco

SENSOR_NAMES = ("imu_ang_vel", "imu_accel")


def default_scene_path() -> Path:
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[3] / "assets/robot/microduck/scene_walk.xml",
        module_path.parents[2] / "assets/robot/microduck/scene_walk.xml",
        Path.cwd() / "assets/robot/microduck/scene_walk.xml",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _required_id(model: Any, obj_type: int, name: str) -> int:
    object_id = mujoco_api.mj_name2id(model, obj_type, name)
    if object_id < 0:
        raise ValueError(f"Required MuJoCo object {name!r} was not found")
    return int(object_id)


def _object_name(model: Any, obj_type: int, object_id: int) -> str:
    name = mujoco_api.mj_id2name(model, obj_type, object_id)
    return name or f"<unnamed:{object_id}>"


def _body_descendants(model: Any, body_id: int) -> set[int]:
    """Return a body and every descendant body in the MuJoCo tree."""

    descendants: set[int] = set()
    for candidate in range(int(model.nbody)):
        current = candidate
        while current >= 0:
            if current == body_id:
                descendants.add(candidate)
                break
            parent = int(model.body_parentid[current])
            if parent == current:
                break
            current = parent
    return descendants


def _resolve_selector(model: Any, obj_type: int, selector: SemanticSelector) -> tuple[int, ...]:
    """Resolve a semantic selector against the compiled native model."""

    if selector.mode == "names":
        result = tuple(_required_id(model, obj_type, name) for name in selector.names)
    elif selector.mode == "regex":
        pattern = re.compile(selector.pattern or "")
        # MuJoCo's Python API does not expose a common object-count accessor;
        # resolve the supported selector types explicitly below.
        count = {
            mujoco_api.mjtObj.mjOBJ_BODY: int(model.nbody),
            mujoco_api.mjtObj.mjOBJ_GEOM: int(model.ngeom),
            mujoco_api.mjtObj.mjOBJ_SITE: int(model.nsite),
        }.get(obj_type)
        if count is None:
            raise ValueError(f"Regex selectors do not support MuJoCo object type {obj_type}")
        result = tuple(
            object_id
            for object_id in range(count)
            if pattern.search(_object_name(model, obj_type, object_id))
        )
        if not result:
            raise ValueError(f"Selector {selector!r} matched no MuJoCo objects")
    elif selector.mode == "body_subtree":
        if obj_type != mujoco_api.mjtObj.mjOBJ_GEOM:
            raise ValueError("body_subtree selectors currently resolve geometry contacts only")
        pattern = re.compile(selector.pattern or "")
        body_ids = tuple(
            body_id
            for body_id in range(int(model.nbody))
            if pattern.search(_object_name(model, mujoco_api.mjtObj.mjOBJ_BODY, body_id))
        )
        if not body_ids:
            raise ValueError(f"Body selector {selector!r} matched no MuJoCo bodies")
        descendants = set().union(*(_body_descendants(model, body_id) for body_id in body_ids))
        result = tuple(
            geom_id
            for geom_id in range(int(model.ngeom))
            if int(model.geom_bodyid[geom_id]) in descendants
        )
        if not result:
            raise ValueError(f"Body selector {selector!r} matched no geometry")
    else:
        raise ValueError(f"Unsupported selector mode {selector.mode!r}")
    return result


@dataclass(frozen=True)
class EntityView:
    """Resolved semantic view of one compiled scene entity."""

    name: str
    kind: str
    root_body_id: int
    body_ids: tuple[int, ...]
    geom_ids: tuple[int, ...]
    joint_ids: tuple[int, ...]
    qpos_indices: torch.Tensor
    qvel_indices: torch.Tensor


def _joint_qpos_width(joint_type: int) -> int:
    # MuJoCo joint enum values are stable: free, ball, slide, hinge.
    return (7, 4, 1, 1)[joint_type]


def _joint_qvel_width(joint_type: int) -> int:
    return (6, 3, 1, 1)[joint_type]


def _resolve_entity_view(
    model: Any,
    entity_cfg: EntityCfg,
    *,
    device: torch.device,
) -> EntityView:
    root_name = entity_cfg.root_body_name or entity_cfg.trunk_body_name
    root_body_id = _required_id(model, mujoco_api.mjtObj.mjOBJ_BODY, root_name)
    body_ids = tuple(sorted(_body_descendants(model, root_body_id)))
    body_id_set = set(body_ids)
    geom_ids = tuple(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in body_id_set
    )
    joint_ids = tuple(
        joint_id
        for joint_id in range(int(model.njnt))
        if int(model.jnt_bodyid[joint_id]) in body_id_set
    )
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    for joint_id in joint_ids:
        joint_type = int(model.jnt_type[joint_id])
        qpos_start = int(model.jnt_qposadr[joint_id])
        qvel_start = int(model.jnt_dofadr[joint_id])
        qpos_indices.extend(range(qpos_start, qpos_start + _joint_qpos_width(joint_type)))
        qvel_indices.extend(range(qvel_start, qvel_start + _joint_qvel_width(joint_type)))
    return EntityView(
        name=entity_cfg.name,
        kind=entity_cfg.kind,
        root_body_id=root_body_id,
        body_ids=body_ids,
        geom_ids=geom_ids,
        joint_ids=joint_ids,
        qpos_indices=torch.tensor(qpos_indices, dtype=torch.long, device=device),
        qvel_indices=torch.tensor(qvel_indices, dtype=torch.long, device=device),
    )


@dataclass(frozen=True)
class MicroDuckModelBundle:
    """Native and device-resident representations of one MicroDuck model."""

    xml_path: Path
    entity_cfg: EntityCfg
    native_model: Any
    torch_model: Any
    device: torch.device
    dtype: torch.dtype
    timestep: float
    decimation: int
    fixed_iterations: bool
    solver_iterations: int
    line_search_iterations: int
    contacts_enabled: bool
    qpos_indices: torch.Tensor
    qvel_indices: torch.Tensor
    default_qpos: torch.Tensor
    backlash_qpos_indices: torch.Tensor
    backlash_qvel_indices: torch.Tensor
    backlash_mask: torch.Tensor
    default_pose: torch.Tensor
    actuator_joint_names: tuple[str, ...]
    sensor_slices: dict[str, slice]
    trunk_body_id: int
    head_body_ids: tuple[int, ...]
    foot_site_ids: tuple[int, ...]
    foot_geom_ids: tuple[int, ...]
    foot_geom_groups: tuple[tuple[int, ...], ...]
    collision_geom_ids: tuple[int, ...]
    actuator_mode: str
    bam_parameters: BamM6Parameters | None
    friction_dof_count: int
    entities: Mapping[str, EntityView]
    observation_size: int = 61
    action_size: int = 14

    @property
    def has_backlash(self) -> bool:
        return bool(self.backlash_mask.any().item())

    def entity(self, name: str) -> EntityView:
        """Return a resolved semantic entity view by task-configured name."""

        try:
            return self.entities[name]
        except KeyError as exc:
            raise KeyError(f"Scene entity {name!r} is not present in the model bundle") from exc

    def new_data(self) -> Any:
        """Create a forward-computed standing state on the target device."""

        # ``mujoco_torch.make_data`` currently allocates floating-point data
        # using its package default (float64) on the host.  Move the complete
        # data tree to the model's requested device and dtype before replacing
        # the initial state; otherwise float32 models fail in contact/Jacobian
        # code and non-CPU models receive mixed-device state.
        # ``make_data`` still constructs several intermediate fields with its
        # package-wide float64 default. MPS rejects float64 tensors entirely,
        # even when they are immediately cast afterward, so allocate those
        # intermediates from a temporary CPU model and only then move the
        # complete data tree to the requested device/dtype.
        data_model = self.torch_model
        if self.device.type != "cpu":
            data_model = mujoco_torch.device_put(self.native_model, dtype=torch.float64)
        data = mujoco_torch.make_data(data_model).to(
            device=self.device,
            dtype=self.dtype,
        )
        qpos = torch.as_tensor(
            self.default_qpos.detach().cpu().numpy(),
            dtype=self.dtype,
            device=self.device,
        )
        qvel = torch.zeros_like(data.qvel)
        ctrl = (
            torch.zeros_like(data.ctrl)
            if self.actuator_mode == "bam"
            else self.default_pose.clone()
        )
        return mujoco_torch.forward(
            self.torch_model,
            data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl),
            fixed_iterations=self.fixed_iterations,
        )

    def fingerprint(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.xml_path.read_bytes()).hexdigest()
        return {
            "xml_sha256": digest,
            "xml_path": str(self.xml_path),
            "nq": int(self.native_model.nq),
            "nv": int(self.native_model.nv),
            "nu": int(self.native_model.nu),
            "nbody": int(self.native_model.nbody),
            "ngeom": int(self.native_model.ngeom),
            "nsensor": int(self.native_model.nsensor),
            "timestep": self.timestep,
            "decimation": self.decimation,
            "fixed_iterations": self.fixed_iterations,
            "solver_iterations": self.solver_iterations,
            "line_search_iterations": self.line_search_iterations,
            "contacts_enabled": self.contacts_enabled,
            "actuator_joint_names": list(self.actuator_joint_names),
            "entity_name": self.entity_cfg.name,
            "entities": {
                name: {
                    "kind": entity.kind,
                    "root_body_id": entity.root_body_id,
                    "body_ids": list(entity.body_ids),
                    "geom_ids": list(entity.geom_ids),
                    "joint_ids": list(entity.joint_ids),
                }
                for name, entity in self.entities.items()
            },
            "robot_xml_path": str(self.entity_cfg.xml_path),
            "scene_xml_path": (
                str(self.entity_cfg.scene_xml_path)
                if self.entity_cfg.scene_xml_path is not None
                else None
            ),
            "keyframe_name": self.entity_cfg.keyframe_name,
            "foot_geom_groups": [list(group) for group in self.foot_geom_groups],
            "actuator_mode": self.actuator_mode,
            "has_backlash": self.has_backlash,
            "head_body_ids": list(self.head_body_ids),
            "bam_parameters": (
                {
                    "model": "m6",
                    "motor": "xl330",
                    "kp_fw": self.bam_parameters.kp_fw,
                    "vin": self.bam_parameters.vin,
                    "vin_drop_gain": self.bam_parameters.vin_drop_gain,
                    "vin_min": self.bam_parameters.vin_min,
                }
                if self.bam_parameters is not None
                else None
            ),
        }


def _compile_bam_model(
    path: Path,
    *,
    parameters: BamM6Parameters,
) -> Any:
    """Compile the XML as the motor/friction model used by upstream BAM.

    BAM's controller supplies torque in ``data.ctrl``.  The XML position
    actuators are consequently converted to unit-gear torque motors, while
    MuJoCo's joint-friction fields are reserved for the per-step BAM budget.
    The initial non-zero friction values are intentional: ``mujoco-torch``
    specializes the constraint layout at ``device_put`` time and must see the
    DOF-friction rows before the first controller update.
    """

    spec = mujoco_api.MjSpec.from_file(str(path))
    # Match the active upstream ``FULL_COLLISION`` config. The source XML
    # supplies the collision geoms, while mjlab applies these per-geom solver
    # fields when it builds the robot entity. Omitting the priority changes
    # contact ordering when a sole is simultaneously touching several meshes.
    for geom in spec.geoms:
        if geom.name in {"left_foot_collision", "right_foot_collision"}:
            geom.condim = 3
            geom.priority = 1
            geom.friction = [1.0, 0.005, 0.0001]
    force_limit = parameters.force_limit(8.2)
    matched = 0
    for actuator in spec.actuators:
        target = actuator.target
        target_name = target if isinstance(target, str) else target.name
        if target_name.startswith("passive_"):
            continue
        actuator.set_to_motor()
        actuator.forcerange = (-force_limit, force_limit)
        actuator.forcelimited = True
        actuator.ctrllimited = False
        actuator.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint = spec.joint(target_name)
        joint.damping = np.zeros((3, 1))
        joint.frictionloss = parameters.friction_base
        joint.armature = parameters.armature
        joint.solref_friction = (-5.0e4, -2.0e2)
        joint.solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)
        matched += 1
    if matched != len(SERVO_JOINT_NAMES):
        raise ValueError(f"Expected {len(SERVO_JOINT_NAMES)} BAM actuators, got {matched}")
    return spec.compile()


def _move_mesh_geometry(model: Any, *, device: torch.device, dtype: torch.dtype) -> Any:
    """Move optional convex-mesh tensors and precomputed metadata to a device.

    Dtype conversion belongs in ``mujoco_torch.device_put``. This device-only
    step is retained because optional tuple fields and nested precomputed
    metadata are not reliably moved by ``TensorClass.to`` on every backend.
    """

    def move_tensor(value: torch.Tensor) -> torch.Tensor:
        value_dtype = getattr(value, "dtype", None)
        if isinstance(value_dtype, torch.dtype) and value_dtype.is_floating_point:
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)

    def move_optional(values: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(None if value is None else move_tensor(value) for value in values)

    convex_fields = {
        "geom_convex_vert": move_optional(model.geom_convex_vert),
        "geom_convex_facenormal": move_optional(model.geom_convex_facenormal),
        "mesh_convex": move_optional(model.mesh_convex),
    }
    # These fields are tuples of optional tensors. TensorClass's public
    # replacement path normalizes tuple contents back to the host, so update
    # the underlying field map to preserve accelerator devices.
    model._tensordict._tensordict.update(convex_fields)

    def move_auxiliary(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return move_tensor(value)
        if isinstance(value, dict):
            return {key: move_auxiliary(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(move_auxiliary(item) for item in value)
        if isinstance(value, list):
            return [move_auxiliary(item) for item in value]
        return value

    precomputed = getattr(model, "_device_precomp", None)
    if isinstance(precomputed, dict):
        moved_precomputed = move_auxiliary(precomputed)
        precomputed.clear()
        precomputed.update(moved_precomputed)
    return model


def load_microduck_model(
    xml_path: Path | None = None,
    *,
    entity_cfg: EntityCfg | None = None,
    entities: Mapping[str, EntityCfg] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    timestep: float = 0.005,
    decimation: int = 4,
    fixed_iterations: bool = False,
    solver_iterations: int | None = None,
    line_search_iterations: int | None = None,
    disable_contacts: bool = False,
    disable_mesh_mesh_contacts: bool = False,
    actuator_mode: str = "bam",
    bam_parameters: BamM6Parameters | None = None,
) -> MicroDuckModelBundle:
    """Load a MicroDuck model and convert it to a ``mujoco-torch`` model.

    ``actuator_mode="bam"`` is the upstream policy-training path.  The
    ``xml`` mode remains available as a diagnostic baseline for isolating
    actuator effects.
    """

    if entity_cfg is None:
        # Preserve the old direct API while making the default explicit.
        from ..robot.model_variants import MICRODUCK_WALK_ROBOT_CFG

        entity_cfg = MICRODUCK_WALK_ROBOT_CFG
    scene_entities = dict(entities or {entity_cfg.name: entity_cfg})
    if entity_cfg.name not in scene_entities:
        scene_entities[entity_cfg.name] = entity_cfg
    path = (xml_path or entity_cfg.load_path or default_scene_path()).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if actuator_mode not in {"bam", "xml"}:
        raise ValueError("actuator_mode must be 'bam' or 'xml'")
    bam_config = bam_parameters or BamM6Parameters()
    native_model = (
        _compile_bam_model(path, parameters=bam_config)
        if actuator_mode == "bam"
        else mujoco_api.MjModel.from_xml_path(str(path))
    )
    native_model.opt.timestep = timestep
    if solver_iterations is not None:
        if solver_iterations < 1:
            raise ValueError("solver_iterations must be positive")
        native_model.opt.iterations = solver_iterations
    if line_search_iterations is not None:
        if line_search_iterations < 1:
            raise ValueError("line_search_iterations must be positive")
        native_model.opt.ls_iterations = line_search_iterations
    if disable_contacts:
        native_model.opt.disableflags |= int(mujoco_api.mjtDisableBit.mjDSBL_CONTACT)
    device_obj = torch.device(device)
    expected_actuator_names = entity_cfg.actuator_joint_names or SERVO_JOINT_NAMES
    if native_model.nu != len(expected_actuator_names):
        raise ValueError(
            f"Expected {len(expected_actuator_names)} actuators, got {native_model.nu}"
        )

    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    actuator_joint_names: list[str] = []
    for actuator_id in range(native_model.nu):
        joint_id = int(native_model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            raise ValueError(f"Actuator {actuator_id} has no joint transmission")
        qpos_indices.append(int(native_model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(native_model.jnt_dofadr[joint_id]))
        actuator_joint_names.append(
            mujoco_api.mj_id2name(native_model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_id)
        )
    if tuple(actuator_joint_names) != expected_actuator_names:
        raise ValueError(
            "Unexpected actuator order: "
            f"{tuple(actuator_joint_names)!r}; expected {expected_actuator_names!r}"
        )

    # Backlash variants add one passive hinge after each servo.  The actuator
    # still drives the servo-side DOF, while the real encoder and firmware
    # position loop read the output-side angle (servo + passive play).  Keep
    # zero-safe addresses and a mask so the same code handles both XML forms.
    backlash_qpos_indices: list[int] = []
    backlash_qvel_indices: list[int] = []
    backlash_mask: list[float] = []
    for servo_name, servo_qpos, servo_qvel in zip(
        SERVO_JOINT_NAMES, qpos_indices, qvel_indices, strict=True
    ):
        backlash_name = f"passive_{servo_name}_backlash"
        backlash_id = mujoco_api.mj_name2id(
            native_model, mujoco_api.mjtObj.mjOBJ_JOINT, backlash_name
        )
        if backlash_id < 0:
            backlash_qpos_indices.append(servo_qpos)
            backlash_qvel_indices.append(servo_qvel)
            backlash_mask.append(0.0)
        else:
            backlash_qpos_indices.append(int(native_model.jnt_qposadr[backlash_id]))
            backlash_qvel_indices.append(int(native_model.jnt_dofadr[backlash_id]))
            backlash_mask.append(1.0)

    if entity_cfg.keyframe_name is not None:
        key_id = mujoco_api.mj_name2id(
            native_model, mujoco_api.mjtObj.mjOBJ_KEY, entity_cfg.keyframe_name
        )
        if key_id < 0:
            if any(cfg.kind != "robot" for cfg in scene_entities.values()):
                # Upstream prop scenes such as scene_ball intentionally omit
                # keyframes because a free joint changes the full qpos size.
                # Task reset terms own the initial placement in that case.
                default_qpos = np.asarray(native_model.qpos0, dtype=np.float64).copy()
            else:
                raise ValueError(
                    f"Required keyframe {entity_cfg.keyframe_name!r} was not found in {path}"
                )
        else:
            default_qpos = np.asarray(native_model.key_qpos[key_id], dtype=np.float64)
    else:
        # A free-body entity such as the future BallKick prop may have no
        # named keyframe. MuJoCo's qpos0 is the correct neutral fallback.
        default_qpos = np.asarray(native_model.qpos0, dtype=np.float64).copy()
    model = mujoco_torch.device_put(native_model, dtype=dtype).to(device_obj)
    model = _move_mesh_geometry(model, device=device_obj, dtype=dtype)
    if disable_mesh_mesh_contacts:
        model._device_precomp["skip_mesh_mesh_contacts"] = True
    entity_views = {
        name: _resolve_entity_view(native_model, cfg, device=device_obj)
        for name, cfg in scene_entities.items()
    }
    robot_view = entity_views.get("robot") or entity_views[entity_cfg.name]
    trunk_body_id = robot_view.root_body_id
    head_body_ids = tuple(
        int(body_id)
        for body_name in entity_cfg.head_body_names
        if (body_id := mujoco_api.mj_name2id(native_model, mujoco_api.mjtObj.mjOBJ_BODY, body_name))
        >= 0
    )
    if entity_cfg.foot_site_selector is None:
        foot_site_ids_resolved: tuple[int, ...] = ()
        foot_geom_groups: tuple[tuple[int, ...], ...] = ()
    else:
        foot_site_ids_resolved = _resolve_selector(
            native_model,
            mujoco_api.mjtObj.mjOBJ_SITE,
            entity_cfg.foot_site_selector,
        )
        if len(foot_site_ids_resolved) != 2:
            raise ValueError(
                "The current velocity observation contract requires exactly two foot sites; "
                f"selector resolved {foot_site_ids_resolved!r}"
            )
        if entity_cfg.foot_contact_selectors is None:
            raise ValueError("Foot contact selectors are required when foot sites are configured")
        foot_geom_groups = tuple(
            _resolve_selector(native_model, mujoco_api.mjtObj.mjOBJ_GEOM, selector)
            for selector in entity_cfg.foot_contact_selectors
        )
        if any(not group for group in foot_geom_groups):
            raise ValueError("Each foot contact selector must resolve at least one geometry")
    return MicroDuckModelBundle(
        xml_path=path,
        entity_cfg=entity_cfg,
        native_model=native_model,
        torch_model=model,
        device=device_obj,
        dtype=dtype,
        timestep=timestep,
        decimation=decimation,
        fixed_iterations=fixed_iterations,
        solver_iterations=int(native_model.opt.iterations),
        line_search_iterations=int(native_model.opt.ls_iterations),
        contacts_enabled=not disable_contacts,
        qpos_indices=torch.tensor(qpos_indices, dtype=torch.long, device=device_obj),
        qvel_indices=torch.tensor(qvel_indices, dtype=torch.long, device=device_obj),
        default_qpos=torch.tensor(default_qpos, dtype=dtype, device=device_obj),
        backlash_qpos_indices=torch.tensor(
            backlash_qpos_indices, dtype=torch.long, device=device_obj
        ),
        backlash_qvel_indices=torch.tensor(
            backlash_qvel_indices, dtype=torch.long, device=device_obj
        ),
        backlash_mask=torch.tensor(backlash_mask, dtype=dtype, device=device_obj),
        default_pose=torch.tensor(default_qpos[qpos_indices], dtype=dtype, device=device_obj),
        actuator_joint_names=tuple(actuator_joint_names),
        sensor_slices={
            name: slice(
                int(
                    native_model.sensor_adr[
                        _required_id(native_model, mujoco_api.mjtObj.mjOBJ_SENSOR, name)
                    ]
                ),
                int(
                    native_model.sensor_adr[
                        _required_id(native_model, mujoco_api.mjtObj.mjOBJ_SENSOR, name)
                    ]
                )
                + int(
                    native_model.sensor_dim[
                        _required_id(native_model, mujoco_api.mjtObj.mjOBJ_SENSOR, name)
                    ]
                ),
            )
            for name in SENSOR_NAMES
            if mujoco_api.mj_name2id(native_model, mujoco_api.mjtObj.mjOBJ_SENSOR, name) >= 0
        },
        trunk_body_id=trunk_body_id,
        head_body_ids=head_body_ids,
        foot_site_ids=tuple(int(value) for value in foot_site_ids_resolved),
        foot_geom_ids=tuple(int(group[0]) for group in foot_geom_groups),
        foot_geom_groups=tuple(tuple(int(value) for value in group) for group in foot_geom_groups),
        collision_geom_ids=tuple(
            geom_id
            for geom_id in robot_view.geom_ids
            if (
                (name := mujoco_api.mj_id2name(native_model, mujoco_api.mjtObj.mjOBJ_GEOM, geom_id))
                is not None
                and name.endswith(entity_cfg.collision_name_suffix)
            )
        ),
        actuator_mode=actuator_mode,
        bam_parameters=bam_config if actuator_mode == "bam" else None,
        friction_dof_count=(
            int(np.count_nonzero(native_model.dof_frictionloss > 0))
            if actuator_mode == "bam"
            else 0
        ),
        entities=entity_views,
        action_size=int(native_model.nu),
    )


# Public generic name for new task code.  Keep the historical name above for
# downstream callers and the existing policy/parity tools.
ModelBundle = MicroDuckModelBundle
