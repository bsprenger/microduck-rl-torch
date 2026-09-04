"""Load and fingerprint the MicroDuck MuJoCo model for torch execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco_torch
import numpy as np
import torch

from .actuation import BamM6Parameters

mujoco_api: Any = mujoco

SERVO_JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
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


@dataclass(frozen=True)
class MicroDuckModelBundle:
    """Native and device-resident representations of one MicroDuck model."""

    xml_path: Path
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
    backlash_qpos_indices: torch.Tensor
    backlash_qvel_indices: torch.Tensor
    backlash_mask: torch.Tensor
    default_pose: torch.Tensor
    actuator_joint_names: tuple[str, ...]
    sensor_slices: dict[str, slice]
    trunk_body_id: int
    head_body_ids: tuple[int, ...]
    foot_site_ids: tuple[int, int]
    foot_geom_ids: tuple[int, int]
    collision_geom_ids: tuple[int, ...]
    actuator_mode: str
    bam_parameters: BamM6Parameters | None
    friction_dof_count: int

    @property
    def observation_size(self) -> int:
        return 61

    @property
    def action_size(self) -> int:
        return 14

    @property
    def has_backlash(self) -> bool:
        return bool(self.backlash_mask.any().item())

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
            np.asarray(self.native_model.key("STAND").qpos),
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


def load_microduck_model(
    xml_path: Path | None = None,
    *,
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

    path = (xml_path or default_scene_path()).resolve()
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
    if native_model.nu != len(SERVO_JOINT_NAMES):
        raise ValueError(f"Expected 14 actuators, got {native_model.nu}")

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
    if tuple(actuator_joint_names) != SERVO_JOINT_NAMES:
        raise ValueError(
            "Unexpected actuator order: "
            f"{tuple(actuator_joint_names)!r}; expected {SERVO_JOINT_NAMES!r}"
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

    stand_id = _required_id(native_model, mujoco_api.mjtObj.mjOBJ_KEY, "STAND")
    stand_qpos = np.asarray(native_model.key_qpos[stand_id], dtype=np.float64)
    model = mujoco_torch.device_put(native_model, dtype=dtype).to(device_obj)
    if disable_mesh_mesh_contacts:
        model._device_precomp["skip_mesh_mesh_contacts"] = True
    trunk_body_id = _required_id(native_model, mujoco_api.mjtObj.mjOBJ_BODY, "trunk_base")
    head_body_ids = tuple(
        int(body_id)
        for body_name in (
            "neck",
            "neck_pitch",
            "yaw_roll_motion",
            "bottom_head_shell",
            "jaw_soft",
            "bearing_roll",
        )
        if (body_id := mujoco_api.mj_name2id(native_model, mujoco_api.mjtObj.mjOBJ_BODY, body_name))
        >= 0
    )
    return MicroDuckModelBundle(
        xml_path=path,
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
        backlash_qpos_indices=torch.tensor(
            backlash_qpos_indices, dtype=torch.long, device=device_obj
        ),
        backlash_qvel_indices=torch.tensor(
            backlash_qvel_indices, dtype=torch.long, device=device_obj
        ),
        backlash_mask=torch.tensor(backlash_mask, dtype=dtype, device=device_obj),
        default_pose=torch.tensor(stand_qpos[qpos_indices], dtype=dtype, device=device_obj),
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
        },
        trunk_body_id=trunk_body_id,
        head_body_ids=head_body_ids,
        foot_site_ids=(
            _required_id(native_model, mujoco_api.mjtObj.mjOBJ_SITE, "left_foot"),
            _required_id(native_model, mujoco_api.mjtObj.mjOBJ_SITE, "right_foot"),
        ),
        foot_geom_ids=(
            _required_id(native_model, mujoco_api.mjtObj.mjOBJ_GEOM, "left_foot_collision"),
            _required_id(native_model, mujoco_api.mjtObj.mjOBJ_GEOM, "right_foot_collision"),
        ),
        collision_geom_ids=tuple(
            geom_id
            for geom_id in range(native_model.ngeom)
            if (
                (name := mujoco_api.mj_id2name(native_model, mujoco_api.mjtObj.mjOBJ_GEOM, geom_id))
                is not None
                and name.endswith("_collision")
            )
        ),
        actuator_mode=actuator_mode,
        bam_parameters=bam_config if actuator_mode == "bam" else None,
        friction_dof_count=(
            int(np.count_nonzero(native_model.dof_frictionloss > 0))
            if actuator_mode == "bam"
            else 0
        ),
    )
