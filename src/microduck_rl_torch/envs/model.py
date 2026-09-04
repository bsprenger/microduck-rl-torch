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
    default_pose: torch.Tensor
    actuator_joint_names: tuple[str, ...]
    sensor_slices: dict[str, slice]
    trunk_body_id: int
    foot_site_ids: tuple[int, int]

    @property
    def observation_size(self) -> int:
        return 61

    @property
    def action_size(self) -> int:
        return 14

    def new_data(self) -> Any:
        """Create a forward-computed standing state on the target device."""

        data = mujoco_torch.make_data(self.torch_model)
        qpos = torch.as_tensor(
            np.asarray(self.native_model.key("STAND").qpos),
            dtype=self.dtype,
            device=self.device,
        )
        qvel = torch.zeros_like(data.qvel)
        ctrl = self.default_pose.clone()
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
        }


def load_microduck_model(
    xml_path: Path | None = None,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    timestep: float = 0.005,
    decimation: int = 4,
    fixed_iterations: bool = False,
    solver_iterations: int | None = None,
    line_search_iterations: int | None = None,
    disable_contacts: bool = False,
    disable_mesh_mesh_contacts: bool = False,
) -> MicroDuckModelBundle:
    """Load the nominal walk model and convert it to a `mujoco-torch` model."""

    path = (xml_path or default_scene_path()).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    native_model = mujoco_api.MjModel.from_xml_path(str(path))
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

    stand_id = _required_id(native_model, mujoco_api.mjtObj.mjOBJ_KEY, "STAND")
    stand_qpos = np.asarray(native_model.key_qpos[stand_id], dtype=np.float64)
    model = mujoco_torch.device_put(native_model, dtype=dtype).to(device_obj)
    if disable_mesh_mesh_contacts:
        model._device_precomp["skip_mesh_mesh_contacts"] = True
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
        trunk_body_id=_required_id(native_model, mujoco_api.mjtObj.mjOBJ_BODY, "trunk_base"),
        foot_site_ids=(
            _required_id(native_model, mujoco_api.mjtObj.mjOBJ_SITE, "left_foot"),
            _required_id(native_model, mujoco_api.mjtObj.mjOBJ_SITE, "right_foot"),
        ),
    )
