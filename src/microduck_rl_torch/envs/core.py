"""Manager-based task environment and lifecycle implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..rendering.config import RenderConfig
from .managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    EventManager,
    ModelMutationManager,
    ObservationManager,
    ResetManager,
    RewardManager,
    SensorState,
    SensorStateManager,
    TaskStateManager,
    TerminationManager,
    TransitionData,
)
from .model import (
    ModelBundle,
    _joint_qpos_width,
    _joint_qvel_width,
    load_model_bundle,
)
from .physics import BatchedPhysicsBackend, PhysicsBackend
from .scene import SceneBuild, SceneBuilder, TerrainManager
from .sensors import SensorManager
from .task_config import TaskEnvCfg


@dataclass(frozen=True)
class EnvStep:
    observation: torch.Tensor | dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: bool | torch.Tensor
    truncated: bool | torch.Tensor
    info: dict[str, Any]


@dataclass
class EnvironmentState:
    """Generic environment state owned by ``ManagerBasedTaskEnv``."""

    sensors: SensorState
    reward_terms: dict[str, torch.Tensor]
    manager_data: dict[str, Any] = field(default_factory=dict)
    task_data: dict[str, Any] = field(default_factory=dict)
    transition: TransitionData | None = None
    pending_reset: bool | torch.Tensor = False


class _EntityDataView:
    """Tensor-backed entity data view for task terms."""

    def __init__(self, env: ManagerBasedTaskEnv, view: Any) -> None:
        self._env = env
        self._view = view

    @property
    def default_joint_pos(self) -> torch.Tensor:
        value = self._env.bundle.default_qpos.index_select(-1, self._view.non_free_qpos_indices)
        return (
            value.unsqueeze(0).expand(self._env.num_envs, -1).clone()
            if self._env.num_envs > 1
            else value
        )

    @property
    def default_joint_vel(self) -> torch.Tensor:
        value = self._env.bundle.default_qvel.index_select(-1, self._view.non_free_qvel_indices)
        return (
            value.unsqueeze(0).expand(self._env.num_envs, -1).clone()
            if self._env.num_envs > 1
            else value
        )

    @property
    def default_root_state(self) -> torch.Tensor:
        if self._view.free_qpos_indices.numel() != 7:
            raise ValueError(f"Entity {self._view.name!r} has no free root state")
        pose = self._env.bundle.default_qpos.index_select(-1, self._view.free_qpos_indices)
        velocity = self._env.bundle.default_qvel.index_select(-1, self._view.free_qvel_indices)
        value = torch.cat((pose, velocity), dim=-1)
        return (
            value.unsqueeze(0).expand(self._env.num_envs, -1).clone()
            if self._env.num_envs > 1
            else value
        )

    @property
    def is_fixed_base(self) -> bool:
        return not bool(self._view.free_qpos_indices.numel())

    @property
    def is_articulated(self) -> bool:
        return bool(self._view.non_free_joint_ids)

    @property
    def is_actuated(self) -> bool:
        return bool(self._view.actuator_ids)

    @property
    def encoder_bias(self) -> torch.Tensor:
        if self._env.state is None:
            return torch.zeros_like(self.default_joint_pos)
        # The Microduck encoder-bias vector is in actuator order. Entity data
        # exposes the bias in articulation-joint order; map only the entity's
        # actuated joints and use zero for passive joints.
        result = torch.zeros_like(self.default_joint_pos)
        actuator_by_joint = {
            int(self._env.bundle.native_model.actuator_trnid[actuator_id, 0]): index
            for index, actuator_id in enumerate(self._view.actuator_ids)
        }
        for index, joint_id in enumerate(self._view.non_free_joint_ids):
            actuator_index = actuator_by_joint.get(joint_id)
            if (
                actuator_index is not None
                and actuator_index < self._env.state.sensors.encoder_bias.numel()
            ):
                bias = self._env.state.sensors.encoder_bias[..., actuator_index]
                result[..., index] = bias
        return result

    def _require_data(self) -> Any:
        data = self._env.data
        if data is None:
            raise RuntimeError("Call reset() before reading entity data")
        return data

    @staticmethod
    def _flat(value: torch.Tensor, *, name: str, size: int) -> torch.Tensor:
        value = torch.as_tensor(value).reshape(-1)
        if value.numel() != size:
            raise ValueError(f"{name} expected {size} values, got {value.numel()}")
        return value

    def _replace(self, **fields: Any) -> None:
        data = self._require_data()
        self._env.data = data.replace(**fields)

    def _ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor | None:
        if env_ids is None:
            return None
        if self._env.num_envs == 1:
            ids = torch.as_tensor([0], dtype=torch.long, device=self._env.bundle.device)
            requested = (
                torch.arange(1, dtype=torch.long, device=self._env.bundle.device)[env_ids]
                if isinstance(env_ids, slice)
                else torch.as_tensor(env_ids, dtype=torch.long, device=self._env.bundle.device)
            ).reshape(-1)
            if requested.numel() != 1 or int(requested.item()) != 0:
                raise ValueError("A scalar environment only accepts env_ids=[0]")
            return ids
        ids = (
            torch.arange(self._env.num_envs, dtype=torch.long, device=self._env.bundle.device)[
                env_ids
            ]
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, dtype=torch.long, device=self._env.bundle.device)
        ).reshape(-1)
        if ids.numel() and (ids.min() < 0 or ids.max() >= self._env.num_envs):
            raise ValueError("env_ids contains an out-of-range environment")
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError("env_ids contains duplicates")
        return ids

    def _rows(
        self,
        value: torch.Tensor,
        *,
        width: int,
        name: str,
        env_ids: torch.Tensor | slice | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalize a scalar or selected batched writer value to row form."""

        value = torch.as_tensor(value)
        ids = self._ids(env_ids)
        expected_rows = self._env.num_envs if ids is None else int(ids.numel())
        if value.ndim == 1:
            if value.numel() != width:
                raise ValueError(f"{name} expected {width} values, got {value.numel()}")
            if expected_rows != 1:
                raise ValueError(f"{name} must provide one row per selected environment")
            return value.reshape(1, width), ids
        if value.ndim != 2 or value.shape[1] != width or value.shape[0] != expected_rows:
            raise ValueError(
                f"{name} must have shape ({expected_rows}, {width}), got {tuple(value.shape)}"
            )
        return value, ids

    def _scatter_rows(
        self,
        current: torch.Tensor,
        values: torch.Tensor,
        ids: torch.Tensor | None,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        result = current.clone()
        if self._env.num_envs == 1:
            result[indices] = values[0]
        elif ids is None:
            result[:, indices] = values
        else:
            result[ids[:, None], indices] = values
        return result

    def write_root_pose(
        self, pose: torch.Tensor, env_ids: torch.Tensor | slice | None = None
    ) -> None:
        indices = self._view.free_qpos_indices
        if indices.numel() != 7:
            raise ValueError(f"Entity {self._view.name!r} has no free root pose")
        qpos = self._require_data().qpos.clone()
        values, ids = self._rows(
            torch.as_tensor(pose, dtype=qpos.dtype, device=qpos.device),
            width=7,
            name="root pose",
            env_ids=env_ids,
        )
        qpos = self._scatter_rows(qpos, values, ids, indices)
        self._replace(qpos=qpos)

    def write_root_velocity(
        self, velocity: torch.Tensor, env_ids: torch.Tensor | slice | None = None
    ) -> None:
        indices = self._view.free_qvel_indices
        if indices.numel() != 6:
            raise ValueError(f"Entity {self._view.name!r} has no free root velocity")
        data = self._require_data()
        velocity = torch.as_tensor(velocity, dtype=data.qvel.dtype, device=data.qvel.device)
        velocity, ids = self._rows(velocity, width=6, name="root velocity", env_ids=env_ids)
        if self._env.num_envs == 1:
            qpos_rows = data.qpos.unsqueeze(0)
        elif ids is None:
            qpos_rows = data.qpos
        else:
            qpos_rows = data.qpos[ids]
        qvel = data.qvel.clone()
        angular_velocity = velocity[..., 3:]
        quat = qpos_rows[..., self._view.free_qpos_indices[3:]]
        # MuJoCo stores free-joint angular velocity in the body frame.  The
        # public entity contract accepts world-frame angular velocity.
        # Read the just-written qpos, not the derived xquat buffer; callers
        # are allowed to batch root pose and velocity writes before forward().
        inverse = torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)
        t = 2.0 * torch.cross(inverse[..., 1:], angular_velocity, dim=-1)
        body_angular = (
            angular_velocity + inverse[..., :1] * t + torch.cross(inverse[..., 1:], t, dim=-1)
        )
        converted = torch.cat((velocity[..., :3], body_angular), dim=-1)
        qvel = self._scatter_rows(qvel, converted, ids, indices)
        self._replace(qvel=qvel)

    def write_root_com_velocity(
        self, velocity: torch.Tensor, env_ids: torch.Tensor | slice | None = None
    ) -> None:
        """Write world-frame velocity of the root body's center of mass.

        MuJoCo's free-joint coordinates describe the root link origin. The
        entity API accepts COM velocity, so convert the requested
        linear COM velocity to link-origin velocity before using the common
        root writer.
        """

        if self._view.free_qvel_indices.numel() != 6:
            raise ValueError(f"Entity {self._view.name!r} has no free root velocity")
        data = self._require_data()
        values, ids = self._rows(
            torch.as_tensor(velocity, dtype=data.qvel.dtype, device=data.qvel.device),
            width=6,
            name="root COM velocity",
            env_ids=env_ids,
        )
        if self._env.num_envs == 1:
            qpos_rows = data.qpos.unsqueeze(0)
        elif ids is None:
            qpos_rows = data.qpos
        else:
            qpos_rows = data.qpos[ids]
        quat = qpos_rows[..., self._view.free_qpos_indices[3:]]
        offset_b = torch.as_tensor(
            self._env.bundle.native_model.body_ipos[self._view.root_body_id],
            dtype=data.qvel.dtype,
            device=data.qvel.device,
        )
        xyz = quat[..., 1:]
        twice = 2.0 * torch.cross(xyz, offset_b.expand_as(xyz), dim=-1)
        offset_w = offset_b + quat[..., :1] * twice + torch.cross(xyz, twice, dim=-1)
        link_linear = values[..., :3] - torch.cross(values[..., 3:], offset_w, dim=-1)
        self.write_root_velocity(torch.cat((link_linear, values[..., 3:]), dim=-1), env_ids)

    def write_root_state(
        self, state: torch.Tensor, env_ids: torch.Tensor | slice | None = None
    ) -> None:
        state = torch.as_tensor(
            state, dtype=self._require_data().qpos.dtype, device=self._require_data().qpos.device
        )
        rows, _ = self._rows(state, width=13, name="root state", env_ids=env_ids)
        self.write_root_pose(rows[..., :7], env_ids=env_ids)
        self.write_root_velocity(rows[..., 7:], env_ids=env_ids)

    def _joint_selection(self, joint_ids: torch.Tensor | slice | None) -> tuple[torch.Tensor, ...]:
        all_ids = self._view.non_free_joint_ids
        if joint_ids is None:
            selected = all_ids
        elif isinstance(joint_ids, slice):
            selected = all_ids[joint_ids]
        else:
            indices = torch.as_tensor(joint_ids, dtype=torch.long).reshape(-1).tolist()
            if any(index < 0 or index >= len(all_ids) for index in indices):
                raise ValueError("Joint selection is outside this entity")
            selected = tuple(all_ids[index] for index in indices)
        return selected

    def write_joint_position(
        self,
        position: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        selected = self._joint_selection(joint_ids)
        qpos = self._require_data().qpos.clone()
        values = torch.as_tensor(position, dtype=qpos.dtype, device=qpos.device)
        expected_width = sum(
            _joint_qpos_width(int(self._env.bundle.native_model.jnt_type[joint_id]))
            for joint_id in selected
        )
        values, ids = self._rows(
            values, width=expected_width, name="joint position", env_ids=env_ids
        )
        widths = [
            _joint_qpos_width(int(self._env.bundle.native_model.jnt_type[joint_id]))
            for joint_id in selected
        ]
        offset = 0
        for joint_id, width in zip(selected, widths, strict=True):
            start = int(self._env.bundle.native_model.jnt_qposadr[joint_id])
            qpos = self._scatter_rows(
                qpos,
                values[..., offset : offset + width],
                ids,
                torch.arange(start, start + width, device=qpos.device),
            )
            offset += width
        self._replace(qpos=qpos)

    def write_joint_velocity(
        self,
        velocity: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        selected = self._joint_selection(joint_ids)
        qvel = self._require_data().qvel.clone()
        values = torch.as_tensor(velocity, dtype=qvel.dtype, device=qvel.device)
        expected_width = sum(
            _joint_qvel_width(int(self._env.bundle.native_model.jnt_type[joint_id]))
            for joint_id in selected
        )
        values, ids = self._rows(
            values, width=expected_width, name="joint velocity", env_ids=env_ids
        )
        widths = [
            _joint_qvel_width(int(self._env.bundle.native_model.jnt_type[joint_id]))
            for joint_id in selected
        ]
        offset = 0
        for joint_id, width in zip(selected, widths, strict=True):
            start = int(self._env.bundle.native_model.jnt_dofadr[joint_id])
            qvel = self._scatter_rows(
                qvel,
                values[..., offset : offset + width],
                ids,
                torch.arange(start, start + width, device=qvel.device),
            )
            offset += width
        self._replace(qvel=qvel)

    def write_joint_state(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        self.write_joint_position(position, joint_ids, env_ids)
        self.write_joint_velocity(velocity, joint_ids, env_ids)

    def write_external_wrench(
        self,
        forces: torch.Tensor | None,
        torques: torch.Tensor | None,
        body_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        data = self._require_data()
        if body_ids is None:
            selected = list(self._view.body_ids)
        elif isinstance(body_ids, slice):
            selected = list(self._view.body_ids[body_ids])
        else:
            indices = torch.as_tensor(body_ids, dtype=torch.long).reshape(-1).tolist()
            selected = [self._view.body_ids[index] for index in indices]
        wrench = data.xfrc_applied.clone()

        def rows(value: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
            value = torch.as_tensor(value, dtype=wrench.dtype, device=wrench.device)
            ids = self._ids(env_ids)
            expected_rows = self._env.num_envs if ids is None else int(ids.numel())
            count = len(selected)
            if value.ndim == 1:
                if value.numel() != 3 * count or expected_rows != 1:
                    raise ValueError(f"{name} must have {3 * count} values for one selected row")
                return value.reshape(1, count, 3), ids
            if value.ndim == 2 and value.shape == (count, 3):
                if expected_rows != 1:
                    raise ValueError(f"{name} must provide one row per selected environment")
                return value.unsqueeze(0), ids
            if value.ndim == 2 and value.shape == (expected_rows, 3 * count):
                return value.reshape(expected_rows, count, 3), ids
            if value.ndim == 3 and value.shape == (expected_rows, count, 3):
                return value, ids
            raise ValueError(
                f"{name} must have shape ({count}, 3), ({expected_rows}, {3 * count}), "
                f"or ({expected_rows}, {count}, 3); got {tuple(value.shape)}"
            )

        def assign(values: torch.Tensor, ids: torch.Tensor | None, component: slice) -> None:
            if self._env.num_envs == 1:
                wrench[selected, component] = values[0]
            elif ids is None:
                wrench[:, selected, component] = values
            else:
                wrench[ids[:, None], selected, component] = values

        if forces is not None:
            values, ids = rows(forces, "External force")
            assign(values, ids, slice(0, 3))
        if torques is not None:
            values, ids = rows(torques, "External torque")
            assign(values, ids, slice(3, 6))
        self._replace(xfrc_applied=wrench)

    def write_ctrl(
        self,
        ctrl: torch.Tensor,
        ctrl_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if ctrl_ids is None:
            selected = list(self._view.actuator_ids)
        elif isinstance(ctrl_ids, slice):
            selected = list(self._view.actuator_ids[ctrl_ids])
        else:
            indices = torch.as_tensor(ctrl_ids, dtype=torch.long).reshape(-1).tolist()
            selected = [self._view.actuator_ids[index] for index in indices]
        control_value = self._require_data().ctrl
        values = torch.as_tensor(ctrl, dtype=control_value.dtype, device=control_value.device)
        ids = self._ids(env_ids)
        expected_rows = self._env.num_envs if ids is None else int(ids.numel())
        if values.ndim == 1:
            if values.numel() != len(selected) or expected_rows != 1:
                raise ValueError("Control values must provide one row per selected environment")
            values = values.unsqueeze(0)
        elif values.ndim != 2 or values.shape != (expected_rows, len(selected)):
            raise ValueError(
                f"Controls must have shape ({expected_rows}, {len(selected)}), "
                f"got {tuple(values.shape)}"
            )
        control = self._require_data().ctrl.clone()
        if self._env.num_envs == 1:
            control[selected] = values[0]
        elif ids is None:
            control[:, selected] = values
        else:
            control[ids[:, None], selected] = values
        self._replace(ctrl=control)

    def clear_state(self, env_ids: torch.Tensor | slice | None = None) -> None:
        data = self._require_data()
        applied = data.xfrc_applied.clone()
        ids = self._ids(env_ids)
        if self._env.num_envs == 1 or ids is None:
            applied.zero_()
        else:
            applied[ids] = 0
        self._replace(xfrc_applied=applied)

    @staticmethod
    def _quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
        """Convert one or more rotation matrices to ``wxyz`` quaternions."""

        # The backend currently runs one environment.  Flattening first keeps
        # the implementation correct for an eventual leading environment
        # dimension without relying on fragile boolean indexing of 0-D tensors.
        flat = matrix.reshape(-1, 3, 3)
        result: list[torch.Tensor] = []
        for value in flat:
            trace = value.trace()
            if bool(trace > 0):
                root = torch.sqrt(torch.clamp(trace + 1.0, min=1.0e-12)) * 2.0
                quat = torch.stack(
                    (
                        0.25 * root,
                        (value[2, 1] - value[1, 2]) / root,
                        (value[0, 2] - value[2, 0]) / root,
                        (value[1, 0] - value[0, 1]) / root,
                    )
                )
            else:
                diagonal = torch.diagonal(value)
                index = int(diagonal.argmax())
                if index == 0:
                    root = (
                        torch.sqrt(torch.clamp(1.0 + 2.0 * value[0, 0] - trace, min=1.0e-12)) * 2.0
                    )
                    quat = torch.stack(
                        (
                            (value[2, 1] - value[1, 2]) / root,
                            0.25 * root,
                            (value[0, 1] + value[1, 0]) / root,
                            (value[0, 2] + value[2, 0]) / root,
                        )
                    )
                elif index == 1:
                    root = (
                        torch.sqrt(torch.clamp(1.0 + 2.0 * value[1, 1] - trace, min=1.0e-12)) * 2.0
                    )
                    quat = torch.stack(
                        (
                            (value[0, 2] - value[2, 0]) / root,
                            (value[0, 1] + value[1, 0]) / root,
                            0.25 * root,
                            (value[1, 2] + value[2, 1]) / root,
                        )
                    )
                else:
                    root = (
                        torch.sqrt(torch.clamp(1.0 + 2.0 * value[2, 2] - trace, min=1.0e-12)) * 2.0
                    )
                    quat = torch.stack(
                        (
                            (value[1, 0] - value[0, 1]) / root,
                            (value[0, 2] + value[2, 0]) / root,
                            (value[1, 2] + value[2, 1]) / root,
                            0.25 * root,
                        )
                    )
            result.append(quat)
        return torch.stack(result).reshape(*matrix.shape[:-2], 4)

    @staticmethod
    def _quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Multiply ``[w, x, y, z]`` quaternions with broadcast support."""

        w1, x1, y1, z1 = first.unbind(dim=-1)
        w2, x2, y2, z2 = second.unbind(dim=-1)
        return torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dim=-1,
        )

    @staticmethod
    def _world_velocity(
        data: Any,
        body_ids: tuple[int, ...],
        subtree_com: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions is None:
            positions = data.xpos[..., list(body_ids), :]
        cvel = data.cvel[..., list(body_ids), :]
        linear_c = cvel[..., 3:6]
        angular_c = cvel[..., :3]
        # ``cvel`` is the MuJoCo com-based velocity about the root subtree
        # COM. Keep the same transport convention as the native MuJoCo
        # frame-velocity sensors.
        offset = subtree_com.unsqueeze(-2) - positions
        linear_w = linear_c - torch.cross(angular_c, offset, dim=-1)
        return torch.cat((linear_w, angular_c), dim=-1)

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        data = self._require_data()
        body = self._view.root_body_id
        return torch.cat((data.xpos[..., body, :], data.xquat[..., body, :]), dim=-1)

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        data = self._require_data()
        return self._world_velocity(
            data,
            (self._view.root_body_id,),
            data.subtree_com[..., self._view.root_body_id, :],
        )[..., 0, :]

    @property
    def root_com_pose_w(self) -> torch.Tensor:
        data = self._require_data()
        body = self._view.root_body_id
        body_iquat = self._env.bundle.torch_model.body_iquat[body]
        quat = self._quat_mul(data.xquat[..., body, :], body_iquat)
        return torch.cat((data.xipos[..., body, :], quat), dim=-1)

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        data = self._require_data()
        return self._world_velocity(
            data,
            (self._view.root_body_id,),
            data.subtree_com[..., self._view.root_body_id, :],
            positions=data.xipos[..., self._view.root_body_id, :].unsqueeze(-2),
        )[..., 0, :]

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        data = self._require_data()
        return torch.cat(
            (
                data.xpos[..., list(self._view.body_ids), :],
                data.xquat[..., list(self._view.body_ids), :],
            ),
            dim=-1,
        )

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        data = self._require_data()
        return self._world_velocity(
            data,
            self._view.body_ids,
            data.subtree_com[..., self._view.root_body_id, :],
        )

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        data = self._require_data()
        body_ids = list(self._view.body_ids)
        body_iquat = self._env.bundle.torch_model.body_iquat[body_ids]
        return torch.cat(
            (
                data.xipos[..., body_ids, :],
                self._quat_mul(data.xquat[..., body_ids, :], body_iquat),
            ),
            dim=-1,
        )

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        data = self._require_data()
        return self._world_velocity(
            data,
            self._view.body_ids,
            data.subtree_com[..., self._view.root_body_id, :],
            positions=data.xipos[..., list(self._view.body_ids), :],
        )

    @property
    def geom_vel_w(self) -> torch.Tensor:
        data = self._require_data()
        body_ids = tuple(
            int(self._env.bundle.native_model.geom_bodyid[geom_id])
            for geom_id in self._view.geom_ids
        )
        return self._world_velocity(
            data,
            body_ids,
            data.subtree_com[..., self._view.root_body_id, :],
            positions=data.geom_xpos[..., list(self._view.geom_ids), :],
        )

    @property
    def site_vel_w(self) -> torch.Tensor:
        data = self._require_data()
        body_ids = tuple(
            int(self._env.bundle.native_model.site_bodyid[site_id])
            for site_id in self._view.site_ids
        )
        return self._world_velocity(
            data,
            body_ids,
            data.subtree_com[..., self._view.root_body_id, :],
            positions=data.site_xpos[..., list(self._view.site_ids), :],
        )

    @property
    def body_external_wrench(self) -> torch.Tensor:
        data = self._require_data()
        return data.xfrc_applied[..., list(self._view.body_ids), :]

    @property
    def geom_pose_w(self) -> torch.Tensor:
        data = self._require_data()
        return torch.cat(
            (
                data.geom_xpos[..., list(self._view.geom_ids), :],
                self._quat_from_matrix(data.geom_xmat[..., list(self._view.geom_ids), :, :]),
            ),
            dim=-1,
        )

    @property
    def site_pose_w(self) -> torch.Tensor:
        data = self._require_data()
        return torch.cat(
            (
                data.site_xpos[..., list(self._view.site_ids), :],
                self._quat_from_matrix(data.site_xmat[..., list(self._view.site_ids), :, :]),
            ),
            dim=-1,
        )

    @property
    def joint_pos(self) -> torch.Tensor:
        data = self._require_data()
        return data.qpos.index_select(-1, self._view.non_free_qpos_indices)

    @property
    def joint_pos_biased(self) -> torch.Tensor:
        """Joint positions after the configured encoder bias."""

        return self.joint_pos + self.encoder_bias

    @property
    def joint_pos_limits(self) -> torch.Tensor:
        """Hard joint limits in the entity's non-free joint order."""

        model = self._env.bundle.native_model
        values = []
        for joint_id in self._view.non_free_joint_ids:
            joint_type = int(model.jnt_type[joint_id])
            width = _joint_qpos_width(joint_type)
            limits = torch.as_tensor(
                model.jnt_range[joint_id],
                dtype=self._env.bundle.dtype,
                device=self._env.bundle.device,
            )
            values.extend([limits] * width)
        if not values:
            return torch.empty(
                (*self.joint_pos.shape[:-1], 0, 2),
                dtype=self._env.bundle.dtype,
                device=self._env.bundle.device,
            )
        return torch.stack(values)

    @property
    def default_joint_pos_limits(self) -> torch.Tensor:
        return self.joint_pos_limits

    @property
    def soft_joint_pos_limits(self) -> torch.Tensor:
        limits = self.joint_pos_limits
        return limits * 0.9

    @property
    def joint_vel(self) -> torch.Tensor:
        data = self._require_data()
        return data.qvel.index_select(-1, self._view.non_free_qvel_indices)

    @property
    def joint_acc(self) -> torch.Tensor:
        data = self._require_data()
        return data.qacc.index_select(-1, self._view.non_free_qvel_indices)

    @property
    def tendon_len(self) -> torch.Tensor:
        data = self._require_data()
        return data.ten_length[..., list(self._view.tendon_ids)]

    @property
    def tendon_vel(self) -> torch.Tensor:
        data = self._require_data()
        return data.ten_velocity[..., list(self._view.tendon_ids)]

    @property
    def actuator_force(self) -> torch.Tensor:
        data = self._require_data()
        return data.actuator_force[..., list(self._view.actuator_ids)]

    @property
    def qfrc_actuator(self) -> torch.Tensor:
        data = self._require_data()
        return data.qfrc_actuator.index_select(-1, self._view.non_free_qvel_indices)

    def _joint_target(self, target_type: str) -> torch.Tensor:
        """Map the global actuator target buffer into stable joint order."""

        result = torch.zeros_like(self.default_joint_pos)
        target = self._env.action_manager.current_target
        if target is None:
            return result
        target = torch.as_tensor(target, dtype=result.dtype, device=result.device)
        target_types = self._env.action_manager.actuator_target_types
        if not target_types:
            target_types = ("position",) * int(target.shape[-1])
        joint_indices = {
            joint_id: index for index, joint_id in enumerate(self._view.non_free_joint_ids)
        }
        model = self._env.bundle.native_model
        for actuator_id in self._view.actuator_ids:
            if actuator_id >= len(target_types) or target_types[actuator_id] != target_type:
                continue
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            joint_index = joint_indices.get(joint_id)
            if joint_index is None or actuator_id >= target.shape[-1]:
                continue
            result[..., joint_index] = target[..., actuator_id]
        return result

    @property
    def joint_pos_target(self) -> torch.Tensor:
        return self._joint_target("position")

    @property
    def joint_vel_target(self) -> torch.Tensor:
        return self._joint_target("velocity")

    @property
    def joint_effort_target(self) -> torch.Tensor:
        return self._joint_target("effort")

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        data = self._require_data()
        gravity = torch.zeros(3, dtype=data.xmat.dtype, device=data.xmat.device)
        gravity[2] = -1.0
        return data.xmat[..., self._view.root_body_id, :, :].transpose(-1, -2) @ gravity

    @property
    def gravity_vec_w(self) -> torch.Tensor:
        gravity = torch.as_tensor(
            self._env.bundle.native_model.opt.gravity,
            dtype=self._env.bundle.dtype,
            device=self._env.bundle.device,
        )
        return (
            gravity.unsqueeze(0).expand(self._env.num_envs, -1).clone()
            if self._env.num_envs > 1
            else gravity
        )

    @property
    def forward_vec_b(self) -> torch.Tensor:
        result = torch.zeros(3, dtype=self._env.bundle.dtype, device=self._env.bundle.device)
        result[0] = 1.0
        return result

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        data = self._require_data()
        rotation = data.xmat[..., self._view.root_body_id, :, :].transpose(-1, -2)
        return (rotation @ self.root_link_lin_vel_w.unsqueeze(-1)).squeeze(-1)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        data = self._require_data()
        rotation = data.xmat[..., self._view.root_body_id, :, :].transpose(-1, -2)
        return (rotation @ self.root_link_ang_vel_w.unsqueeze(-1)).squeeze(-1)

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        data = self._require_data()
        rotation = data.xmat[..., self._view.root_body_id, :, :].transpose(-1, -2)
        return (rotation @ self.root_com_lin_vel_w.unsqueeze(-1)).squeeze(-1)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        data = self._require_data()
        rotation = data.xmat[..., self._view.root_body_id, :, :].transpose(-1, -2)
        return (rotation @ self.root_com_ang_vel_w.unsqueeze(-1)).squeeze(-1)

    # Expose component accessors in addition to packed pose/velocity.
    root_link_pos_w = property(lambda self: self.root_link_pose_w[..., :3])
    root_link_quat_w = property(lambda self: self.root_link_pose_w[..., 3:])
    root_link_lin_vel_w = property(lambda self: self.root_link_vel_w[..., :3])
    root_link_ang_vel_w = property(lambda self: self.root_link_vel_w[..., 3:])
    root_com_pos_w = property(lambda self: self.root_com_pose_w[..., :3])
    root_com_quat_w = property(lambda self: self.root_com_pose_w[..., 3:])
    root_com_lin_vel_w = property(lambda self: self.root_com_vel_w[..., :3])
    root_com_ang_vel_w = property(lambda self: self.root_com_vel_w[..., 3:])
    body_link_pos_w = property(lambda self: self.body_link_pose_w[..., :3])
    body_link_quat_w = property(lambda self: self.body_link_pose_w[..., 3:])
    body_link_lin_vel_w = property(lambda self: self.body_link_vel_w[..., :3])
    body_link_ang_vel_w = property(lambda self: self.body_link_vel_w[..., 3:])
    body_com_pos_w = property(lambda self: self.body_com_pose_w[..., :3])
    body_com_quat_w = property(lambda self: self.body_com_pose_w[..., 3:])
    body_com_lin_vel_w = property(lambda self: self.body_com_vel_w[..., :3])
    body_com_ang_vel_w = property(lambda self: self.body_com_vel_w[..., 3:])
    geom_pos_w = property(lambda self: self.geom_pose_w[..., :3])
    geom_quat_w = property(lambda self: self.geom_pose_w[..., 3:])
    site_pos_w = property(lambda self: self.site_pose_w[..., :3])
    site_quat_w = property(lambda self: self.site_pose_w[..., 3:])


class EntityRuntime:
    """Namespaced scene entity with task-term target-writing operations."""

    def __init__(self, env: ManagerBasedTaskEnv, view: Any) -> None:
        self._env = env
        self._view = view
        self.data = _EntityDataView(env, view)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._view, name)

    def _resolve_actuators(self, ids: torch.Tensor | None) -> torch.Tensor:
        if ids is None:
            return torch.tensor(
                self._view.actuator_ids,
                dtype=torch.long,
                device=self._env.bundle.device,
            )
        ids = torch.as_tensor(ids, dtype=torch.long, device=self._env.bundle.device).reshape(-1)
        actuator_ids = set(self._view.actuator_ids)
        local_joint_count = len(self._view.non_free_joint_ids)
        if set(ids.tolist()).issubset(range(local_joint_count)):
            mapping = {
                int(self._env.bundle.native_model.actuator_trnid[aid, 0]): aid
                for aid in self._view.actuator_ids
            }
            try:
                return torch.tensor(
                    [
                        mapping[self._view.non_free_joint_ids[int(joint_id)]]
                        for joint_id in ids.tolist()
                    ],
                    dtype=torch.long,
                    device=self._env.bundle.device,
                )
            except KeyError as exc:
                raise ValueError("Joint selection contains an unactuated joint") from exc
        if set(ids.tolist()).issubset(actuator_ids):
            return ids
        raise ValueError("Target ids do not belong to this scene entity")

    def set_joint_position_target(
        self, target: torch.Tensor, *, joint_ids: torch.Tensor | None = None
    ) -> None:
        self._env.action_manager.write_target(
            self._resolve_actuators(joint_ids), target, target_type="position"
        )

    def set_joint_velocity_target(
        self, target: torch.Tensor, *, joint_ids: torch.Tensor | None = None
    ) -> None:
        self._env.action_manager.write_target(
            self._resolve_actuators(joint_ids), target, target_type="velocity"
        )

    def set_joint_effort_target(
        self, target: torch.Tensor, *, joint_ids: torch.Tensor | None = None
    ) -> None:
        self._env.action_manager.write_target(
            self._resolve_actuators(joint_ids), target, target_type="effort"
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset only this entity through the shared origin-aware defaults."""

        if self._env.data is None:
            raise RuntimeError("Call reset() before resetting an individual entity")
        reset_state = self._env.reset_manager.build(self._env, env_ids, apply_terms=False)
        data = self._env.data
        qpos, qvel, ctrl = data.qpos.clone(), data.qvel.clone(), data.ctrl.clone()
        indices = self._view.qpos_indices
        velocity_indices = self._view.qvel_indices
        actuator_indices = torch.as_tensor(
            self._view.actuator_ids, dtype=torch.long, device=self._env.bundle.device
        )
        ids = self._env._ids(env_ids) if env_ids is not None else None

        def copy_rows(
            target: torch.Tensor, source: torch.Tensor, selected: torch.Tensor | None
        ) -> None:
            if self._env.num_envs == 1:
                target[indices if target is qpos else velocity_indices] = source[
                    indices if target is qpos else velocity_indices
                ]
            elif selected is None:
                target[:, indices if target is qpos else velocity_indices] = source[
                    :, indices if target is qpos else velocity_indices
                ]
            else:
                object_indices = indices if target is qpos else velocity_indices
                target[selected[:, None], object_indices] = source[
                    selected[:, None], object_indices
                ]

        reset_qpos = reset_state.qpos
        reset_qvel = reset_state.qvel
        if reset_qpos is None or reset_qvel is None:
            raise RuntimeError("Reset state must provide qpos and qvel")
        copy_rows(qpos, reset_qpos, ids)
        copy_rows(qvel, reset_qvel, ids)
        if actuator_indices.numel():
            if self._env.num_envs == 1:
                ctrl[actuator_indices] = reset_state.ctrl[actuator_indices]
            elif ids is None:
                ctrl[:, actuator_indices] = reset_state.ctrl[:, actuator_indices]
            else:
                ctrl[ids[:, None], actuator_indices] = reset_state.ctrl[
                    ids[:, None], actuator_indices
                ]
        self._env.data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        self._env.physics.forward(env_ids=env_ids)


class SceneRuntime:
    """Namespaced runtime view of entities, sensors, and terrain."""

    def __init__(self, env: ManagerBasedTaskEnv) -> None:
        self._env = env
        self.entities = {
            name: EntityRuntime(env, view) for name, view in env.bundle.entities.items()
        }
        self.sensors = env.sensor_manager
        self.terrain = env.terrain_manager

    def __getitem__(self, name: str) -> Any:
        if name in self.entities:
            return self.entities[name]
        if name in self.sensors.active_sensors:
            return self.sensors.get_sensor(name)
        if name == "terrain":
            return self.terrain
        available = [*self.entities, *self.sensors.active_sensors, "terrain"]
        raise KeyError(f"Scene element {name!r} not found. Available: {available}")

    def apply_environment_origins(self, qpos: torch.Tensor) -> torch.Tensor:
        """Translate every configured free entity exactly once."""

        origins = self.terrain.env_origins
        for view in self._env.bundle.entities.values():
            if not view.free_qpos_indices.numel():
                continue
            indices = view.free_qpos_indices[:3]
            if self._env.num_envs > 1:
                qpos[..., indices] += origins
            else:
                qpos[..., indices] += origins[0]
        return qpos

    def reset_to_default(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset all entity defaults through the canonical reset service.

        This API deliberately does not run task reset terms, command/event
        managers, or observation bookkeeping.  It is the scene-level default
        state operation; :class:`ManagerBasedTaskEnv` owns the complete
        episode lifecycle.  Both paths share ``ResetManager``'s origin-aware
        entity initialization, so direct scene resets cannot place one
        selected entity at world origin while normal resets use terrain
        origins.
        """

        if self._env.data is None and env_ids is not None:
            raise RuntimeError("A partial scene reset requires initialized physics data")
        reset_state = self._env.reset_manager.build(self._env, env_ids, apply_terms=False)
        self._env.physics.reset(
            qpos=reset_state.qpos,
            qvel=reset_state.qvel,
            ctrl=reset_state.ctrl,
            env_ids=env_ids,
            mocap_pos=reset_state.mocap_pos,
            mocap_quat=reset_state.mocap_quat,
            xfrc_applied=reset_state.xfrc_applied,
        )
        self._env.sensor_manager.reset(self._env, env_ids)
        self._env.physics.forward(env_ids=env_ids)
        self._env.sensor_manager.sense(self._env, env_ids)
        if self._env.state is not None:
            self._env.action_manager.reset(env_ids)
            self._env.sensor_state_manager.refresh_baseline(self._env, env_ids)


def _quat_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Create a normalized ZYX quaternion."""

    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack(
        (
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ),
        dim=-1,
    )


class ManagerBasedTaskEnv:
    """Manager-based task environment with the complete lifecycle owner.

    The environment owns configuration, manager execution, task state, and
    lifecycle ordering. :class:`PhysicsBackend` owns only model/data creation
    and low-level simulation mechanics. Task factories configure manager terms,
    while the environment executes the same lifecycle for every task.
    """

    def __init__(
        self,
        task_cfg: TaskEnvCfg,
        *,
        bundle: ModelBundle | None = None,
        command: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        domain_randomization: bool | None = None,
        render_mode: str | None = None,
        render_config: RenderConfig | None = None,
        num_envs: int | None = None,
        **load_options: Any,
    ) -> None:
        if not task_cfg.scene.entities:
            raise ValueError("Task scene must contain at least one named entity")
        primary_entity_name = self._select_primary_entity(task_cfg)
        primary_cfg = task_cfg.scene.entities[primary_entity_name]
        if task_cfg.scene.primary_entity is None:
            task_cfg.scene.primary_entity = primary_entity_name
        elif task_cfg.scene.primary_entity != primary_entity_name:
            raise ValueError(
                "TaskEnvCfg.primary_entity and SceneCfg.primary_entity must agree: "
                f"{primary_entity_name!r} != {task_cfg.scene.primary_entity!r}"
            )
        scene_build: SceneBuild = SceneBuilder().build(task_cfg.scene)
        if bundle is None:
            model_options = dict(load_options)
            requested_actuator_mode = model_options.pop(
                "actuator_mode", task_cfg.actions.actuator_mode
            )
            if requested_actuator_mode != task_cfg.actions.actuator_mode:
                raise ValueError(
                    "actuator_mode must be selected in TaskEnvCfg.actions, not overridden "
                    "by a model-loading option"
                )
            model_options.setdefault("timestep", task_cfg.physics_timestep)
            model_options.setdefault("decimation", task_cfg.decimation)
            model_options.setdefault("fixed_iterations", task_cfg.fixed_iterations)
            model_options.setdefault("solver_iterations", task_cfg.solver_iterations)
            model_options.setdefault("line_search_iterations", task_cfg.line_search_iterations)
            model_options.setdefault("nconmax", task_cfg.nconmax)
            model_options.setdefault("collision_policy", task_cfg.collision_policy)
            # This is an explicit task/backend capability setting.  Do not
            # silently infer it from terrain kind: disabling mesh-mesh pairs
            # changes collision semantics, while leaving them enabled can be
            # prohibitively expensive for a large procedural scene.
            model_options.setdefault(
                "disable_mesh_mesh_contacts", task_cfg.default_disable_mesh_mesh_contacts()
            )
            bundle = load_model_bundle(
                xml_path=scene_build.xml_path,
                entity_cfg=primary_cfg,
                entities=task_cfg.scene.entities,
                device=device,
                dtype=dtype,
                actuator_mode=requested_actuator_mode,
                **model_options,
            )
        elif (
            bundle.xml_path.resolve() != scene_build.xml_path.resolve()
            or (
                task_cfg.nconmax is not None
                and int(bundle.native_model.nconmax) != task_cfg.nconmax
            )
            or (
                task_cfg.default_disable_mesh_mesh_contacts()
                and "disable_mesh_mesh_contacts" in task_cfg.metadata
                and bundle.disable_mesh_mesh_contacts
                != task_cfg.default_disable_mesh_mesh_contacts()
            )
        ):
            # An injected bundle is a useful testing/diagnostic seam, but it
            # must never silently execute a different world than the task
            # configuration. A sensor-materialized wrapper can be mechanically
            # identical to the supplied entity XML, so first compare the
            # compiled structural signature. If it differs (terrain, prop,
            # or world content), rebuild from the authoritative SceneBuild.
            import mujoco

            scene_model = mujoco.MjModel.from_xml_path(str(scene_build.xml_path))
            # Declarative contact/IMU sensors may be materialized into a
            # wrapper even when the injected bundle already contains the same
            # physical scene. Sensors do not alter dynamics; the mechanical
            # signature is what detects an actually wrong terrain/prop/world.
            signature = ("nq", "nv", "nu", "nbody", "ngeom", "nsite", "ntendon")
            structurally_same = all(
                int(getattr(scene_model, name)) == int(getattr(bundle.native_model, name))
                for name in signature
            )
            if task_cfg.nconmax is not None:
                structurally_same = structurally_same and (
                    int(bundle.native_model.nconmax) == task_cfg.nconmax
                )
            if task_cfg.default_disable_mesh_mesh_contacts():
                structurally_same = structurally_same and (
                    bundle.disable_mesh_mesh_contacts
                    == task_cfg.default_disable_mesh_mesh_contacts()
                )
            if not structurally_same:
                bundle = load_model_bundle(
                    xml_path=scene_build.xml_path,
                    entity_cfg=primary_cfg,
                    entities=task_cfg.scene.entities,
                    device=bundle.device,
                    dtype=bundle.dtype,
                    timestep=bundle.timestep,
                    decimation=bundle.decimation,
                    fixed_iterations=bundle.fixed_iterations,
                    solver_iterations=bundle.solver_iterations,
                    line_search_iterations=bundle.line_search_iterations,
                    nconmax=task_cfg.nconmax,
                    disable_contacts=not bundle.contacts_enabled,
                    actuator_mode=bundle.actuator_mode,
                    bam_parameters=bundle.bam_parameters,
                    collision_policy=task_cfg.collision_policy,
                    disable_mesh_mesh_contacts=task_cfg.default_disable_mesh_mesh_contacts(),
                )
        configured_decimation = load_options.get("decimation", bundle.decimation)
        self.num_envs = int(task_cfg.scene.num_envs if num_envs is None else num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.num_envs > 1:
            self.physics = BatchedPhysicsBackend(
                bundle,
                num_envs=self.num_envs,
                actuator_mode=task_cfg.actions.actuator_mode,
                decimation=configured_decimation,
            )
        else:
            self.physics = PhysicsBackend(
                bundle,
                actuator_mode=task_cfg.actions.actuator_mode,
                decimation=configured_decimation,
            )
        if bundle.primary_entity_cfg.xml_path.resolve() != primary_cfg.xml_path.resolve():
            raise ValueError(
                "Task primary entity and model bundle refer to different XMLs: "
                f"{primary_cfg.xml_path} != {bundle.primary_entity_cfg.xml_path}"
            )
        missing_entities = set(task_cfg.scene.entities) - set(bundle.entities)
        if missing_entities:
            raise ValueError(
                f"Model bundle is missing configured scene entities: {sorted(missing_entities)!r}"
            )
        self.task_cfg = task_cfg
        self.cfg = task_cfg
        self.scene_build = scene_build
        self.bundle = self.physics.bundle
        self.terrain_manager = TerrainManager(
            task_cfg.scene.terrain,
            num_envs=self.num_envs,
            device=self.bundle.device,
            env_spacing=task_cfg.scene.env_spacing,
        )
        if self.num_envs > 1:
            self.terrain_manager.set_generators(self.physics.generators)
        else:
            self.terrain_manager.set_generator(self.physics._generator)
        self.decimation = self.physics.decimation
        self.actuator_mode = self.physics.actuator_mode
        self.config = task_cfg.task
        self.auto_reset = task_cfg.auto_reset
        self.domain_randomization = (
            task_cfg.metadata.get("domain_randomization", False)
            if domain_randomization is None
            else domain_randomization
        )
        self.sensor_manager = SensorManager(task_cfg.scene.sensors, self.bundle)
        # ``env.sensors`` is the task-facing first-class sensor namespace.
        self.sensors = self.sensor_manager
        self.physics.set_substep_callback(lambda: self.sensor_manager.update(self))
        self.scene = SceneRuntime(self)
        self.reset_manager = ResetManager(task_cfg.reset_state)
        self.model_mutation_manager = ModelMutationManager(task_cfg.model_mutations)
        self.sensor_state_manager = SensorStateManager(task_cfg.sensor_state)
        self.action_manager = ActionManager(task_cfg.actions)
        self.action_manager.prepare_terms(self)
        self.command_manager = CommandManager(task_cfg.commands, command=command)
        self.observation_manager = ObservationManager(task_cfg.observations)
        self.reward_manager = RewardManager(
            task_cfg.rewards,
            scale_by_dt=task_cfg.reward_scale_by_dt,
        )
        self.termination_manager = TerminationManager(task_cfg.terminations)
        self.event_manager = EventManager(task_cfg.events)
        # Startup events are construction-time mutations. Run them once after
        # every manager exists rather than deferring them until the first reset.
        self.curriculum_manager = CurriculumManager(task_cfg.curriculum)
        self.task_state_manager = TaskStateManager(task_cfg.task_state)
        self.state: EnvironmentState | None = None
        # Unlike episode-local step counts, this clock survives row resets and
        # is used by events authored with global wall-clock semantics.
        self._global_step_count = 0
        self.event_manager.startup(self)
        self._generator = self.physics._generator
        self._startup_events_applied = True
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode {render_mode!r}; use None or 'rgb_array'")
        self.render_mode = render_mode
        self.render_config = render_config or task_cfg.viewer
        self.metadata = {
            "render_modes": [None, "rgb_array"],
            "render_fps": 1.0 / (self.bundle.timestep * self.decimation),
            "num_envs": self.num_envs,
        }
        self._renderer: Any | None = None

    @staticmethod
    def _select_primary_entity(task_cfg: TaskEnvCfg) -> str:
        """Resolve the entity that owns the environment's root/action facade.

        Explicit task configuration wins.  The action graph is the next most
        useful source for generic tasks; ``robot`` remains a convenience for
        existing Microduck configs, not a required scene name.
        """

        entities = task_cfg.scene.entities
        if task_cfg.primary_entity is not None:
            return task_cfg.primary_entity
        if task_cfg.scene.primary_entity is not None:
            if task_cfg.scene.primary_entity not in entities:
                raise ValueError(
                    f"scene.primary_entity {task_cfg.scene.primary_entity!r} is not "
                    "in scene.entities"
                )
            return task_cfg.scene.primary_entity
        action_entities = tuple(
            dict.fromkeys(
                entity
                for term in task_cfg.actions.terms.values()
                if getattr(term, "enabled", True)
                and isinstance(entity := getattr(term, "entity", None), str)
                and entity in entities
            )
        )
        if len(action_entities) == 1:
            return action_entities[0]
        if "robot" in entities:
            return "robot"
        return next(iter(entities))

    @property
    def data(self) -> Any | None:
        return self.physics.data

    @data.setter
    def data(self, value: Any | None) -> None:
        self.physics.data = value

    @property
    def step_count(self) -> int:
        return self.physics.step_count

    @step_count.setter
    def step_count(self, value: int) -> None:
        self.physics.step_count = value

    @property
    def step_counts(self) -> torch.Tensor:
        if self.num_envs == 1:
            return torch.tensor([self.step_count], dtype=torch.long, device=self.bundle.device)
        return self.physics.step_counts

    @property
    def device(self) -> torch.device:
        """Device on which the environment's tensors are resident."""

        return self.bundle.device

    @property
    def physics_dt(self) -> float:
        """One backend integration timestep in seconds."""

        return float(self.bundle.timestep)

    @property
    def step_dt(self) -> float:
        """One policy-control timestep in seconds."""

        return self.physics_dt * self.decimation

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Per-environment elapsed control-step count."""

        return self.step_counts

    @property
    def max_episode_length(self) -> int:
        runtime = self.task_cfg.runtime
        if runtime is None:
            raise RuntimeError("Task runtime configuration has not been initialized")
        return int(runtime.episode_length_steps)

    @property
    def max_episode_length_s(self) -> float:
        return self.max_episode_length * self.step_dt

    @property
    def common_step_counter(self) -> int:
        """Global control-step counter, preserved across episode resets."""

        return self._global_step_count

    @property
    def unwrapped(self) -> ManagerBasedTaskEnv:
        return self

    @property
    def command(self) -> torch.Tensor:
        if self.command_manager.command is None:
            raise RuntimeError("Command manager has not been reset")
        return self.command_manager.command

    @command.setter
    def command(self, value: torch.Tensor) -> None:
        self.command_manager.set_command(value)

    @property
    def transition(self) -> TransitionData | None:
        """Expose the current transition without duplicating ownership."""

        return None if self.state is None else self.state.transition

    def _get_renderer(self) -> Any:
        if self._renderer is not None:
            return self._renderer
        if self.render_mode != "rgb_array":
            raise RuntimeError("Renderer requested while render_mode is disabled")
        if self.render_config.backend == "mujoco":
            from ..rendering.native import NativeRenderer

            self._renderer = NativeRenderer(
                self.bundle,
                width=self.render_config.width,
                height=self.render_config.height,
                camera=self.render_config.camera,
            )
        else:
            from ..rendering.torch_renderer import TorchRenderer

            self._renderer = TorchRenderer(
                self.bundle,
                width=self.render_config.width,
                height=self.render_config.height,
                camera=self.render_config.camera,
                ray_chunk_size=self.render_config.ray_chunk_size,
            )
        return self._renderer

    def render(self) -> Any | None:
        """Render the current state as an RGB uint8 array when enabled."""

        if self.render_mode is None:
            return None
        if self.data is None:
            raise RuntimeError("Call reset() before rendering")
        return self._get_renderer().render(self)

    def close(self) -> None:
        """Release renderer resources owned by the environment."""

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def entity(self, name: str) -> Any:
        """Return a namespaced scene entity for task terms."""

        try:
            return self.scene.entities[name]
        except KeyError as exc:
            raise KeyError(f"Scene entity {name!r} is not configured") from exc

    @property
    def _bam_vin(self) -> torch.Tensor | None:
        return self.physics._bam_vin

    @_bam_vin.setter
    def _bam_vin(self, value: torch.Tensor | None) -> None:
        self.physics._bam_vin = value

    @property
    def _bam_drop_gain(self) -> torch.Tensor | float | None:
        return self.physics._bam_drop_gain

    @_bam_drop_gain.setter
    def _bam_drop_gain(self, value: torch.Tensor | float | None) -> None:
        self.physics._bam_drop_gain = value

    @property
    def _bam_friction_scale(self) -> torch.Tensor | float:
        return self.physics._bam_friction_scale

    @_bam_friction_scale.setter
    def _bam_friction_scale(self, value: torch.Tensor | float) -> None:
        self.physics._bam_friction_scale = value

    def _random(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.physics.random(dtype=dtype)

    def _random_tensor(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        normal: bool = False,
    ) -> torch.Tensor:
        """Draw independent random values through the backend-owned streams."""

        return self.physics.random_tensor(shape, dtype=dtype, normal=normal)

    def _sample_range(
        self,
        low: float,
        high: float,
        env_ids: torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        return self.physics.sample_range(low, high, env_ids=env_ids)

    def _sample_delay(
        self,
        low: int,
        high: int,
        env_ids: torch.Tensor | slice | None = None,
        *,
        per_env: bool = True,
    ) -> int | torch.Tensor:
        return self.physics.sample_delay(low, high, env_ids=env_ids, per_env=per_env)

    def _sync_task_state(self) -> None:
        """Publish task-component data through the environment state facade."""

        if self.state is not None:
            self.state.task_data.update(self.task_state_manager.data)
            self.state.manager_data["task_state"] = self.task_state_manager.data

    def _ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.bundle.device)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.bundle.device)[env_ids]
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.bundle.device).reshape(-1)
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.num_envs):
            raise ValueError("env_ids contains an out-of-range environment index")
        return ids

    def _joint_measurements(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return resolved output position and motor velocity in actuator order."""

        return self.physics.actuator_measurements()

    def _encoder_velocity(self) -> torch.Tensor:
        """Return the output-side velocity seen by the encoder."""

        return self.physics.encoder_velocity()

    def _reset_all(
        self,
        *,
        command: torch.Tensor | None,
        seed: int | None,
        randomize: bool | None,
        finalize: bool = True,
    ) -> torch.Tensor | dict[str, torch.Tensor] | None:
        """Run a generic full-reset lifecycle assembled by managers."""

        reset_ids = self._ids(None)
        self.physics.set_seed(seed)
        self._generator = self.physics._generator
        if seed is not None:
            self.terrain_manager.set_seed(seed)
        if randomize is not None:
            self.domain_randomization = randomize
        if command is not None:
            self.command_manager.set_command(command)
        # Curriculum callbacks run before terrain/model/reset-state assembly so
        # difficulty-dependent state is visible to the episode being created.
        curriculum = self.curriculum_manager.compute(self, reset_ids)
        self.terrain_manager.reset(reset_ids)
        self.model_mutation_manager.reset(self, None)
        reset_state = self.reset_manager.build(self, None)
        self.physics.reset(
            qpos=reset_state.qpos,
            qvel=reset_state.qvel,
            ctrl=reset_state.ctrl,
            mocap_pos=reset_state.mocap_pos,
            mocap_quat=reset_state.mocap_quat,
            xfrc_applied=reset_state.xfrc_applied,
        )
        self.sensor_manager.reset(self)
        self.physics.forward()
        self.sensor_manager.sense(self)
        self.state = EnvironmentState(
            sensors=self.sensor_state_manager.initialize(self),
            reward_terms={},
            task_data={},
            pending_reset=(
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.bundle.device)
                if self.num_envs > 1
                else False
            ),
        )
        self.state.manager_data["curriculum"] = curriculum
        self.event_manager.apply_reset(self)
        # Reset events operate on the freshly reset state, then the backend is
        # forwarded again before any baseline or returned observation is read.
        # This prevents a reset callback that edits qpos/qvel/model state from
        # leaving derived kinematics stale.
        self.physics.forward()
        self.sensor_manager.sense(self)
        self.sensor_state_manager.refresh_baseline(self, sample_calibration=False)
        self.task_state_manager.reset(self)
        self._sync_task_state()
        self.observation_manager.reset(self)
        self.action_manager.reset()
        self.reward_manager.reset(self)
        self.curriculum_manager.reset(self)
        self.command_manager.reset(self)
        self.event_manager.reset(self, apply_terms=False)
        self.termination_manager.reset(self)
        if not finalize:
            return None
        # Stateful commands (phase/posture generators) publish their reset-time
        # value through the same zero-dt compute used by the command manager.
        self.command_manager.compute(self, dt=0.0)
        return self._update_observation_groups()

    def _reset_selected_generic(
        self,
        env_ids: torch.Tensor | slice,
        *,
        command: torch.Tensor | None,
        seed: int | None,
        randomize: bool | None,
        finalize: bool = True,
    ) -> torch.Tensor | dict[str, torch.Tensor] | None:
        """Reset only selected vector rows through the same generic managers."""

        ids = self._ids(env_ids)
        if not ids.numel():
            return self.observation() if finalize else None
        if self.num_envs == 1:
            return self._reset_all(
                command=command, seed=seed, randomize=randomize, finalize=finalize
            )
        self.physics.set_seed(seed, env_ids=ids)
        self._generator = self.physics._generator
        if seed is not None:
            self.terrain_manager.set_seed(seed, env_ids=ids)
        if randomize is not None:
            self.domain_randomization = randomize
        if command is not None:
            self.command_manager.set_command(torch.as_tensor(command), ids)
        curriculum = self.curriculum_manager.compute(self, ids)
        self.terrain_manager.reset(ids)
        self.model_mutation_manager.reset(self, ids)
        reset_state = self.reset_manager.build(self, ids)
        self.physics.reset(
            qpos=reset_state.qpos,
            qvel=reset_state.qvel,
            ctrl=reset_state.ctrl,
            env_ids=ids,
            mocap_pos=reset_state.mocap_pos,
            mocap_quat=reset_state.mocap_quat,
            xfrc_applied=reset_state.xfrc_applied,
        )
        self.sensor_manager.reset(self, ids)
        if self.state is None:
            raise RuntimeError("Partial reset requires initialized environment state")
        if isinstance(self.state.pending_reset, torch.Tensor):
            self.state.pending_reset[ids] = False
        else:
            self.state.pending_reset = False
        self.state.manager_data["curriculum"] = curriculum
        self.event_manager.apply_reset(self, ids)
        self.physics.forward(env_ids=ids)
        self.sensor_manager.sense(self, ids)
        self.sensor_state_manager.refresh_baseline(self, ids)
        self.task_state_manager.reset(self, ids)
        self._sync_task_state()
        self.observation_manager.reset(self, ids)
        self.action_manager.reset(ids)
        self.reward_manager.reset(self, ids)
        self.curriculum_manager.reset(self, ids)
        self.command_manager.reset(self, ids)
        self.event_manager.reset(self, ids, apply_terms=False)
        self.termination_manager.reset(self, ids)
        if not finalize:
            return None
        self.command_manager.compute(self, dt=0.0, env_ids=ids)
        return self._update_observation_groups()

    def reset(
        self,
        command: torch.Tensor | None = None,
        *,
        env_ids: torch.Tensor | slice | None = None,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if env_ids is not None:
            if self.state is None:
                raise RuntimeError("Partial reset requires an initialized environment")
            result = self._reset_selected_generic(
                env_ids, command=command, seed=seed, randomize=randomize
            )
            if result is None:
                raise RuntimeError("A finalized reset must return observations")
            return result
        result = self._reset_all(command=command, seed=seed, randomize=randomize)
        if result is None:
            raise RuntimeError("A finalized reset must return observations")
        return result

    def _reset_selected(
        self,
        env_ids: torch.Tensor | slice,
        *,
        command: torch.Tensor | None,
        seed: int | None,
        randomize: bool | None,
        finalize: bool = True,
    ) -> torch.Tensor | dict[str, torch.Tensor] | None:
        """Reset selected rows through the generic reset manager lifecycle."""

        return self._reset_selected_generic(
            env_ids,
            command=command,
            seed=seed,
            randomize=randomize,
            finalize=finalize,
        )

    def _next_interval_step(self, interval: tuple[float, float]) -> int:
        seconds = float(self._sample_range(*interval).reshape(-1)[0])
        steps = torch.ceil(torch.as_tensor(seconds / (self.bundle.timestep * self.decimation)))
        return max(1, int(steps.item()))

    @staticmethod
    def _lag_value(value: int | torch.Tensor) -> int:
        """Return a safe history capacity for scalar or per-env lag samples."""

        if isinstance(value, torch.Tensor):
            return int(value.max().item()) if value.numel() else 0
        return int(value)

    def _observation_noise(self, shape: torch.Size, scale: float) -> torch.Tensor:
        """Generate one term's configured uniform observation noise."""

        if not self.domain_randomization or scale == 0.0:
            return torch.zeros(shape, dtype=self.bundle.dtype, device=self.bundle.device)
        return (self._random_tensor(shape, dtype=self.bundle.dtype) * 2.0 - 1.0) * scale

    def observation(
        self, group: str = "actor", *, update_history: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute one named observation group."""

        return self.observation_manager.compute(self, group, update_history=update_history)

    def _update_observation_groups(self) -> torch.Tensor | dict[str, torch.Tensor]:
        """Advance each configured observation group's temporal state once."""

        observation = self.observation(update_history=True)
        for name, group in self.task_cfg.observations.groups.items():
            if group.enabled and name != "actor":
                self.observation(name, update_history=True)
        return observation

    def observations(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute every enabled configured observation group."""

        return {
            name: self.observation(name)
            for name, group in self.task_cfg.observations.groups.items()
            if group.enabled
        }

    def step(self, action: torch.Tensor) -> EnvStep:
        if self.data is None or self.state is None:
            self.reset()
        if self.data is None or self.state is None:
            raise RuntimeError("Call reset() before step()")
        pending = self.state.pending_reset
        if (isinstance(pending, torch.Tensor) and bool(pending.any().item())) or (
            isinstance(pending, bool) and pending
        ):
            raise RuntimeError(
                "One or more environment rows are done; call reset(env_ids=...) before stepping"
            )
        action = torch.as_tensor(action, dtype=self.bundle.dtype, device=self.bundle.device)
        expected_action_shape = (
            (self.task_cfg.action_size,)
            if self.num_envs == 1
            else (self.num_envs, self.task_cfg.action_size)
        )
        if action.shape != expected_action_shape:
            raise ValueError(
                f"Expected action shape {expected_action_shape}, got {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all():
            raise ValueError("Action contains non-finite values")
        control_dt = self.bundle.timestep * self.decimation
        sensor_context = self.sensor_state_manager.begin_step(self)
        self.task_state_manager.pre_physics(self, control_dt)
        self._sync_task_state()
        self.event_manager.apply(self, "pre_physics")
        applied_action = self.action_manager.process_action(action)
        target = self.action_manager.apply_action()
        direct_ctrl = self.action_manager.current_ctrl
        kwargs: dict[str, Any] = {"target_type": self.action_manager.target_type}
        if direct_ctrl is not None:
            kwargs["direct_ctrl"], kwargs["direct_ctrl_mask"] = direct_ctrl
        self.physics.step(target, **kwargs)
        self._global_step_count += 1
        self.task_state_manager.post_physics(self, control_dt)
        self._sync_task_state()
        # Post-physics events observe the freshly integrated state.  Reward and
        # termination terms use the command that produced this transition;
        # command resampling happens afterward.
        self.event_manager.apply(self, "post_physics")
        transition = self.sensor_state_manager.end_step(self, action, sensor_context)
        self.state.transition = transition
        # Task state derived from this transition is finalized before reward
        # and termination evaluation.  Managers therefore see one coherent
        # post-physics snapshot rather than a mixture of current and stale
        # persistent state.
        self.task_state_manager.compute(self, control_dt)
        self._sync_task_state()
        # Terminations describe the state produced by physics.  Evaluate them
        # before reward terms, matching the manager lifecycle and
        # avoiding a reward function accidentally changing whether a state is
        # considered terminal.
        finite = torch.isfinite(self.data.qpos).all(dim=-1) & torch.isfinite(self.data.qvel).all(
            dim=-1
        )
        if self.num_envs == 1:
            finite = bool(finite.item())
        terminated, truncated, termination_values = self.termination_manager.evaluate(
            self, finite=finite
        )
        bad_orientation_value = termination_values.get("bad_orientation", False)
        reward, terms = self.reward_manager.compute(self)
        self.state.reward_terms = terms
        reward_finite = torch.isfinite(reward)
        if self.num_envs == 1:
            if not bool(reward_finite.item()):
                finite = False
                terminated = True
                termination_values["non_finite"] = True
        else:
            finite = finite & reward_finite
            non_finite_reward = ~reward_finite
            terminated = terminated | non_finite_reward
            if "non_finite" in termination_values:
                termination_values["non_finite"] = (
                    torch.as_tensor(termination_values["non_finite"], device=finite.device)
                    | non_finite_reward
                )
        done = torch.as_tensor(
            terminated, dtype=torch.bool, device=self.bundle.device
        ) | torch.as_tensor(truncated, dtype=torch.bool, device=self.bundle.device)
        transition_step = (
            self.step_count if self.num_envs == 1 else self.step_counts.detach().clone()
        )
        time_value = self.data.time.detach().clone()
        if isinstance(time_value, torch.Tensor) and time_value.numel() == 1:
            time_value = float(time_value.item())

        # Start completed rows immediately after reward/termination evaluation.
        # Capture the terminal observation while the transition
        # state is still live, then reset only those rows before running the
        # post-reset command/event/observation stages below.
        terminal_observation: torch.Tensor | dict[str, torch.Tensor] | None = None
        if self.auto_reset and bool(done.any().item()):
            self.physics.forward()
            self.sensor_manager.sense(self)
            terminal_observation = self.observation(update_history=True)
            if isinstance(terminal_observation, dict):
                finite_terms = [
                    torch.isfinite(value).all(dim=-1)
                    if self.num_envs > 1 and value.ndim > 1
                    else torch.isfinite(value).all()
                    for value in terminal_observation.values()
                ]
                terminal_finite = torch.stack(finite_terms).all(dim=0)
            else:
                terminal_finite = torch.isfinite(terminal_observation).all(dim=-1)
            if self.num_envs == 1:
                if not bool(terminal_finite.item()):
                    finite = False
                    terminated = True
                    termination_values["non_finite"] = True
            else:
                finite = finite & terminal_finite
                non_finite = ~terminal_finite
                terminated = terminated | non_finite
                termination_values["non_finite"] = ~finite
            done = torch.as_tensor(
                terminated, dtype=torch.bool, device=self.bundle.device
            ) | torch.as_tensor(truncated, dtype=torch.bool, device=self.bundle.device)
            if self.num_envs == 1:
                if bool(done.item()):
                    self._reset_all(command=None, seed=None, randomize=None, finalize=False)
            else:
                done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)
                if done_ids.numel():
                    self._reset_selected(
                        done_ids,
                        command=None,
                        seed=None,
                        randomize=None,
                        finalize=False,
                    )

        # MuJoCo's integrator forwards before, rather than after, integration.
        # This is the final transition/reset forward before commands, step
        # events, and observations consume derived kinematics.
        self.physics.forward()
        self.command_manager.step(self)
        self.event_manager.apply(self, "step")
        self.event_manager.apply(self, "interval")
        self.sensor_manager.sense(self)
        observation = self._update_observation_groups()
        if isinstance(observation, dict):
            finite_terms = [
                torch.isfinite(value).all(dim=-1)
                if self.num_envs > 1 and value.ndim > 1
                else torch.isfinite(value).all()
                for value in observation.values()
            ]
            observation_finite = torch.stack(finite_terms).all(dim=0)
        else:
            observation_finite = torch.isfinite(observation).all(dim=-1)
        if not self.auto_reset or terminal_observation is None:
            if self.num_envs == 1:
                if not bool(observation_finite.item()):
                    finite = False
                    terminated = True
                    termination_values["non_finite"] = True
            else:
                finite = finite & observation_finite
                non_finite = ~finite
                terminated = terminated | non_finite
                termination_values["non_finite"] = non_finite
            done = torch.as_tensor(
                terminated, dtype=torch.bool, device=self.bundle.device
            ) | torch.as_tensor(truncated, dtype=torch.bool, device=self.bundle.device)
        info: dict[str, Any] = {
            "step": transition_step,
            "time": time_value,
            "finite": finite,
            "bad_orientation": bad_orientation_value,
            "terminations": termination_values,
            "reward_terms": {
                name: (float(value.item()) if value.numel() == 1 else value.detach().clone())
                for name, value in terms.items()
            },
            "curriculum": self.curriculum_manager.last_extras,
            "applied_action": applied_action.detach().clone(),
        }
        # Preserve the terminal observation for learning code even when the
        # manager-owned lifecycle immediately starts the next episode.
        info["terminal_observation"] = (
            {name: value.detach().clone() for name, value in terminal_observation.items()}
            if isinstance(terminal_observation, dict)
            else (
                terminal_observation.detach().clone()
                if terminal_observation is not None
                else (
                    {name: value.detach().clone() for name, value in observation.items()}
                    if isinstance(observation, dict)
                    else observation.detach().clone()
                )
            )
        )
        if not self.auto_reset and self.num_envs == 1:
            if bool(done.item()):
                self.state.pending_reset = True
        elif not self.auto_reset:
            if not isinstance(self.state.pending_reset, torch.Tensor):
                raise RuntimeError("Batched environment pending-reset state must be a tensor")
            self.state.pending_reset[done] = True
        return EnvStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def snapshot(self) -> dict[str, Any]:
        if self.data is None:
            raise RuntimeError("Call reset() before snapshot()")
        return {
            "qpos": self.data.qpos.detach().clone(),
            "qvel": self.data.qvel.detach().clone(),
            "qacc": self.data.qacc.detach().clone(),
            "ctrl": self.data.ctrl.detach().clone(),
            "sensordata": self.data.sensordata.detach().clone(),
            "time": (
                float(self.data.time.item())
                if self.data.time.numel() == 1
                else self.data.time.detach().clone()
            ),
        }


__all__ = ["EnvStep", "EnvironmentState", "ManagerBasedTaskEnv", "SensorState", "TransitionData"]
