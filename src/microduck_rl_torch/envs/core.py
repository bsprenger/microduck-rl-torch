"""Manager-based task environment and lifecycle implementation."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import torch

from ..rendering.config import RenderConfig
from .config import sample_uniform
from .managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    EventManager,
    ObservationManager,
    RewardManager,
    TaskStateManager,
    TerminationManager,
)
from .model import (
    ModelBundle,
    _joint_qpos_width,
    _joint_qvel_width,
    load_model_bundle,
)
from .physics import BatchedPhysicsBackend, PhysicsBackend
from .rewards import foot_contact_mask
from .scene import SceneBuild, SceneBuilder, TerrainManager
from .sensors import SensorManager
from .task_config import TaskEnvCfg


@dataclass(frozen=True)
class EnvStep:
    observation: torch.Tensor
    reward: torch.Tensor
    terminated: bool | torch.Tensor
    truncated: bool | torch.Tensor
    info: dict[str, Any]


@dataclass
class SensorState:
    """Environment-owned sensor and actuator histories.

    This state is deliberately separate from task state. A future task can
    retain the same sensor buffers while adding posture, prop, or phase data
    under :class:`EnvironmentState.task_data`.
    """

    last_action: torch.Tensor
    previous_action: torch.Tensor
    previous_joint_velocity: torch.Tensor
    previous_foot_positions: torch.Tensor | None
    foot_air_time: torch.Tensor | None
    foot_contact: torch.Tensor | None
    imu_ang_vel_history: list[torch.Tensor]
    projected_gravity_history: list[torch.Tensor]
    delay_buffer: list[torch.Tensor]
    delay_lag: int | torch.Tensor
    imu_lag: int | torch.Tensor
    encoder_bias: torch.Tensor
    imu_quaternion: torch.Tensor


@dataclass(frozen=True)
class TransitionData:
    """Data produced by one transition and visible to manager terms."""

    action: torch.Tensor
    previous_action: torch.Tensor
    previous_foot_positions: torch.Tensor | None
    foot_air_time: torch.Tensor | None
    foot_contact: torch.Tensor | None
    foot_touchdown: torch.Tensor | None


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
    """Small upstream-shaped data facade for custom entity/action terms."""

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
        # The MicroDuck encoder-bias vector is in actuator order.  Entity data
        # follows upstream and exposes the bias in articulation-joint order;
        # map only the entity's actuated joints and use zero for passive joints.
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
        force: torch.Tensor | None,
        torque: torch.Tensor | None,
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

        if force is not None:
            values, ids = rows(force, "External force")
            assign(values, ids, slice(0, 3))
        if torque is not None:
            values, ids = rows(torque, "External torque")
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

    @property
    def joint_pos_target(self) -> torch.Tensor:
        target = self._env.action_manager.current_target
        if target is None:
            return self.default_joint_pos.clone()
        return target.index_select(
            -1, torch.as_tensor(self._view.actuator_ids, device=target.device)
        )

    @property
    def joint_vel_target(self) -> torch.Tensor:
        target = self._env.action_manager.current_target
        size = self.default_joint_pos.shape[-1]
        if target is None or self._env.action_manager.target_type not in {"velocity", "effort"}:
            return torch.zeros(
                (*self.default_joint_pos.shape[:-1], size),
                dtype=self._env.bundle.dtype,
                device=self._env.bundle.device,
            )
        return target.index_select(
            -1, torch.as_tensor(self._view.actuator_ids, device=target.device)
        )

    @property
    def joint_effort_target(self) -> torch.Tensor:
        target = self._env.action_manager.current_target
        size = self.default_joint_pos.shape[-1]
        if target is None or self._env.action_manager.target_type != "effort":
            return torch.zeros(
                (*self.default_joint_pos.shape[:-1], size),
                dtype=self._env.bundle.dtype,
                device=self._env.bundle.device,
            )
        return target.index_select(
            -1, torch.as_tensor(self._view.actuator_ids, device=target.device)
        )

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

    # Upstream exposes component accessors in addition to packed pose/velocity.
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
        joint_ids = set(self._view.joint_ids)
        if set(ids.tolist()).issubset(actuator_ids):
            return ids
        if set(ids.tolist()).issubset(joint_ids):
            mapping = {
                int(self._env.bundle.native_model.actuator_trnid[aid, 0]): aid
                for aid in self._view.actuator_ids
            }
            try:
                return torch.tensor(
                    [mapping[int(joint_id)] for joint_id in ids.tolist()],
                    dtype=torch.long,
                    device=self._env.bundle.device,
                )
            except KeyError as exc:
                raise ValueError("Joint selection contains an unactuated joint") from exc
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


class SceneRuntime:
    """Unified upstream-shaped scene namespace for entities and sensors."""

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
    and low-level simulation mechanics. This mirrors upstream's
    ``ManagerBasedRlEnv``: task factories configure manager terms, while the
    environment executes the same lifecycle for every task.
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
        # ``robot`` is the current MicroDuck convenience name, not a physics
        # requirement.  Generic tasks may make a prop, arm, or another entity
        # the first/primary scene object and configure their own managers.
        primary_entity_name = (
            "robot" if "robot" in task_cfg.scene.entities else next(iter(task_cfg.scene.entities))
        )
        robot_cfg = task_cfg.scene.entities[primary_entity_name]
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
            model_options.setdefault("collision_policy", task_cfg.collision_policy)
            bundle = load_model_bundle(
                xml_path=scene_build.xml_path,
                entity_cfg=robot_cfg,
                entities=task_cfg.scene.entities,
                device=device,
                dtype=dtype,
                actuator_mode=requested_actuator_mode,
                **model_options,
            )
        elif bundle.xml_path.resolve() != scene_build.xml_path.resolve():
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
            if not structurally_same:
                bundle = load_model_bundle(
                    xml_path=scene_build.xml_path,
                    entity_cfg=robot_cfg,
                    entities=task_cfg.scene.entities,
                    device=bundle.device,
                    dtype=bundle.dtype,
                    timestep=bundle.timestep,
                    decimation=bundle.decimation,
                    fixed_iterations=bundle.fixed_iterations,
                    solver_iterations=bundle.solver_iterations,
                    line_search_iterations=bundle.line_search_iterations,
                    disable_contacts=not bundle.contacts_enabled,
                    actuator_mode=bundle.actuator_mode,
                    bam_parameters=bundle.bam_parameters,
                )
        configured_decimation = load_options.get("decimation", bundle.decimation)
        self.num_envs = int(task_cfg.scene.num_envs if num_envs is None else num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        physics_type = PhysicsBackend if self.num_envs == 1 else BatchedPhysicsBackend
        self.physics = (
            physics_type(
                bundle,
                num_envs=self.num_envs,
                actuator_mode=task_cfg.actions.actuator_mode,
                decimation=configured_decimation,
                actuator_delay_lag=task_cfg.actions.actuator_delay_lag,
            )
            if self.num_envs > 1
            else physics_type(
                bundle,
                actuator_mode=task_cfg.actions.actuator_mode,
                decimation=configured_decimation,
                actuator_delay_lag=task_cfg.actions.actuator_delay_lag,
            )
        )
        if bundle.entity_cfg.xml_path.resolve() != robot_cfg.xml_path.resolve():
            raise ValueError(
                "Task entity and model bundle refer to different robot XMLs: "
                f"{robot_cfg.xml_path} != {bundle.entity_cfg.xml_path}"
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
        self.action_scale = task_cfg.actions.scale
        self.decimation = self.physics.decimation
        self.actuator_mode = self.physics.actuator_mode
        self.config = task_cfg.task
        self.auto_reset = task_cfg.auto_reset
        self._velocity_state_enabled = bool(task_cfg.metadata.get("velocity_state", False))
        self.sensor_manager = SensorManager(task_cfg.scene.sensors, self.bundle)
        # ``env.sensors`` is the task-facing first-class sensor namespace.
        self.sensors = self.sensor_manager
        self.physics.set_substep_callback(lambda: self.sensor_manager.update(self))
        self.scene = SceneRuntime(self)
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
        self.curriculum_manager = CurriculumManager(task_cfg.curriculum)
        self.task_state_manager = TaskStateManager(task_cfg.task_state)
        action_delay_lag = task_cfg.actions.delay_lag
        if isinstance(action_delay_lag, tuple):
            low, high = action_delay_lag
            if low < 0 or low > high:
                raise ValueError("action delay range must satisfy 0 <= low <= high")
            self.action_delay_range = (low, high)
        else:
            if action_delay_lag < 0:
                raise ValueError("action_delay_lag must be non-negative")
            self.action_delay_range = (action_delay_lag, action_delay_lag)
        self.domain_randomization = (
            task_cfg.metadata.get("domain_randomization", False)
            if domain_randomization is None
            else domain_randomization
        )
        self.state: EnvironmentState | None = None
        self._generator = self.physics._generator
        # Generic managers own their caches/state. This field remains only as
        # a compatibility scratch slot for the existing velocity terms; new
        # tasks must use manager-owned state or ``state.task_data``.
        self._velocity_reward_cache: dict[str, torch.Tensor] | None = None
        self._startup_events_applied = False
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
    ) -> int | torch.Tensor:
        return self.physics.sample_delay(low, high, env_ids=env_ids)

    def _apply_environment_origins(self, qpos: torch.Tensor) -> torch.Tensor:
        """Translate every movable configured entity to its selected terrain patch."""

        origins = self.terrain_manager.env_origins
        for _entity_name, view in self.bundle.entities.items():
            if not view.free_qpos_indices.numel():
                continue
            # All free entities are task-scene objects, not world geometry, and
            # therefore share the environment origin.  This is the same
            # scene-wide translation applied by upstream's env_origins.
            indices = view.free_qpos_indices[:3]
            if self.num_envs > 1:
                qpos[..., indices] += origins
            else:
                qpos[..., indices] += origins[0]
        return qpos

    def _root_height_index(self) -> int | None:
        """Return the primary scene entity's free-joint height coordinate."""

        primary = self.bundle.entities.get("robot") or next(
            (view for view in self.bundle.entities.values() if view.free_qpos_indices.numel()),
            None,
        )
        if primary is None or primary.free_qpos_indices.numel() != 7:
            return None
        return int(primary.free_qpos_indices[2])

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

    def _initial_observation(self) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.data is None:
            raise RuntimeError("Call reset() before initializing observation state")
        imu_slice = self.bundle.sensor_slices.get("imu_ang_vel")
        if imu_slice is None:
            imu_slice = next(
                (
                    sensor_slice
                    for name, sensor_slice in self.bundle.sensor_slices.items()
                    if name.endswith("/imu_ang_vel")
                ),
                None,
            )
        base_ang_vel = (
            self.data.sensordata[..., imu_slice].clone() if imu_slice is not None else None
        )
        gravity_world = torch.zeros(3, dtype=self.bundle.dtype, device=self.bundle.device)
        gravity_world[2] = -1.0
        primary = self.bundle.entities.get("robot") or next(iter(self.bundle.entities.values()))
        gravity = self.data.xmat[..., primary.root_body_id, :, :].transpose(-1, -2) @ gravity_world
        return base_ang_vel, gravity

    def _joint_measurements(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return resolved output position and motor velocity in actuator order."""

        return self.physics.actuator_measurements()

    def _encoder_velocity(self) -> torch.Tensor:
        """Return the output-side velocity seen by the encoder."""

        return self.physics.encoder_velocity()

    def _restore_model_defaults(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Restore only the model instances participating in this reset."""

        self.physics.restore_model_defaults(env_ids)

    def _apply_domain_randomization(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Apply independent model mutations to the selected env instances."""

        # The built-in MicroDuck randomization recipe is a task component. A
        # generic task with a different model must opt into its own mutation
        # terms/events instead of inheriting trunk/head/foot assumptions.
        if not self._velocity_state_enabled:
            return

        if self.num_envs == 1:
            selected = (self.physics, self.bundle, self._generator)
            instances = (selected,)
        else:
            ids = self._ids(env_ids)
            instances = tuple(
                (
                    self.physics.instances[index],
                    self.physics.instances[index].bundle,
                    self.physics.instances[index]._generator,
                )
                for index in ids.tolist()
            )
        randomization = getattr(self.config, "randomization", None)
        for backend, bundle, generator in instances:
            if not self.domain_randomization or randomization is None:
                backend.configure_bam(
                    vin=None,
                    drop_gain=None,
                    friction_scale=torch.ones((), dtype=bundle.dtype, device=bundle.device),
                )
                continue
            base_dof_armature = backend.base_field("dof_armature")
            base_body_mass = backend.base_field("body_mass")
            base_body_inertia = backend.base_field("body_inertia")
            base_geom_friction = backend.base_field("geom_friction")
            base_native_dof_armature = backend.base_field("native_dof_armature")
            base_native_body_mass = backend.base_field("native_body_mass")
            base_native_body_inertia = backend.base_field("native_body_inertia")
            base_native_geom_friction = backend.base_field("native_geom_friction")
            trunk = bundle.root_body_id
            if getattr(self.config, "randomize_mass_inertia", False):
                mass_scale = backend.sample_range(*randomization.mass_inertia_range)
                bundle.torch_model.body_mass[trunk] = base_body_mass[trunk] * mass_scale
                bundle.torch_model.body_inertia[trunk] = base_body_inertia[trunk] * mass_scale
                bundle.native_model.body_mass[trunk] = base_native_body_mass[trunk] * float(
                    mass_scale
                )
                bundle.native_model.body_inertia[trunk] = base_native_body_inertia[trunk] * float(
                    mass_scale
                )
            if getattr(self.config, "randomize_com", False):
                com_delta = (
                    torch.rand(3, generator=generator, device=bundle.device, dtype=bundle.dtype)
                    * 2.0
                    - 1.0
                ) * randomization.com_range
                bundle.torch_model.body_ipos[trunk] += com_delta
                bundle.native_model.body_ipos[trunk] += com_delta.detach().cpu().numpy()
            if getattr(self.config, "randomize_head_com", False):
                for body_id in bundle.handle("head_body_ids"):
                    delta = (
                        torch.rand(3, generator=generator, device=bundle.device, dtype=bundle.dtype)
                        * 2.0
                        - 1.0
                    ) * randomization.head_com_range
                    bundle.torch_model.body_ipos[body_id] += delta
                    bundle.native_model.body_ipos[body_id] += delta.detach().cpu().numpy()
            if getattr(self.config, "randomize_foot_friction", False) and bundle.handle(
                "foot_geom_groups"
            ):
                foot_scale = backend.sample_range(*randomization.foot_friction_range)
                foot_ids = sorted(
                    {geom_id for group in bundle.handle("foot_geom_groups") for geom_id in group}
                )
                bundle.torch_model.geom_friction[foot_ids] = (
                    base_geom_friction[foot_ids] * foot_scale
                )
                bundle.native_model.geom_friction[foot_ids] = base_native_geom_friction[
                    foot_ids
                ] * float(foot_scale)
            if getattr(self.config, "randomize_armature", False):
                armature_scale = backend.sample_range(*randomization.armature_range)
                bundle.torch_model.dof_armature[bundle.qvel_indices] = (
                    base_dof_armature[bundle.qvel_indices] * armature_scale
                )
                indices = bundle.qvel_indices.cpu().numpy()
                bundle.native_model.dof_armature[indices] = base_native_dof_armature[
                    indices
                ] * float(armature_scale)
            if self.actuator_mode == "bam":
                vin = backend.sample_range(*randomization.vin_range)
                drop_gain = backend.sample_range(*randomization.vin_drop_gain_range)
                friction_scale = (
                    backend.sample_range(*randomization.joint_friction_range)
                    if getattr(self.config, "randomize_joint_friction", False)
                    else torch.ones((), dtype=bundle.dtype, device=bundle.device)
                )
            else:
                vin = drop_gain = None
                friction_scale = torch.ones((), dtype=bundle.dtype, device=bundle.device)
            backend.recompute_model_constants()
            backend.configure_bam(vin=vin, drop_gain=drop_gain, friction_scale=friction_scale)

    def reset(
        self,
        command: torch.Tensor | None = None,
        *,
        env_ids: torch.Tensor | slice | None = None,
        seed: int | None = None,
        randomize: bool | None = None,
    ) -> torch.Tensor:
        if env_ids is not None:
            if self.state is None:
                raise RuntimeError("Partial reset requires an initialized environment")
            return self._reset_selected(env_ids, command=command, seed=seed, randomize=randomize)
        reset_ids = self._ids(None)
        self.physics.set_seed(seed)
        self._generator = self.physics._generator
        if seed is not None:
            self.terrain_manager.set_seed(seed)
        if randomize is not None:
            self.domain_randomization = randomize
        if command is not None:
            self.command_manager.set_command(command)
        self.terrain_manager.reset(reset_ids)
        self._restore_model_defaults(reset_ids)
        self._apply_domain_randomization(reset_ids)
        qpos = self.bundle.default_qpos.clone()
        if self.num_envs > 1:
            qpos = qpos.unsqueeze(0).expand(self.num_envs, -1).clone()
            self._apply_environment_origins(qpos)
        elif self.terrain_manager.env_origins.numel():
            self._apply_environment_origins(qpos)
        initial_height_range = getattr(self.config, "initial_height_range", None)
        root_height_index = self._root_height_index()
        if (
            self.domain_randomization
            and initial_height_range is not None
            and root_height_index is not None
        ):
            # The task's height is relative to the selected terrain origin.
            # Assigning an absolute z here erases generated ramp/rough-terrain
            # elevations and can put a reset body inside the support geometry.
            qpos[..., root_height_index] += self._sample_range(*initial_height_range)
        if self.domain_randomization and getattr(self.config, "randomize_base_orientation", False):
            randomization = getattr(self.config, "randomization", None)
            if randomization is None:
                raise RuntimeError("Base orientation randomization requires randomization ranges")
            roll = torch.deg2rad(
                self._sample_range(
                    -randomization.base_roll_degrees,
                    randomization.base_roll_degrees,
                )
            )
            pitch = torch.deg2rad(
                self._sample_range(
                    -randomization.base_pitch_degrees,
                    randomization.base_pitch_degrees,
                )
            )
            primary = self.bundle.entities.get("robot") or next(iter(self.bundle.entities.values()))
            free_indices = primary.free_qpos_indices
            if free_indices.numel() != 7:
                raise ValueError("Base orientation randomization requires a free primary entity")
            qpos[..., free_indices[3:7]] = _quat_from_euler(roll, pitch, torch.zeros_like(roll))
        qvel = self.bundle.default_qvel.clone()
        if self.num_envs > 1:
            qvel = qvel.unsqueeze(0).expand(self.num_envs, -1).clone()
        reset_ctrl = (
            torch.zeros(
                self.bundle.native_model.nu,
                dtype=self.bundle.dtype,
                device=self.bundle.device,
            )
            if self.actuator_mode == "bam"
            else self.bundle.default_ctrl.clone()
        )
        if self.num_envs > 1:
            reset_ctrl = reset_ctrl.unsqueeze(0).expand(self.num_envs, -1).clone()
        self.physics.reset(qpos=qpos, qvel=qvel, ctrl=reset_ctrl)
        self._velocity_reward_cache = None
        self.sensor_manager.reset(self)
        base_ang_vel, gravity = self._initial_observation()
        randomization = getattr(self.config, "randomization", None)
        data = self.data
        if data is None:
            raise RuntimeError("Physics backend did not produce data during reset")
        foot_positions = (
            data.site_xpos[..., list(self.bundle.handle("foot_site_ids")), :].clone()
            if self._velocity_state_enabled and self.bundle.handle("foot_site_ids")
            else None
        )
        zero_action = torch.zeros(
            self.task_cfg.action_size, dtype=self.bundle.dtype, device=self.bundle.device
        )
        if self.num_envs > 1:
            zero_action = zero_action.unsqueeze(0).expand(self.num_envs, -1).clone()
        encoder_bias_zero = torch.zeros(
            (self.num_envs, self.bundle.action_size)
            if self.num_envs > 1
            else (self.bundle.action_size,),
            dtype=self.bundle.dtype,
            device=self.bundle.device,
        )
        low_delay, high_delay = self.action_delay_range
        delay_lag = self._sample_delay(low_delay, high_delay)
        if (
            self.domain_randomization
            and getattr(self.config, "randomize_actuator_delay", False)
            and self.action_delay_range == (0, 0)
        ):
            delay_lag = self._sample_delay(3, 6)
        imu_lag_range = getattr(self.config, "imu_delay_lag", (0, 0))
        imu_lag = self._sample_delay(*imu_lag_range) if self.domain_randomization else 0
        self.state = EnvironmentState(
            sensors=SensorState(
                last_action=zero_action.clone(),
                previous_action=zero_action.clone(),
                previous_joint_velocity=self._encoder_velocity().clone(),
                previous_foot_positions=foot_positions,
                foot_air_time=(
                    torch.zeros(
                        (
                            self.num_envs,
                            len(self.bundle.handle("foot_geom_groups")),
                        )
                        if self.num_envs > 1
                        else len(self.bundle.handle("foot_geom_groups")),
                        dtype=self.bundle.dtype,
                        device=self.bundle.device,
                    )
                    if self._velocity_state_enabled and self.bundle.handle("foot_geom_groups")
                    else None
                ),
                foot_contact=(
                    foot_contact_mask(self.data, self.bundle)
                    if self._velocity_state_enabled
                    else None
                ),
                imu_ang_vel_history=(
                    []
                    if base_ang_vel is None or not self._velocity_state_enabled
                    else [base_ang_vel]
                ),
                projected_gravity_history=(
                    [] if base_ang_vel is None or not self._velocity_state_enabled else [gravity]
                ),
                delay_buffer=[
                    zero_action.clone() for _ in range(max(self._lag_value(delay_lag) + 1, 1))
                ],
                delay_lag=delay_lag,
                imu_lag=imu_lag,
                encoder_bias=(
                    sample_uniform(
                        (getattr(randomization, "encoder_bias_range", (0.0, 0.0)),)
                        * self.bundle.action_size,
                        generator=(
                            self.physics.generators if self.num_envs > 1 else self._generator
                        ),
                        device=self.bundle.device,
                        dtype=self.bundle.dtype,
                        batch_size=self.num_envs,
                    )
                    if self.domain_randomization
                    and getattr(self.config, "randomize_encoder_bias", False)
                    and randomization is not None
                    else encoder_bias_zero
                ),
                imu_quaternion=(
                    torch.tensor(
                        [1.0, 0.0, 0.0, 0.0],
                        dtype=self.bundle.dtype,
                        device=self.bundle.device,
                    )
                    .unsqueeze(0)
                    .expand(self.num_envs, -1)
                    .clone()
                    if self.num_envs > 1
                    else torch.tensor(
                        [1.0, 0.0, 0.0, 0.0],
                        dtype=self.bundle.dtype,
                        device=self.bundle.device,
                    )
                ),
            ),
            reward_terms={},
            # A full reset starts a new task episode.  Task components are
            # re-created/reset below; carrying the previous dictionary here
            # leaks episode state through the lifecycle boundary.
            task_data={},
            pending_reset=(
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.bundle.device)
                if self.num_envs > 1
                else False
            ),
        )
        if not self._startup_events_applied:
            self.event_manager.startup(self)
            self._startup_events_applied = True
        # ``None`` is the full-reset sentinel for every manager.  Passing an
        # all-environment tensor would look like a partial reset to managers
        # that must clear schedules and episode-local term state.
        self.action_manager.reset(self)
        if (
            self.domain_randomization
            and getattr(self.config, "randomize_imu_orientation", False)
            and randomization is not None
            and getattr(randomization, "imu_angle_degrees", 0.0) > 0
        ):
            axis = self._random_tensor(
                (self.num_envs, 3) if self.num_envs > 1 else (3,),
                dtype=self.bundle.dtype,
                normal=True,
            )
            axis = axis / (torch.linalg.vector_norm(axis, dim=-1, keepdim=True) + 1e-8)
            angle = torch.deg2rad(self._random() * randomization.imu_angle_degrees)
            half = angle / 2.0
            self.state.sensors.imu_quaternion = torch.cat(
                (torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)),
                dim=-1,
            )
        self.event_manager.reset(self)
        # Reset events are allowed to mutate qpos/qvel.  Refresh all derived
        # quantities and history baselines after those mutations so the first
        # policy step cannot observe a synthetic foot velocity/contact edge.
        self.physics.forward()
        self.sensor_manager.update(self)
        base_ang_vel, gravity = self._initial_observation()
        data = self.data
        if data is None:
            raise RuntimeError("Physics backend lost data after reset event")
        self.state.sensors.previous_joint_velocity = self._encoder_velocity().clone()
        if self.bundle.handle("foot_site_ids"):
            self.state.sensors.previous_foot_positions = data.site_xpos[
                ..., list(self.bundle.handle("foot_site_ids")), :
            ].clone()
        self.state.sensors.foot_contact = (
            foot_contact_mask(data, self.bundle) if self._velocity_state_enabled else None
        )
        self.state.sensors.imu_ang_vel_history = [] if base_ang_vel is None else [base_ang_vel]
        self.state.sensors.projected_gravity_history = [] if base_ang_vel is None else [gravity]
        # Task-specific persistent state is reset after physical reset events
        # and the final forward pass, so a phase/prop/task term sees the actual
        # episode-start state rather than pre-event defaults.
        self.task_state_manager.reset(self)
        self._sync_task_state()
        self.command_manager.reset(self)
        self.observation_manager.reset(self)
        self.reward_manager.reset(self)
        self.termination_manager.reset(self)
        self.curriculum_manager.reset(self)
        return self.observation()

    def _reset_selected(
        self,
        env_ids: torch.Tensor | slice,
        *,
        command: torch.Tensor | None,
        seed: int | None,
        randomize: bool | None,
    ) -> torch.Tensor:
        """Reset selected vector rows without disturbing live siblings."""

        ids = self._ids(env_ids)
        if not ids.numel():
            return self.observation()
        if self.num_envs == 1:
            # There is no sibling row to preserve in the scalar backend.  Use
            # the canonical full-reset lifecycle so scalar and batched reset
            # contracts cannot drift.
            return self.reset(command=command, seed=seed, randomize=randomize)
        self.physics.set_seed(seed, env_ids=ids)
        self._generator = self.physics._generator
        if seed is not None:
            self.terrain_manager.set_seed(seed, env_ids=ids)
        if randomize is not None:
            self.domain_randomization = randomize
        if command is not None:
            command_value = torch.as_tensor(command)
            if command_value.shape == (self.command_manager.size,):
                self.command_manager.set_command(command_value, ids)
            else:
                self.command_manager.set_command(command_value, ids)
        self.terrain_manager.reset(ids)
        self._restore_model_defaults(ids)
        self._apply_domain_randomization(ids)
        qpos = self.bundle.default_qpos.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._apply_environment_origins(qpos)
        initial_height_range = getattr(self.config, "initial_height_range", None)
        root_height_index = self._root_height_index()
        if (
            self.domain_randomization
            and initial_height_range is not None
            and root_height_index is not None
        ):
            qpos[ids, root_height_index] += self._sample_range(*initial_height_range, env_ids=ids)
        qvel = self.bundle.default_qvel.unsqueeze(0).expand(self.num_envs, -1).clone()
        reset_ctrl = (
            (
                torch.zeros_like(self.bundle.default_ctrl)
                if self.actuator_mode == "bam"
                else self.bundle.default_ctrl.clone()
            )
            .unsqueeze(0)
            .expand(self.num_envs, -1)
            .clone()
        )
        self.physics.reset(qpos=qpos, qvel=qvel, ctrl=reset_ctrl, env_ids=ids)
        self.sensor_manager.reset(self, ids)
        data = self.data
        if data is None or self.state is None:
            raise RuntimeError("Physics backend did not produce data during partial reset")
        zero_action = torch.zeros(
            (self.num_envs, self.task_cfg.action_size),
            dtype=self.bundle.dtype,
            device=self.bundle.device,
        )
        sensor = self.state.sensors
        sensor.last_action[ids] = zero_action[ids]
        sensor.previous_action[ids] = zero_action[ids]
        sensor.previous_joint_velocity[ids] = self._encoder_velocity()[ids]
        if sensor.previous_foot_positions is not None:
            sensor.previous_foot_positions[ids] = data.site_xpos[
                ..., list(self.bundle.handle("foot_site_ids")), :
            ][ids]
        if sensor.foot_air_time is not None:
            sensor.foot_air_time[ids] = 0
        sensor.foot_contact = (
            foot_contact_mask(data, self.bundle) if self._velocity_state_enabled else None
        )
        if isinstance(self.state.pending_reset, torch.Tensor):
            self.state.pending_reset[ids] = False
        else:
            self.state.pending_reset = False
        randomization = getattr(self.config, "randomization", None)
        if (
            self.domain_randomization
            and getattr(self.config, "randomize_encoder_bias", False)
            and randomization is not None
        ):
            encoder_bias = sample_uniform(
                (getattr(randomization, "encoder_bias_range", (0.0, 0.0)),)
                * self.bundle.action_size,
                generator=(self.physics.generators if self.num_envs > 1 else self._generator),
                device=self.bundle.device,
                dtype=self.bundle.dtype,
                batch_size=self.num_envs,
            )
            if sensor.encoder_bias.ndim == 2:
                sensor.encoder_bias[ids] = encoder_bias[ids]
            else:
                sensor.encoder_bias = encoder_bias
        elif sensor.encoder_bias.ndim == 2:
            sensor.encoder_bias[ids] = 0
        else:
            sensor.encoder_bias.zero_()
        if sensor.imu_quaternion.ndim == 2:
            sensor.imu_quaternion[ids] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=self.bundle.dtype, device=self.bundle.device
            )
        else:
            sensor.imu_quaternion = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=self.bundle.dtype, device=self.bundle.device
            )
        # Re-sample per-episode IMU misalignment for exactly the rows being
        # reset.  A sibling must retain its calibration while a partial reset
        # starts a fresh episode.
        if (
            self.domain_randomization
            and getattr(self.config, "randomize_imu_orientation", False)
            and randomization is not None
            and getattr(randomization, "imu_angle_degrees", 0.0) > 0
        ):
            axis = self._random_tensor((self.num_envs, 3), dtype=self.bundle.dtype, normal=True)
            axis = axis / (torch.linalg.vector_norm(axis, dim=-1, keepdim=True) + 1e-8)
            angle = torch.deg2rad(
                self._random_tensor((self.num_envs,), dtype=self.bundle.dtype)
                * randomization.imu_angle_degrees
            )
            half = angle / 2.0
            quaternion = torch.cat(
                (torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)),
                dim=-1,
            )
            sensor.imu_quaternion[ids] = quaternion[ids]

        self.action_manager.reset(self, ids)
        self.event_manager.reset(self, ids)
        self.physics.forward(env_ids=ids)
        self.sensor_manager.update(self)
        base_ang_vel, gravity = self._initial_observation()
        if base_ang_vel is not None:
            if sensor.imu_ang_vel_history:
                for history in sensor.imu_ang_vel_history:
                    history[ids] = base_ang_vel[ids]
                for history in sensor.projected_gravity_history:
                    history[ids] = gravity[ids]
            else:
                sensor.imu_ang_vel_history = [base_ang_vel]
                sensor.projected_gravity_history = [gravity]
        for buffered_action in sensor.delay_buffer:
            buffered_action[ids] = 0
        sensor.delay_lag = self._sample_delay(*self.action_delay_range, env_ids=ids)
        if (
            self.domain_randomization
            and getattr(self.config, "randomize_actuator_delay", False)
            and self.action_delay_range == (0, 0)
        ):
            sensor.delay_lag = self._sample_delay(3, 6, env_ids=ids)
        sensor.imu_lag = (
            self._sample_delay(*getattr(self.config, "imu_delay_lag", (0, 0)), env_ids=ids)
            if self.domain_randomization
            else 0
        )
        sensor.previous_joint_velocity[ids] = self._encoder_velocity()[ids]
        if sensor.previous_foot_positions is not None:
            sensor.previous_foot_positions[ids] = data.site_xpos[
                ..., list(self.bundle.handle("foot_site_ids")), :
            ][ids]
        sensor.foot_contact = (
            foot_contact_mask(data, self.bundle) if self._velocity_state_enabled else None
        )
        self.task_state_manager.reset(self, ids)
        self._sync_task_state()
        self.command_manager.reset(self, ids)
        self.observation_manager.reset(self, ids)
        self.reward_manager.reset(self, ids)
        self.termination_manager.reset(self, ids)
        self.curriculum_manager.reset(self, ids)
        return self.observation()

    def _next_interval_step(self, interval: tuple[float, float]) -> int:
        seconds = float(self._sample_range(*interval).reshape(-1)[0])
        return max(1, int(round(seconds / (self.bundle.timestep * self.decimation))))

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

    def observation(self, group: str = "actor") -> torch.Tensor:
        """Compute one named observation group."""

        return self.observation_manager.compute(self, group)

    def observations(self) -> dict[str, torch.Tensor]:
        """Compute every enabled configured observation group."""

        return {
            name: self.observation(name)
            for name, group in self.task_cfg.observations.groups.items()
            if group.enabled
        }

    def _apply_push(self) -> None:
        if self.data is None or self.state is None:
            return
        if not self.domain_randomization or not getattr(
            self.config, "randomize_velocity_pushes", False
        ):
            return
        randomization = getattr(self.config, "randomization", None)
        if randomization is None:
            return
        push_low, push_high = randomization.velocity_push_range
        push = torch.stack(
            (self._sample_range(push_low, push_high), self._sample_range(push_low, push_high)),
            dim=-1,
        )
        if self.num_envs == 1 and push.ndim == 2:
            push = push.squeeze(0)
        qvel = self.data.qvel.clone()
        qvel[..., :2] += push[..., :2]
        self.physics.forward(qvel=qvel)

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
        sensor = self.state.sensors
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
        self.task_state_manager.pre_physics(self, control_dt)
        self._sync_task_state()
        self.event_manager.apply(self, "pre_physics")
        previous_foot_contact = (
            sensor.foot_contact.clone() if sensor.foot_contact is not None else None
        )
        previous_foot_air_time = (
            sensor.foot_air_time.clone() if sensor.foot_air_time is not None else None
        )
        applied_action, target = self.action_manager.prepare(self, action)
        direct_ctrl = self.action_manager.current_ctrl
        physics_step = self.physics.step
        try:
            step_parameters = inspect.signature(physics_step).parameters
        except (TypeError, ValueError):
            step_parameters = {}
        accepts_target_type = "target_type" in step_parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in step_parameters.values()
        )
        if accepts_target_type:
            kwargs: dict[str, Any] = {"target_type": self.action_manager.target_type}
            if direct_ctrl is not None:
                kwargs["direct_ctrl"], kwargs["direct_ctrl_mask"] = direct_ctrl
            physics_step(target, **kwargs)
        else:
            # Keep instrumentation/adapters with the original one-argument
            # backend seam valid without masking exceptions raised inside the
            # actual physics implementation.
            physics_step(target)
        sensor.last_action = action.clone()
        self.task_state_manager.post_physics(self, control_dt)
        self._sync_task_state()
        current_contact = (
            self.sensor_manager.foot_contact_mask(
                self.bundle, fallback=lambda: foot_contact_mask(self.data, self.bundle)
            )
            if self._velocity_state_enabled
            else None
        )
        touchdown = (
            current_contact & ~previous_foot_contact if previous_foot_contact is not None else None
        )
        if sensor.foot_air_time is not None:
            sensor.foot_air_time = torch.where(
                current_contact,
                torch.zeros_like(sensor.foot_air_time),
                sensor.foot_air_time + self.bundle.timestep * self.decimation,
            )
        sensor.foot_contact = current_contact
        base_ang_vel, gravity = self._initial_observation()
        if base_ang_vel is not None:
            sensor.imu_ang_vel_history.append(base_ang_vel)
            sensor.projected_gravity_history.append(gravity)
            if len(sensor.imu_ang_vel_history) > 4:
                sensor.imu_ang_vel_history.pop(0)
                sensor.projected_gravity_history.pop(0)
        if (
            self.domain_randomization
            and getattr(self.config, "delay_update_period", 0) > 0
            and self.step_count % getattr(self.config, "delay_update_period", 1) == 0
        ):
            sensor.imu_lag = self._sample_delay(*getattr(self.config, "imu_delay_lag", (0, 0)))
        # Post-physics events observe the freshly integrated state.  Reward and
        # termination terms use the command that produced this transition;
        # command resampling happens after them, matching upstream.
        self.event_manager.apply(self, "post_physics")
        transition = TransitionData(
            action=action,
            previous_action=sensor.previous_action,
            previous_foot_positions=sensor.previous_foot_positions,
            foot_air_time=previous_foot_air_time,
            foot_contact=current_contact,
            foot_touchdown=touchdown,
        )
        self.state.transition = transition
        self._velocity_reward_cache = None
        reward, terms = self.reward_manager.compute(self)
        self.state.reward_terms = terms
        if self.bundle.handle("foot_site_ids"):
            sensor.previous_foot_positions = self.data.site_xpos[
                ..., list(self.bundle.handle("foot_site_ids")), :
            ].clone()
        finite = (
            torch.isfinite(self.data.qpos).all(dim=-1)
            & torch.isfinite(self.data.qvel).all(dim=-1)
            & torch.isfinite(reward)
        )
        if self.num_envs == 1:
            finite = bool(finite.item())
        terminated, truncated, termination_values = self.termination_manager.evaluate(
            self, finite=finite
        )
        bad_orientation_value = termination_values.get("bad_orientation", False)
        self.curriculum_manager.step(self)
        self.command_manager.step(self)
        self.task_state_manager.step(self, control_dt)
        self._sync_task_state()
        self.event_manager.apply(self, "step")
        self.event_manager.apply(self, "interval")
        observation = self.observation()
        observation_finite = torch.isfinite(observation).all(dim=-1)
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
        time_value = self.data.time.detach().clone()
        if isinstance(time_value, torch.Tensor) and time_value.numel() == 1:
            time_value = float(time_value.item())
        info: dict[str, Any] = {
            "step": self.step_count if self.num_envs == 1 else self.step_counts.detach().clone(),
            "time": time_value,
            "finite": finite,
            "bad_orientation": bad_orientation_value,
            "terminations": termination_values,
            "reward_terms": {
                name: (float(value.item()) if value.numel() == 1 else value.detach().clone())
                for name, value in terms.items()
            },
            "applied_action": applied_action.detach().clone(),
        }
        done = torch.as_tensor(
            terminated, dtype=torch.bool, device=self.bundle.device
        ) | torch.as_tensor(truncated, dtype=torch.bool, device=self.bundle.device)
        # Preserve the terminal observation for learning code even when the
        # manager-owned lifecycle immediately starts the next episode.
        info["terminal_observation"] = observation.detach().clone()
        if self.auto_reset:
            if self.num_envs == 1:
                if bool(done.item()):
                    observation = self.reset()
            else:
                done_ids = torch.nonzero(done, as_tuple=False).reshape(-1)
                if done_ids.numel():
                    observation = self._reset_selected(
                        done_ids, command=None, seed=None, randomize=None
                    )
        elif self.num_envs == 1:
            if bool(done.item()):
                self.state.pending_reset = True
        else:
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
