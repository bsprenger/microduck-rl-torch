"""Generic episode-start state assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from ..model import _joint_qpos_width, _joint_qvel_width
from ..task_config import ResetStateTermCfg, TermCollection
from .base import resolve_term


@dataclass
class ResetState:
    """Mutable complete physical state passed through reset terms."""

    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor
    # Optional backend state for entities that are not represented by qpos /
    # qvel (for example mocap bodies).  A reset term may allocate and replace
    # these fields; the physics backend applies them atomically with the
    # generalized coordinates.
    mocap_pos: torch.Tensor | None = None
    mocap_quat: torch.Tensor | None = None
    xfrc_applied: torch.Tensor | None = None


class ResetManager:
    """Compose generic scene defaults with task-owned reset-state terms.

    The manager owns the reset-state mutation boundary.  The environment only
    asks for a fully assembled state and passes it to the physics backend; it
    never needs to know whether a task starts standing, prone, mid-roll, or
    with a dynamic prop in a task-relative pose.
    """

    def __init__(self, terms: TermCollection) -> None:
        self.terms = terms
        self._resolved_terms: dict[str, Any] = {}

    @staticmethod
    def _ids(env: Any, env_ids: torch.Tensor | slice | None) -> torch.Tensor | slice | None:
        return env_ids

    def _term(self, name: str, cfg: Any, env: Any) -> Any:
        if name not in self._resolved_terms:
            self._resolved_terms[name] = resolve_term(cfg, env)
        return self._resolved_terms[name]

    @staticmethod
    def _invoke(
        term: Any, env: Any, state: ResetState, env_ids: Any, params: dict[str, Any]
    ) -> Any:
        callback = getattr(term, "apply", term)
        if not callable(callback):
            raise TypeError(f"Reset term {type(term).__name__} is not callable")
        result = callback(env, state, env_ids, **params)
        if result is not None and not isinstance(result, ResetState):
            raise TypeError(
                f"Reset term {callback!r} returned {type(result).__name__}; "
                "expected ResetState or None"
            )
        return result

    @staticmethod
    def _assert_selected_rows_unchanged(
        before: ResetState,
        after: ResetState,
        env: Any,
        env_ids: torch.Tensor | slice | None,
    ) -> None:
        """Reject reset terms that mutate rows outside a partial reset.

        Reset terms intentionally receive the full transaction so they can
        express global tensor operations.  For a partial reset, however, the
        manager must make accidental full-batch writes fail loudly rather than
        corrupting live sibling episodes.
        """

        if env_ids is None or env.num_envs == 1:
            return
        ids = env._ids(env_ids)
        selected = torch.zeros(env.num_envs, dtype=torch.bool, device=ids.device)
        selected[ids] = True
        outside = ~selected

        def changed(before_value: torch.Tensor | None, after_value: torch.Tensor | None) -> bool:
            if before_value is None or after_value is None:
                return before_value is not after_value
            if before_value.shape != after_value.shape:
                return True
            if before_value.ndim == 0 or before_value.shape[0] != env.num_envs:
                return not torch.equal(before_value, after_value)
            return bool((before_value[outside] != after_value[outside]).any().item())

        for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "xfrc_applied"):
            if changed(getattr(before, name), getattr(after, name)):
                raise RuntimeError(
                    f"Reset term mutated {name} outside selected env_ids={ids.tolist()}"
                )

    def build(
        self,
        env: Any,
        env_ids: torch.Tensor | slice | None = None,
        *,
        apply_terms: bool = True,
    ) -> ResetState:
        """Build generic defaults and optionally apply task reset terms.

        ``apply_terms=False`` is the canonical scene/entity-default path used
        by :meth:`SceneRuntime.reset_to_default`; normal environment resets
        leave it enabled so task-specific initial poses and props participate
        in the same transaction.
        """

        qpos = env.bundle.default_qpos.clone()
        qvel = env.bundle.default_qvel.clone()
        ctrl = (
            torch.zeros(
                env.bundle.native_model.nu,
                dtype=env.bundle.dtype,
                device=env.bundle.device,
            )
            if env.actuator_mode == "bam"
            else env.bundle.default_ctrl.clone()
        )
        if env.num_envs > 1:
            qpos = qpos.unsqueeze(0).expand(env.num_envs, -1).clone()
            qvel = qvel.unsqueeze(0).expand(env.num_envs, -1).clone()
            ctrl = ctrl.unsqueeze(0).expand(env.num_envs, -1).clone()
        reset_state = ResetState(qpos=qpos, qvel=qvel, ctrl=ctrl)
        env.scene.apply_environment_origins(reset_state.qpos)
        self._apply_entity_init_states(env, reset_state, env_ids)
        if not apply_terms:
            return reset_state
        for name, cfg in self.terms.items():
            if not cfg.enabled:
                continue
            if not isinstance(cfg, ResetStateTermCfg) and not hasattr(cfg, "func"):
                raise TypeError(f"Reset term {name!r} has unsupported config {type(cfg).__name__}")
            term = self._term(name, cfg, env)
            before = ResetState(
                qpos=reset_state.qpos.clone(),
                qvel=reset_state.qvel.clone(),
                ctrl=reset_state.ctrl.clone(),
                mocap_pos=None if reset_state.mocap_pos is None else reset_state.mocap_pos.clone(),
                mocap_quat=(
                    None if reset_state.mocap_quat is None else reset_state.mocap_quat.clone()
                ),
                xfrc_applied=(
                    None if reset_state.xfrc_applied is None else reset_state.xfrc_applied.clone()
                ),
            )
            result = self._invoke(term, env, reset_state, env_ids, dict(getattr(cfg, "params", {})))
            if result is not None:
                self._assert_selected_rows_unchanged(before, result, env, env_ids)
                reset_state = result
            else:
                self._assert_selected_rows_unchanged(before, reset_state, env, env_ids)
        return reset_state

    @staticmethod
    def _apply_entity_init_states(
        env: Any, state: ResetState, env_ids: torch.Tensor | slice | None
    ) -> None:
        """Apply generic per-entity root/joint reset declarations."""

        selected = env._ids(env_ids) if env_ids is not None else None

        def assign(target: torch.Tensor, indices: torch.Tensor, value: torch.Tensor) -> None:
            if env.num_envs == 1:
                target[indices] = value
            elif selected is None:
                target[:, indices] = value
            else:
                target[selected[:, None], indices] = value

        for entity_name, cfg in env.bundle.entity_configs.items():
            init = cfg.init_state
            view = env.bundle.entity(entity_name)
            free_pose = view.free_qpos_indices
            free_vel = view.free_qvel_indices
            if free_pose.numel() == 7:
                pose = state.qpos[..., free_pose].clone()
                if init.pos is not None:
                    position = torch.as_tensor(
                        init.pos, dtype=state.qpos.dtype, device=state.qpos.device
                    )
                    if env.num_envs == 1:
                        position = position + env.terrain_manager.env_origins[0]
                    else:
                        position = position + env.terrain_manager.env_origins
                    pose[..., :3] = position
                if init.quat is not None:
                    pose[..., 3:] = torch.as_tensor(
                        init.quat, dtype=state.qpos.dtype, device=state.qpos.device
                    )
                pose_value = pose if selected is None or env.num_envs == 1 else pose[selected]
                assign(state.qpos, free_pose, pose_value)
                velocity = state.qvel[..., free_vel].clone()
                if init.linear_velocity is not None:
                    velocity[..., :3] = torch.as_tensor(
                        init.linear_velocity, dtype=state.qvel.dtype, device=state.qvel.device
                    )
                if init.angular_velocity is not None:
                    angular = torch.as_tensor(
                        init.angular_velocity, dtype=state.qvel.dtype, device=state.qvel.device
                    )
                    quat = pose[..., 3:]
                    inverse = torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)
                    twice = 2.0 * torch.cross(inverse[..., 1:], angular, dim=-1)
                    velocity[..., 3:] = (
                        angular
                        + inverse[..., :1] * twice
                        + torch.cross(inverse[..., 1:], twice, dim=-1)
                    )
                velocity_value = (
                    velocity if selected is None or env.num_envs == 1 else velocity[selected]
                )
                assign(state.qvel, free_vel, velocity_value)

            non_free_names = tuple(
                name
                for joint_id, name in zip(view.joint_ids, view.joint_names, strict=True)
                if joint_id in set(view.non_free_joint_ids)
            )

            def matching_joint_ids(
                pattern: str,
                *,
                entity_name: str = entity_name,
                non_free_names: tuple[str, ...] = non_free_names,
                non_free_joint_ids: tuple[int, ...] = tuple(view.non_free_joint_ids),
            ) -> tuple[int, ...]:
                matches = tuple(
                    joint_id
                    for joint_id, name in zip(non_free_joint_ids, non_free_names, strict=True)
                    if name == pattern or re.search(pattern, name) is not None
                )
                if not matches:
                    raise KeyError(
                        f"Initial joint pattern {pattern!r} not found in entity {entity_name!r}"
                    )
                return matches

            for name, value in init.joint_pos.items():
                for joint_id in matching_joint_ids(name):
                    start = int(env.bundle.native_model.jnt_qposadr[joint_id])
                    width = _joint_qpos_width(int(env.bundle.native_model.jnt_type[joint_id]))
                    indices = torch.arange(start, start + width, device=state.qpos.device)
                    values = torch.as_tensor(
                        value, dtype=state.qpos.dtype, device=state.qpos.device
                    ).reshape(-1)
                    if values.numel() not in {1, width}:
                        raise ValueError(f"Initial joint {name!r} needs {width} position values")
                    if values.numel() == 1:
                        values = values.expand(width)
                    if env.num_envs == 1:
                        assign(state.qpos, indices, values)
                    else:
                        assign(
                            state.qpos,
                            indices,
                            values.expand(env.num_envs, -1)
                            if selected is None
                            else values.expand(selected.numel(), -1),
                        )
            for name, value in init.joint_vel.items():
                for joint_id in matching_joint_ids(name):
                    start = int(env.bundle.native_model.jnt_dofadr[joint_id])
                    width = _joint_qvel_width(int(env.bundle.native_model.jnt_type[joint_id]))
                    indices = torch.arange(start, start + width, device=state.qvel.device)
                    values = torch.as_tensor(
                        value, dtype=state.qvel.dtype, device=state.qvel.device
                    ).reshape(-1)
                    if values.numel() not in {1, width}:
                        raise ValueError(f"Initial joint {name!r} needs {width} velocity values")
                    if values.numel() == 1:
                        values = values.expand(width)
                    if env.num_envs == 1:
                        assign(state.qvel, indices, values)
                    else:
                        assign(
                            state.qvel,
                            indices,
                            values.expand(env.num_envs, -1)
                            if selected is None
                            else values.expand(selected.numel(), -1),
                        )


__all__ = ["ResetManager", "ResetState"]
