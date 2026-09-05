"""Deterministic native golden trajectories and Torch comparison helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from microduck_rl_torch.envs import ManagerBasedTaskEnv
from microduck_rl_torch.envs.model import MicroDuckModelBundle
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg

from .native import NativeMicroDuckEnv


@dataclass(frozen=True)
class GoldenTrajectory:
    """A compact, versioned transition tape produced by native MuJoCo."""

    metadata: dict[str, Any]
    observations: np.ndarray
    actions: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    ctrl: np.ndarray
    sensordata: np.ndarray
    times: np.ndarray
    foot_contacts: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
            observations=self.observations,
            actions=self.actions,
            qpos=self.qpos,
            qvel=self.qvel,
            qacc=self.qacc,
            ctrl=self.ctrl,
            sensordata=self.sensordata,
            times=self.times,
            foot_contacts=self.foot_contacts,
            rewards=self.rewards,
            terminated=self.terminated,
            truncated=self.truncated,
        )

    @classmethod
    def load(cls, path: Path) -> GoldenTrajectory:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                metadata=json.loads(str(data["metadata"])),
                observations=data["observations"],
                actions=data["actions"],
                qpos=data["qpos"],
                qvel=data["qvel"],
                qacc=data["qacc"],
                ctrl=data["ctrl"],
                sensordata=data["sensordata"],
                times=data["times"],
                foot_contacts=data["foot_contacts"],
                rewards=data["rewards"],
                terminated=data["terminated"],
                truncated=data["truncated"],
            )


def _native_termination(env: NativeMicroDuckEnv) -> tuple[bool, bool]:
    finite = bool(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all())
    quaternion = env.data.xquat[env.trunk_body_id]
    cos_tilt = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
    bad_orientation = cos_tilt < np.cos(np.deg2rad(70.0))
    return (not finite or bool(bad_orientation), env._step_count >= 1000)


def generate_native_trajectory(
    bundle: MicroDuckModelBundle,
    actions: np.ndarray,
    *,
    command: np.ndarray,
    metadata: dict[str, Any] | None = None,
    disable_contacts: bool | None = None,
    action_delay_lag: int = 0,
) -> GoldenTrajectory:
    """Roll a fixed action tape through native MuJoCo and record every field."""

    actions = np.asarray(actions, dtype=np.float64)
    command = np.asarray(command, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != bundle.action_size:
        raise ValueError(f"Expected actions with shape (steps, 14), got {actions.shape}")
    if command.shape != (13,):
        raise ValueError(f"Expected command shape (13,), got {command.shape}")
    contacts_enabled = bundle.contacts_enabled if disable_contacts is None else not disable_contacts
    env = NativeMicroDuckEnv(
        bundle=bundle,
        timestep=bundle.timestep,
        decimation=bundle.decimation,
        solver_iterations=bundle.solver_iterations,
        line_search_iterations=bundle.line_search_iterations,
        disable_contacts=disable_contacts is True,
        action_delay_lag=action_delay_lag,
    )
    env.command[:] = command
    observations = [env.reset()]
    initial = env.snapshot()
    qpos = [initial["qpos"]]
    qvel = [initial["qvel"]]
    qacc = [initial["qacc"]]
    ctrl = [initial["ctrl"]]
    sensordata = [initial["sensordata"]]
    times = [initial["time"]]
    foot_contacts = [initial["foot_contact"]]
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    for action in actions:
        observations.append(env.step(action))
        snapshot = env.snapshot()
        qpos.append(snapshot["qpos"])
        qvel.append(snapshot["qvel"])
        qacc.append(snapshot["qacc"])
        ctrl.append(snapshot["ctrl"])
        sensordata.append(snapshot["sensordata"])
        times.append(snapshot["time"])
        foot_contacts.append(snapshot["foot_contact"])
        rewards.append(env.last_reward)
        done, timeout = _native_termination(env)
        terminated.append(done)
        truncated.append(timeout)
    trajectory_metadata = {
        "schema_version": 1,
        "reference": "mujoco-c",
        "model": bundle.fingerprint(),
        "command": command.tolist(),
        "steps": int(actions.shape[0]),
        "contacts_enabled": contacts_enabled,
        "action_delay_lag": action_delay_lag,
    }
    if metadata:
        trajectory_metadata.update(metadata)
    return GoldenTrajectory(
        metadata=trajectory_metadata,
        observations=np.asarray(observations, dtype=np.float32),
        actions=actions,
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        qacc=np.asarray(qacc),
        ctrl=np.asarray(ctrl),
        sensordata=np.asarray(sensordata),
        times=np.asarray(times, dtype=np.float64),
        foot_contacts=np.asarray(foot_contacts, dtype=bool),
        rewards=np.asarray(rewards, dtype=np.float64),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
    )


def generate_action_tape(steps: int, *, seed: int = 20260903) -> np.ndarray:
    """Create a small deterministic control tape for backend regression."""

    if steps < 1:
        raise ValueError("steps must be positive")
    generator = np.random.default_rng(seed)
    return generator.normal(0.0, 0.05, size=(steps, 14)).astype(np.float64)


def rollout_torch(
    bundle: MicroDuckModelBundle,
    trajectory: GoldenTrajectory,
    *,
    command: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Replay a golden action tape through the target environment."""

    expected_contacts = trajectory.metadata.get("contacts_enabled")
    if expected_contacts is not None and bool(expected_contacts) != bundle.contacts_enabled:
        raise ValueError(
            "Golden trajectory contact setting does not match the model bundle: "
            f"fixture={expected_contacts}, bundle={bundle.contacts_enabled}"
        )
    action_delay_lag = int(trajectory.metadata.get("action_delay_lag", 0))
    task_cfg = make_microduck_velocity_env_cfg()
    task_cfg.actions.delay_lag = action_delay_lag
    environment = ManagerBasedTaskEnv(
        task_cfg,
        bundle=bundle,
        command=command,
        domain_randomization=False,
    )
    observations = [environment.reset().detach().cpu().numpy()]
    if environment.state is None:
        raise RuntimeError("Environment state disappeared during reset")
    qpos = [environment.snapshot()["qpos"].detach().cpu().numpy()]
    qvel = [environment.snapshot()["qvel"].detach().cpu().numpy()]
    qacc = [environment.snapshot()["qacc"].detach().cpu().numpy()]
    ctrl = [environment.snapshot()["ctrl"].detach().cpu().numpy()]
    sensordata = [environment.snapshot()["sensordata"].detach().cpu().numpy()]
    times = [environment.snapshot()["time"]]
    initial_foot_contact = environment.state.sensors.foot_contact
    if initial_foot_contact is None:
        raise RuntimeError("Golden trajectory requires foot-contact state")
    foot_contacts = [initial_foot_contact.detach().cpu().numpy()]
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    for action in trajectory.actions:
        result = environment.step(torch.as_tensor(action, dtype=bundle.dtype, device=bundle.device))
        snapshot = environment.snapshot()
        observations.append(result.observation.detach().cpu().numpy())
        qpos.append(snapshot["qpos"].detach().cpu().numpy())
        qvel.append(snapshot["qvel"].detach().cpu().numpy())
        qacc.append(snapshot["qacc"].detach().cpu().numpy())
        ctrl.append(snapshot["ctrl"].detach().cpu().numpy())
        sensordata.append(snapshot["sensordata"].detach().cpu().numpy())
        times.append(snapshot["time"])
        if environment.state is None:
            raise RuntimeError("Environment state disappeared during rollout")
        foot_contact = environment.state.sensors.foot_contact
        if foot_contact is None:
            raise RuntimeError("Golden trajectory requires foot-contact state")
        foot_contacts.append(foot_contact.detach().cpu().numpy())
        rewards.append(float(result.reward))
        terminated.append(result.terminated)
        truncated.append(result.truncated)
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "qpos": np.asarray(qpos),
        "qvel": np.asarray(qvel),
        "qacc": np.asarray(qacc),
        "ctrl": np.asarray(ctrl),
        "sensordata": np.asarray(sensordata),
        "times": np.asarray(times, dtype=np.float64),
        "foot_contacts": np.asarray(foot_contacts, dtype=bool),
        "rewards": np.asarray(rewards),
        "terminated": np.asarray(terminated),
        "truncated": np.asarray(truncated),
    }


def compare_trajectory(
    expected: GoldenTrajectory,
    actual: dict[str, np.ndarray],
    *,
    tolerances: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return max absolute errors and raise on shape or tolerance failures."""

    tolerances = tolerances or {
        "observations": 0.01,
        "qpos": 0.002,
        "qvel": 0.05,
        "qacc": 5.0,
        "ctrl": 1e-10,
        "sensordata": 0.05,
        "times": 1e-12,
        "rewards": 0.05,
    }
    errors: dict[str, float] = {}
    for field in tolerances:
        reference = getattr(expected, field)
        candidate = actual[field]
        if reference.shape != candidate.shape:
            raise AssertionError(f"{field} shape mismatch: {reference.shape} != {candidate.shape}")
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise AssertionError(f"{field} contains non-finite values")
        error = float(np.max(np.abs(reference.astype(np.float64) - candidate.astype(np.float64))))
        errors[field] = error
        if error > tolerances[field]:
            raise AssertionError(
                f"{field} max abs error {error:.6g} exceeds {tolerances[field]:.6g}"
            )
    for field in ("terminated", "truncated"):
        reference = getattr(expected, field)
        candidate = actual[field]
        if not np.array_equal(reference, candidate):
            raise AssertionError(f"{field} mismatch: {reference} != {candidate}")
    if expected.foot_contacts.shape != actual["foot_contacts"].shape:
        raise AssertionError(
            "foot_contacts shape mismatch: "
            f"{expected.foot_contacts.shape} != {actual['foot_contacts'].shape}"
        )
    if not np.array_equal(expected.foot_contacts, actual["foot_contacts"]):
        raise AssertionError(
            f"foot_contacts mismatch: {expected.foot_contacts} != {actual['foot_contacts']}"
        )
    return errors
