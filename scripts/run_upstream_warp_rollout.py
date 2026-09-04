"""Run the upstream MicroDuck mjlab/MuJoCo-Warp task and save a parity trace.

This file intentionally imports only upstream packages plus NumPy. The parent
verification command launches it with the upstream project's interpreter and
source tree, which keeps the reference rollout independent from the local
Torch implementation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def _one_env_array(value: Any) -> np.ndarray:
    """Convert a batched Torch/NumPy value to the single environment array."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array).copy()


def _actor_observation(observations: Any) -> np.ndarray:
    try:
        value = observations["actor"]
    except (KeyError, IndexError, TypeError):
        try:
            value = observations["policy"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "Upstream observation manager returned no actor/policy group"
            ) from error
    result = _one_env_array(value).reshape(-1)
    if result.shape != (61,):
        raise RuntimeError(f"Expected upstream actor observation shape (61,), got {result.shape}")
    return result.astype(np.float32, copy=False)


def _set_command(term: Any, value: np.ndarray, *, name: str, device: Any) -> None:
    """Set a command term's live tensor without relying on private manager APIs."""

    import torch

    target = torch.as_tensor(value, device=device, dtype=torch.float32)
    if name == "twist":
        for attribute in ("vel_command_b", "vel_command_w"):
            command = getattr(term, attribute, None)
            if command is not None:
                command[0].copy_(target)
        return
    command = getattr(term, "_command", None)
    if command is None:
        command = getattr(term, "command", None)
    if command is None:
        raise RuntimeError(f"Could not locate live tensor for upstream command {name!r}")
    command[0].copy_(target)


def _make_deterministic_config(steps: int) -> Any:
    from copy import deepcopy

    config_module = importlib.import_module("mjlab_microduck.tasks.microduck_velocity_env_cfg")
    make_microduck_velocity_env_cfg = config_module.make_microduck_velocity_env_cfg

    config = make_microduck_velocity_env_cfg(play=False, rough=False)
    config.scene.num_envs = 1
    config.episode_length_s = max(float(steps) * 0.02 + 1.0, 10.0)
    config.observations["actor"].enable_corruption = False

    # Keep only startup/history setup and a deterministic base reset. All other
    # events include domain randomization or interval pushes that would make a
    # backend comparison depend on random-number-generator implementation.
    for name in tuple(config.events):
        if name not in {"expand_bam_friction_fields", "reset_action_history", "reset_base"}:
            config.events.pop(name)
    # Curriculum terms mutate the same randomization ranges and may look up the
    # event terms removed above. They are not part of this fixed-reference
    # rollout, so disable them as a unit rather than leaving dangling lookups.
    for name in tuple(config.curriculum):
        config.curriculum.pop(name)
    reset_base = config.events.get("reset_base")
    if reset_base is not None:
        pose_range = reset_base.params.get("pose_range", {})
        for key in tuple(pose_range):
            pose_range[key] = (0.12, 0.12) if key == "z" else (0.0, 0.0)
        velocity_range = reset_base.params.get("velocity_range", {})
        for key in tuple(velocity_range):
            velocity_range[key] = (0.0, 0.0)

    # The official BAM actuator has a 3--6 control-tick delay. The local
    # deterministic reference path uses zero delay; set the upstream cfg to the
    # same value and keep the observation-side one-tick joint-velocity delay.
    robot = deepcopy(config.scene.entities["robot"])
    for actuator_group in robot.articulation.actuators:
        actuator_group.delay_min_lag = 0
        actuator_group.delay_max_lag = 0
        if hasattr(actuator_group, "vin_range"):
            actuator_group.vin_range = (7.4, 7.4)
        if hasattr(actuator_group, "vin_drop_gain_range"):
            actuator_group.vin_drop_gain_range = (0.1, 0.1)
    config.scene.entities["robot"] = robot

    for command in config.commands.values():
        if hasattr(command, "resampling_time_range"):
            command.resampling_time_range = (1.0e6, 1.0e6)
    return config


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run_rollout(
    *,
    policy_path: Path,
    output_path: Path,
    steps: int,
    seed: int,
    vx: float,
    vy: float,
    vtheta: float,
    device: str,
    upstream_root: Path,
) -> None:
    import onnxruntime as ort
    import torch

    manager_module = importlib.import_module("mjlab.envs")
    manager_based_rl_env = manager_module.ManagerBasedRlEnv

    if steps < 1:
        raise ValueError("steps must be positive")
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = _make_deterministic_config(steps)
    env = manager_based_rl_env(cfg=config, device=device)
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    twist = np.asarray([vx, vy, vtheta], dtype=np.float32)
    head_pose = np.zeros(4, dtype=np.float32)
    body_pose = np.zeros(6, dtype=np.float32)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    terminated: list[bool] = []
    truncated: list[bool] = []

    try:
        env.reset(seed=seed)
        command_terms = {
            name: env.command_manager.get_term(name) for name in ("twist", "head_pose", "body_pose")
        }
        decimation = int(getattr(env.cfg, "decimation", 4))
        physics_dt = float(env.physics_dt)

        for _step in range(steps):
            _set_command(command_terms["twist"], twist, name="twist", device=env.device)
            _set_command(command_terms["head_pose"], head_pose, name="head_pose", device=env.device)
            _set_command(command_terms["body_pose"], body_pose, name="body_pose", device=env.device)
            observation_buffer = env.observation_manager.compute(update_history=True)
            observation = _actor_observation(observation_buffer)
            action = np.asarray(
                session.run([output_name], {input_name: observation[None, :]})[0], dtype=np.float32
            ).reshape(-1)
            if action.shape != (14,) or not np.isfinite(action).all():
                raise RuntimeError(
                    f"Upstream policy returned invalid action shape/value: {action.shape}"
                )

            torch_action = torch.as_tensor(action, device=env.device).reshape(1, -1)
            env.action_manager.process_action(torch_action)
            for _ in range(decimation):
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(dt=physics_dt)

            observations.append(observation)
            actions.append(action.copy())
            qpos.append(_one_env_array(env.sim.data.qpos).reshape(-1))
            qvel.append(_one_env_array(env.sim.data.qvel).reshape(-1))
            terminated.append(False)
            truncated.append(False)

        metadata = {
            "schema_version": 1,
            "backend": "upstream-mjlab-mujoco-warp",
            "task": "Mjlab-Velocity-Flat-MicroDuck",
            "upstream_root": str(upstream_root.resolve()),
            "upstream_revision": _git_revision(upstream_root),
            "policy_path": str(policy_path.resolve()),
            "seed": seed,
            "steps": steps,
            "command": [
                float(vx),
                float(vy),
                float(vtheta),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "device": str(device),
            "num_envs": 1,
            "decimation": decimation,
            "physics_dt": physics_dt,
            "control_dt": physics_dt * decimation,
            "deterministic_events": True,
            "observation_corruption": False,
            "actuator_delay_lag": 0,
            "joint_velocity_observation_delay_lag": 1,
            "domain_randomization": False,
            "qpos_dim": int(qpos[0].shape[0]),
            "qvel_dim": int(qvel[0].shape[0]),
        }
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        qpos=np.asarray(qpos, dtype=np.float64),
        qvel=np.asarray(qvel, dtype=np.float64),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vx", type=float, default=0.15)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vtheta", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args(argv)
    run_rollout(
        policy_path=args.policy_path,
        output_path=args.output,
        steps=args.steps,
        seed=args.seed,
        vx=args.vx,
        vy=args.vy,
        vtheta=args.vtheta,
        device=args.device,
        upstream_root=args.upstream_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
