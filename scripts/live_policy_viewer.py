"""Run the golden Microduck policy in the Torch environment with a live viewer."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import torch

from microduck_rl_torch.envs import ManagerBasedTaskEnv
from microduck_rl_torch.policies import OnnxPolicy, fetch_policy
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg

mujoco_api: Any = mujoco


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _actor_observation(value: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(value, dict):
        raise RuntimeError("The live viewer requires a concatenated actor observation")
    return value


def _sync_native_data(env: ManagerBasedTaskEnv, native_data: Any) -> None:
    if env.data is None:
        raise RuntimeError("The environment must be reset before syncing the viewer")
    state = env.data
    native_data.qpos[:] = state.qpos.detach().cpu().numpy()
    native_data.qvel[:] = state.qvel.detach().cpu().numpy()
    native_data.ctrl[:] = state.ctrl.detach().cpu().numpy()
    native_data.time = float(state.time)
    mujoco_api.mj_forward(env.bundle.native_model, native_data)


def main() -> None:
    device = os.environ.get("MICRODUCK_DEVICE", _default_device())
    artifact = fetch_policy("alpha_walking", output_dir=Path("artifacts/hf"))
    policy = OnnxPolicy(artifact)
    env = ManagerBasedTaskEnv(make_microduck_velocity_env_cfg(), device=device)
    observation = _actor_observation(env.reset(seed=0))
    native_data = mujoco_api.MjData(env.bundle.native_model)

    try:
        with mujoco.viewer.launch_passive(env.bundle.native_model, native_data) as viewer:
            while viewer.is_running():
                started = time.monotonic()
                transition = env.step(policy(observation))
                if bool(transition.terminated) or bool(transition.truncated):
                    observation = _actor_observation(env.reset(seed=0))
                else:
                    observation = _actor_observation(transition.observation)
                _sync_native_data(env, native_data)
                viewer.sync()
                time.sleep(max(0.0, env.step_dt - (time.monotonic() - started)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
