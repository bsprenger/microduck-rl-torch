"""Render the Torch MicroDuck environment driven by an ONNX golden policy."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Literal

import mujoco
import mujoco_torch
import numpy as np
import torch

from microduck_rl_torch.envs import NominalMicroDuckEnv
from microduck_rl_torch.envs.model import MicroDuckModelBundle, load_microduck_model
from microduck_rl_torch.envs.observations import command_vector
from microduck_rl_torch.policies.huggingface import OnnxPolicy, PolicyArtifact

from .video import VideoWriter, convert_video_to_gif

RenderBackend = Literal["mujoco", "mujoco-torch"]
CameraName = Literal["free", "head_camera"]
mujoco_api: Any = mujoco


def _free_camera() -> Any:
    camera = mujoco_api.MjvCamera()
    mujoco_api.mjv_defaultCamera(camera)
    camera.type = mujoco_api.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 135.0
    camera.elevation = -18.0
    camera.distance = 0.65
    camera.lookat[:] = (0.0, 0.0, 0.10)
    return camera


def _copy_torch_state_to_native(
    bundle: MicroDuckModelBundle,
    env: NominalMicroDuckEnv,
    data: Any,
) -> None:
    if env.data is None:
        raise RuntimeError("Environment must be reset before rendering")
    state = env.data
    data.qpos[:] = state.qpos.detach().cpu().numpy()
    data.qvel[:] = state.qvel.detach().cpu().numpy()
    data.ctrl[:] = state.ctrl.detach().cpu().numpy()
    data.time = float(state.time)
    mujoco_api.mj_forward(bundle.native_model, data)


def _render_native(
    bundle: MicroDuckModelBundle,
    env: NominalMicroDuckEnv,
    renderer: mujoco.Renderer,
    render_data: Any,
    camera: Any,
) -> np.ndarray:
    _copy_torch_state_to_native(bundle, env, render_data)
    if not isinstance(camera, str):
        camera.lookat[:2] = render_data.qpos[:2]
    renderer.update_scene(render_data, camera=camera)
    return np.asarray(renderer.render(), dtype=np.uint8)


def _render_torch(
    bundle: MicroDuckModelBundle,
    data: Any,
    *,
    camera_id: int,
    width: int,
    height: int,
    precomp: dict[str, Any],
    ray_chunk_size: int,
) -> np.ndarray:
    rgb, _depth, _segmentation = mujoco_torch.render(
        bundle.torch_model,
        data,
        camera_id=camera_id,
        width=width,
        height=height,
        precomp=precomp,
        ray_chunk_size=ray_chunk_size,
    )
    return rgb.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()


def render_policy_rollout(
    artifact: PolicyArtifact,
    *,
    output: str | Path,
    gif_output: str | Path | None = None,
    steps: int = 250,
    seconds: float | None = None,
    fps: int = 25,
    render_every: int = 2,
    width: int = 320,
    height: int = 240,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    render_backend: RenderBackend = "mujoco",
    camera: CameraName = "free",
    vx: float = 0.15,
    vy: float = 0.0,
    vtheta: float = 0.0,
    fixed_iterations: bool = True,
    solver_iterations: int = 4,
    line_search_iterations: int = 4,
    disable_contacts: bool = False,
    disable_mesh_mesh_contacts: bool = False,
    gif_fps: int = 12,
    gif_width: int = 720,
    gif_colors: int = 48,
    ray_chunk_size: int = 256,
    progress: bool = True,
) -> dict[str, Any]:
    """Roll out the HF policy in the Torch env and save MP4/GIF artifacts.

    The default ``mujoco`` renderer uses native MuJoCo only to rasterize the
    current Torch state, retaining the complete CAD visual model while the
    dynamics and observations come from ``mujoco-torch``. The optional
    ``mujoco-torch`` backend exercises its pure-Torch ray renderer and uses the
    robot's attached ``head_camera``.
    """

    if render_every < 1:
        raise ValueError("render_every must be positive")
    if seconds is not None and seconds <= 0:
        raise ValueError("seconds must be positive")
    if render_backend not in ("mujoco", "mujoco-torch"):
        raise ValueError(f"Unknown render backend {render_backend!r}")
    if render_backend == "mujoco-torch" and camera == "free":
        raise ValueError("The mujoco-torch renderer requires --camera head_camera")

    bundle = load_microduck_model(
        device=device,
        dtype=dtype,
        fixed_iterations=fixed_iterations,
        solver_iterations=solver_iterations,
        line_search_iterations=line_search_iterations,
        disable_contacts=disable_contacts,
        disable_mesh_mesh_contacts=disable_mesh_mesh_contacts,
    )
    environment = NominalMicroDuckEnv(
        bundle,
        command=command_vector(vx=vx, vy=vy, vtheta=vtheta, device=bundle.device),
    )
    if seconds is not None:
        control_timestep = bundle.timestep * environment.decimation
        steps = max(1, int(round(seconds / control_timestep)))
    if steps < 1:
        raise ValueError("steps must be positive")
    policy = OnnxPolicy(artifact)
    observation = environment.reset()
    if progress:
        print(
            f"Rendering {steps} Torch-environment steps with {render_backend} -> {output}",
            file=sys.stderr,
            flush=True,
        )

    renderer: mujoco.Renderer | None = None
    render_data: Any | None = None
    render_camera: Any | None = None
    render_precomp: dict[str, Any] | None = None
    camera_id = 0
    if render_backend == "mujoco":
        renderer = mujoco.Renderer(bundle.native_model, height=height, width=width)
        render_data = mujoco_api.MjData(bundle.native_model)
        render_camera = _free_camera() if camera == "free" else camera
    else:
        camera_id = int(bundle.native_model.camera("head_camera").id)
        render_precomp = mujoco_torch.precompute_render_data(bundle.torch_model)

    writer = VideoWriter(output, width=width, height=height, fps=fps)
    started = time.monotonic()
    rendered_frames = 0
    completed_steps = 0
    finite = True
    succeeded = False
    start_position = np.asarray(bundle.native_model.key("STAND").qpos[:2], dtype=np.float64)
    final_position = start_position.copy()
    try:
        for step in range(steps):
            if step % render_every == 0:
                if render_backend == "mujoco":
                    assert (
                        renderer is not None
                        and render_data is not None
                        and render_camera is not None
                    )
                    frame = _render_native(
                        bundle,
                        environment,
                        renderer,
                        render_data,
                        render_camera,
                    )
                else:
                    assert render_precomp is not None and environment.data is not None
                    frame = _render_torch(
                        bundle,
                        environment.data,
                        camera_id=camera_id,
                        width=width,
                        height=height,
                        precomp=render_precomp,
                        ray_chunk_size=ray_chunk_size,
                    )
                writer.write(frame)
                rendered_frames += 1
                if progress and (rendered_frames == 1 or rendered_frames % 10 == 0):
                    print(
                        f"  frame {rendered_frames} (step {step + 1}/{steps})",
                        file=sys.stderr,
                        flush=True,
                    )

            action = policy(observation)
            if action.shape != (bundle.action_size,) or not torch.isfinite(action).all():
                finite = False
                break
            result = environment.step(action.to(device=bundle.device, dtype=bundle.dtype))
            observation = result.observation
            completed_steps += 1
            assert environment.data is not None
            final_position = environment.data.qpos[:2].detach().cpu().numpy()
            finite = finite and result.info["finite"]
            if result.terminated:
                break
        succeeded = True
    finally:
        if renderer is not None:
            renderer.close()
        if succeeded:
            writer.close()
        else:
            writer.abort()

    if rendered_frames == 0:
        writer.abort()
        raise RuntimeError("Policy rollout produced no renderable frames")
    gif_path = None
    if gif_output is not None:
        gif_path = convert_video_to_gif(
            output,
            gif_output,
            fps=gif_fps,
            width=gif_width,
            colors=gif_colors,
        )

    return {
        "status": "pass" if finite else "fail",
        "policy": artifact.policy_name,
        "policy_revision": artifact.revision,
        "policy_sha256": artifact.sha256,
        "steps_requested": steps,
        "steps_completed": completed_steps,
        "rendered_frames": rendered_frames,
        "render_backend": render_backend,
        "camera": camera,
        "contacts_enabled": bundle.contacts_enabled,
        "output": str(Path(output)),
        "gif_output": str(gif_path) if gif_path is not None else None,
        "distance": float(np.linalg.norm(final_position - start_position)),
        "wall_seconds": time.monotonic() - started,
    }
