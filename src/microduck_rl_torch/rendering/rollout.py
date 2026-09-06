"""Render the Torch MicroDuck environment driven by an ONNX golden policy."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from microduck_rl_torch.envs import ManagerBasedTaskEnv, SceneBuilder
from microduck_rl_torch.envs.model import load_model_bundle
from microduck_rl_torch.envs.observations import command_vector
from microduck_rl_torch.envs.task_config import TaskEnvCfg
from microduck_rl_torch.policies.huggingface import OnnxPolicy, PolicyArtifact
from microduck_rl_torch.tasks import make_microduck_velocity_env_cfg

from .config import CameraConfig, RenderBackend, RenderConfig
from .video import VideoWriter, convert_video_to_gif

ActuatorMode = Literal["bam", "xml"]
CameraName = Literal["free", "head_camera"]


def _camera_config(camera: CameraName) -> CameraConfig:
    if camera == "free":
        return CameraConfig()
    if camera == "head_camera":
        return CameraConfig(name=camera, follow_root=False)
    raise ValueError(f"Unknown camera {camera!r}")


def render_policy_rollout(
    artifact: PolicyArtifact | None = None,
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
    dtype: torch.dtype = torch.float32,
    actuator_mode: ActuatorMode = "xml",
    render_backend: RenderBackend = "mujoco",
    camera: CameraName = "free",
    vx: float = 0.3,
    vy: float = 0.0,
    vtheta: float = 0.0,
    fixed_iterations: bool = True,
    solver_iterations: int = 4,
    line_search_iterations: int = 4,
    disable_mesh_mesh_contacts: bool = False,
    gif_fps: int = 25,
    gif_width: int = 720,
    gif_colors: int = 48,
    ray_chunk_size: int = 256,
    progress: bool = True,
    task_cfg: TaskEnvCfg | None = None,
    policy: Callable[[torch.Tensor], torch.Tensor] | None = None,
    command: torch.Tensor | None = None,
) -> dict[str, object]:
    """Roll out the HF policy in the Torch env and save MP4/GIF artifacts.

    Rendering is an environment capability. The rollout owns only policy
    stepping and recording; the environment owns the selected renderer and
    its native/Torch state bridge. Frames are captured after control steps,
    so ``render_every`` and ``fps`` must describe the same simulated cadence.
    Floor contacts are always enabled for rendering; only detailed mesh-to-mesh
    contacts may be disabled for performance.
    """

    if render_every < 1:
        raise ValueError("render_every must be positive")
    if fps < 1:
        raise ValueError("fps must be positive")
    if seconds is not None and seconds <= 0:
        raise ValueError("seconds must be positive")
    if actuator_mode not in ("bam", "xml"):
        raise ValueError(f"Unknown actuator mode {actuator_mode!r}")
    camera_config = _camera_config(camera)

    # The default remains the checked-in golden walk, but the environment and
    # policy are injectable so render validation is a generic task capability.
    task_cfg = (task_cfg or make_microduck_velocity_env_cfg()).clone()
    task_cfg.actions.actuator_mode = actuator_mode
    if task_cfg.scene.num_envs != 1:
        raise ValueError("render_policy_rollout renders exactly one environment at a time")
    scene_build = SceneBuilder().build(task_cfg.scene)
    primary_name = (
        "robot" if "robot" in task_cfg.scene.entities else next(iter(task_cfg.scene.entities))
    )
    bundle = load_model_bundle(
        xml_path=scene_build.xml_path,
        entity_cfg=task_cfg.scene.entities[primary_name],
        entities=task_cfg.scene.entities,
        device=device,
        dtype=dtype,
        fixed_iterations=fixed_iterations,
        solver_iterations=solver_iterations,
        line_search_iterations=line_search_iterations,
        disable_mesh_mesh_contacts=disable_mesh_mesh_contacts,
        actuator_mode=actuator_mode,
        collision_policy=task_cfg.collision_policy,
    )
    if not bundle.contacts_enabled:
        raise RuntimeError("Rendering requires floor contacts; contact disabling is not supported")
    environment = ManagerBasedTaskEnv(
        task_cfg,
        bundle=bundle,
        command=(
            command
            if command is not None
            else command_vector(vx=vx, vy=vy, vtheta=vtheta, device=bundle.device)
            if task_cfg.commands.terms
            else None
        ),
        domain_randomization=False,
        render_mode="rgb_array",
        render_config=RenderConfig(
            backend=render_backend,
            width=width,
            height=height,
            camera=camera_config,
            ray_chunk_size=ray_chunk_size,
        ),
    )
    if seconds is not None:
        control_timestep = bundle.timestep * environment.decimation
        steps = max(1, int(round(seconds / control_timestep)))
    if steps < 1:
        raise ValueError("steps must be positive")

    control_timestep = bundle.timestep * environment.decimation
    effective_fps = 1.0 / (control_timestep * render_every)
    if not np.isclose(effective_fps, fps, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"fps={fps} does not match the simulated cadence: render_every={render_every} "
            f"at control_dt={control_timestep:g}s produces {effective_fps:g} FPS"
        )

    if policy is None:
        if artifact is None:
            raise ValueError("Provide either a PolicyArtifact or a callable policy")
        policy = OnnxPolicy(artifact)
    observation = environment.reset()
    if progress:
        print(
            f"Rendering {steps} Torch-environment steps with {render_backend} "
            f"({fps} FPS, post-step frames) -> {output}",
            file=sys.stderr,
            flush=True,
        )

    writer = VideoWriter(output, width=width, height=height, fps=fps)
    started = time.monotonic()
    rendered_frames = 0
    completed_steps = 0
    finite = True
    succeeded = False
    root_view = bundle.entity(bundle.entity_cfg.name)
    root_indices = root_view.free_qpos_indices[:2]
    start_position = (
        bundle.default_qpos.index_select(0, root_indices)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
    )
    final_position = start_position.copy()
    try:
        for _step in range(steps):
            action = torch.as_tensor(policy(observation), device=bundle.device, dtype=bundle.dtype)
            if action.shape != (bundle.action_size,) or not torch.isfinite(action).all():
                finite = False
                break
            result = environment.step(action.to(device=bundle.device, dtype=bundle.dtype))
            observation = result.observation
            completed_steps += 1
            assert environment.data is not None
            final_position = (
                environment.data.qpos.index_select(-1, root_indices).detach().cpu().numpy()
            )
            finite = finite and bool(result.info["finite"])

            if completed_steps % render_every == 0 or result.terminated:
                frame = environment.render()
                if frame is None:
                    raise RuntimeError("RGB rendering was unexpectedly disabled")
                writer.write(frame)
                rendered_frames += 1
                if progress and (rendered_frames == 1 or rendered_frames % 10 == 0):
                    print(
                        f"  frame {rendered_frames} (step {completed_steps}/{steps})",
                        file=sys.stderr,
                        flush=True,
                    )

            if result.terminated:
                break
        succeeded = True
    finally:
        try:
            if succeeded:
                writer.close()
            else:
                writer.abort()
        finally:
            environment.close()

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
        "policy": artifact.policy_name
        if artifact is not None
        else getattr(policy, "__name__", "callable"),
        "policy_revision": artifact.revision if artifact is not None else None,
        "policy_sha256": artifact.sha256 if artifact is not None else None,
        "steps_requested": steps,
        "steps_completed": completed_steps,
        "rendered_frames": rendered_frames,
        "render_backend": render_backend,
        "camera": camera,
        "actuator_mode": actuator_mode,
        "contacts_enabled": bundle.contacts_enabled,
        "render_every": render_every,
        "render_fps": fps,
        "frame_semantics": "post_step",
        "output": str(Path(output)),
        "gif_output": str(gif_path) if gif_path is not None else None,
        "distance": float(np.linalg.norm(final_position - start_position)),
        "wall_seconds": time.monotonic() - started,
    }
