"""Installed command-line entry point for golden-policy rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microduck_rl_torch.policies.huggingface import fetch_policy
from microduck_rl_torch_verification.validate import _artifact_from_local

from .rollout import render_policy_rollout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="alpha_walking")
    parser.add_argument("--policy-dir", type=Path, default=Path("artifacts/hf"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/render/microduck-alpha-walking.mp4"),
    )
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument(
        "--seconds",
        type=float,
        help="Requested simulated rollout duration; takes precedence over --steps",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--actuator-mode",
        choices=("xml", "bam"),
        default="xml",
        help=(
            "Actuator model for the visual rollout (xml reproduces the original "
            "golden video; bam is the upstream parity path)"
        ),
    )
    parser.add_argument(
        "--render-backend",
        choices=("mujoco", "mujoco-torch"),
        default="mujoco",
    )
    parser.add_argument("--camera", choices=("free", "head_camera"), default="free")
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vtheta", type=float, default=0.0)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--line-search-iterations", type=int, default=4)
    parser.add_argument(
        "--fixed-iterations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--contacts", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument(
        "--mesh-mesh-contacts",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Enable or skip detailed mesh-to-mesh contacts; plane contacts remain independent",
    )
    parser.add_argument("--gif-fps", type=int, default=25)
    parser.add_argument("--gif-width", type=int, default=720)
    parser.add_argument("--gif-colors", type=int, default=48)
    parser.add_argument(
        "--ray-chunk-size",
        type=int,
        default=256,
        help="Pixels processed per chunk by the pure Torch mesh ray renderer",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress rollout progress messages on stderr",
    )
    args = parser.parse_args(argv)

    artifact = (
        fetch_policy(args.policy, output_dir=args.policy_dir)
        if args.download
        else _artifact_from_local(args.policy_dir, args.policy)
    )
    result = render_policy_rollout(
        artifact,
        output=args.output,
        gif_output=args.gif,
        steps=args.steps,
        seconds=args.seconds,
        fps=args.fps,
        render_every=args.render_every,
        width=args.width,
        height=args.height,
        device=args.device,
        actuator_mode=args.actuator_mode,
        render_backend=args.render_backend,
        camera=args.camera,
        vx=args.vx,
        vy=args.vy,
        vtheta=args.vtheta,
        fixed_iterations=args.fixed_iterations,
        solver_iterations=args.solver_iterations,
        line_search_iterations=args.line_search_iterations,
        disable_contacts=args.contacts == "disabled",
        disable_mesh_mesh_contacts=args.mesh_mesh_contacts == "disabled",
        gif_fps=args.gif_fps,
        gif_width=args.gif_width,
        gif_colors=args.gif_colors,
        ray_chunk_size=args.ray_chunk_size,
        progress=not args.quiet,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
