#!/usr/bin/env python3
"""Generate a compact native-MuJoCo golden trajectory fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from microduck_rl_torch.envs.model import default_scene_path, load_model_bundle
from microduck_rl_torch_verification.trajectory import (
    generate_action_tape,
    generate_native_trajectory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/microduck_bam_golden.npz"),
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--solver-iterations", type=int, default=30)
    parser.add_argument("--line-search-iterations", type=int, default=30)
    parser.add_argument("--xml", type=Path, default=default_scene_path())
    parser.add_argument("--vx", type=float, default=0.15)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vtheta", type=float, default=0.0)
    parser.add_argument(
        "--contacts",
        choices=("enabled", "disabled"),
        default="disabled",
        help="Contact mode for the exact native golden fixture",
    )
    parser.add_argument("--action-delay-lag", type=int, default=0)
    args = parser.parse_args()
    bundle = load_model_bundle(
        args.xml,
        fixed_iterations=True,
        solver_iterations=args.solver_iterations,
        line_search_iterations=args.line_search_iterations,
        disable_contacts=args.contacts == "disabled",
        disable_mesh_mesh_contacts=True,
    )
    actions = generate_action_tape(args.steps, seed=args.seed)
    command = np.zeros(13, dtype=np.float64)
    command[:3] = (args.vx, args.vy, args.vtheta)
    trajectory = generate_native_trajectory(
        bundle,
        actions,
        command=command,
        metadata={
            "action_tape": "numpy.default_rng.normal(scale=0.05)",
            "action_seed": args.seed,
            "solver_iterations": args.solver_iterations,
            "line_search_iterations": args.line_search_iterations,
            "contacts": args.contacts,
        },
        disable_contacts=args.contacts == "disabled",
        action_delay_lag=args.action_delay_lag,
    )
    trajectory.save(args.output)
    print(f"Wrote {args.output} ({args.steps} transitions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
