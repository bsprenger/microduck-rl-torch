#!/usr/bin/env python3
"""Validate a policy already downloaded into an artifact directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microduck_rl_torch.policies.huggingface import validate_policy_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", type=Path, default=Path("artifacts/hf"))
    parser.add_argument("--policy", default="alpha_walking")
    args = parser.parse_args(argv)
    filename = args.policy if args.policy.endswith(".onnx") else f"{args.policy}.onnx"
    metadata = validate_policy_artifact(
        args.policy_dir / filename,
        args.policy_dir / "manifest.json",
        filename=filename,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
