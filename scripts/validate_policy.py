#!/usr/bin/env python3
"""Validate a policy already downloaded into an artifact directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microduck_rl_torch.policies.huggingface import (
    resolve_policy_filename,
    validate_policy_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", type=Path, default=Path("artifacts/hf"))
    parser.add_argument("--policy", default="alpha_walking")
    args = parser.parse_args(argv)
    manifest_path = args.policy_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    filename = resolve_policy_filename(manifest, args.policy)
    metadata = validate_policy_artifact(
        args.policy_dir / filename,
        manifest_path,
        filename=filename,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
