#!/usr/bin/env python3
"""Run a script or module through macOS's MuJoCo-compatible interpreter."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _find_mjpython() -> str:
    candidate = shutil.which("mjpython")
    if candidate:
        return candidate
    sibling = Path(sys.executable).with_name("mjpython")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    raise RuntimeError("mjpython is required on macOS but was not found in the active environment")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    raw_args = sys.argv[1:]
    forwarded_args: list[str] = []
    if "--" in raw_args:
        separator = raw_args.index("--")
        forwarded_args = raw_args[separator + 1 :]
        raw_args = raw_args[:separator]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", "-m")
    parser.add_argument("target", nargs="?")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args(raw_args), forwarded_args


def main() -> None:
    args, forwarded_args = _parse_args()
    if not args.module and not args.target:
        raise SystemExit("Specify a script path or --module MODULE")
    if platform.system() != "Darwin":
        command = [sys.executable]
    else:
        command = [_find_mjpython()]
        base_lib = Path(sys.base_prefix) / "lib"
        if base_lib.is_dir():
            old = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{base_lib}:{old}" if old else str(base_lib)
    if args.module:
        command.extend(["-m", args.module])
        command.extend(forwarded_args)
    else:
        command.append(args.target)
        command.extend(args.script_args)
    raise SystemExit(subprocess.run(command, check=False, env=os.environ).returncode)  # noqa: S603


if __name__ == "__main__":
    main()
