"""Keep Hatchling's source archive free of repository ignore metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Remove Hatchling's automatically forced root ignore file from sdists."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        del version
        if self.target_name == "sdist":
            build_data.get("force_include", {}).pop(str(Path(self.root, ".gitignore")), None)
