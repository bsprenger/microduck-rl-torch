"""Validate that the built archives contain only publishable project files."""

from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path

ARCHIVE_ROOT = Path("dist")
EXPECTED_SDIST_FILES = {
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
}
FORBIDDEN_PATH = re.compile(
    r"(^|/)(?:\.git(?:/|ignore|modules|attributes)|\.hg(?:/|ignore|tags)?|\.svn(?:/|ignore)?|"
    r"CVS|AGENTS\.md|\.agents|\.codex|\.cache|"
    r"\.pytest_cache|\.ruff_cache|\.DS_Store|__pycache__|.*\.py[cod]|.*\.log|.*\.LOG|"
    r".*\.ses|\.coverage)(/|$)"
)


def _archive_members(archive: Path) -> set[str]:
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as handle:
            return {member.name for member in handle.getmembers()}
    with zipfile.ZipFile(archive) as handle:
        return set(handle.namelist())


def main() -> None:
    sdists = sorted(ARCHIVE_ROOT.glob("*.tar.gz"), key=lambda path: path.stat().st_mtime)
    wheels = sorted(ARCHIVE_ROOT.glob("*.whl"), key=lambda path: path.stat().st_mtime)
    if not sdists or not wheels:
        raise SystemExit("uv build did not produce both an sdist and a wheel")

    sdist_members = _archive_members(sdists[-1])
    wheel_members = _archive_members(wheels[-1])
    forbidden = sorted(
        member for member in sdist_members | wheel_members if FORBIDDEN_PATH.search(member)
    )
    if forbidden:
        raise SystemExit(f"published archives contain forbidden files: {forbidden}")

    sdist_names = {Path(member).name for member in sdist_members}
    missing = sorted(EXPECTED_SDIST_FILES - sdist_names)
    if missing:
        raise SystemExit(f"sdist is missing expected project files: {missing}")

    print(f"Checked {sdists[-1].name} ({len(sdist_members)} files)")
    print(f"Checked {wheels[-1].name} ({len(wheel_members)} files)")


if __name__ == "__main__":
    main()
