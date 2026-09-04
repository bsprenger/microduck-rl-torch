from pathlib import Path

import pytest


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]
