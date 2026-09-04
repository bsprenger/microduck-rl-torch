"""Curriculum manager; a no-op manager is useful for flat velocity parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..task_config import TermCollection


@dataclass
class CurriculumManager:
    terms: TermCollection

    def step(self, env: Any) -> None:
        for term in self.terms.values():
            if term.enabled and term.func is not None:
                term.func(env)
