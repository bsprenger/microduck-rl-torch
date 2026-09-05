"""Curriculum manager; a no-op manager is useful for flat velocity parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..task_config import TermCollection
from .base import call_term, reset_term, resolve_term


@dataclass
class CurriculumManager:
    terms: TermCollection

    def __post_init__(self) -> None:
        self._resolved_terms: dict[str, Any] = {}

    def reset(self, env: Any, env_ids: torch.Tensor | None = None) -> None:
        for name, term in self.terms.items():
            if term.enabled:
                function = self._resolved_terms.get(name)
                if function is None:
                    function = resolve_term(term, env)
                    self._resolved_terms[name] = function
                reset_term(function, env_ids)

    def step(self, env: Any) -> None:
        for name, term in self.terms.items():
            if not term.enabled:
                continue
            if name not in self._resolved_terms:
                self._resolved_terms[name] = resolve_term(term, env)
            function = self._resolved_terms[name]
            if function is None:
                raise RuntimeError(f"Curriculum term {name!r} has no function")
            call_term(function, env, term.params)
