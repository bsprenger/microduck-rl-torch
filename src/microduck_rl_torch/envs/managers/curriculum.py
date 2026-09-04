"""Curriculum manager; a no-op manager is useful for flat velocity parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..task_config import TermCollection
from .base import call_event_term, reset_term, resolve_term


@dataclass
class CurriculumManager:
    terms: TermCollection

    def __post_init__(self) -> None:
        self._resolved_terms: dict[str, Any] = {}
        self._last_extras: dict[str, Any] = {}

    def reset(self, env: Any, env_ids: torch.Tensor | None = None) -> None:
        for name, term in self.terms.items():
            if term.enabled:
                function = self._resolved_terms.get(name)
                if function is None:
                    function = resolve_term(term, env)
                    self._resolved_terms[name] = function
                reset_term(function, env_ids)

    @property
    def last_extras(self) -> dict[str, Any]:
        return self._last_extras

    def compute(self, env: Any, env_ids: torch.Tensor | slice | None = None) -> dict[str, Any]:
        """Evaluate active curricula for the selected environments."""

        callback_ids: torch.Tensor | slice = (
            torch.arange(env.num_envs, dtype=torch.long, device=env.bundle.device)
            if env_ids is None
            else env_ids
        )
        extras: dict[str, Any] = {}
        for name, term in self.terms.items():
            if not term.enabled:
                continue
            if name not in self._resolved_terms:
                self._resolved_terms[name] = resolve_term(term, env)
            function = self._resolved_terms[name]
            if function is None:
                raise RuntimeError(f"Curriculum term {name!r} has no function")
            value = call_event_term(function, env, callback_ids, term.params)
            if value is not None:
                extras[name] = value
        self._last_extras = extras
        if env.state is not None:
            env.state.manager_data["curriculum"] = extras
        return extras
