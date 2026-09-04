"""Explicit reset/pre-physics/post-physics event lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..task_config import TermCollection
from .base import TaskRuntimeContext

EventStage = Literal["reset", "pre_physics", "post_physics"]


@dataclass
class EventManager:
    terms: TermCollection

    def apply(self, env: Any, stage: EventStage) -> None:
        for _name, term in self.terms.items():
            if not term.enabled or term.func is None:
                continue
            configured_stage = term.params.get("stage", "pre_physics")
            if configured_stage == stage:
                term.func(TaskRuntimeContext(env))
