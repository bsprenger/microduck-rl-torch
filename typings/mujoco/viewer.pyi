"""Checker-only surface for MuJoCo's optional interactive viewer module."""

from typing import Any

from . import MjData, MjModel


class PassiveViewer:
    def __enter__(self) -> PassiveViewer: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None: ...

    def is_running(self) -> bool: ...
    def sync(self) -> None: ...


def launch_passive(model: MjModel, data: MjData) -> PassiveViewer: ...
