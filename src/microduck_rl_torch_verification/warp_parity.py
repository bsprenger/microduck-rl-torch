"""Compare a Torch rollout with a rollout from the upstream MuJoCo-Warp task."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PARITY_CHECKPOINTS = (1, 5, 10, 25, 50, 100, 250, 500)
PARITY_FIELDS = ("actions", "observations", "qpos", "qvel")


@dataclass(frozen=True)
class WarpParityTrace:
    """One control-rate trajectory from either simulator backend.

    ``observations[i]`` is the actor observation used to produce
    ``actions[i]``. ``qpos[i]`` and ``qvel[i]`` are the simulator state after
    applying that action for one policy step (including physics decimation).
    """

    metadata: dict[str, Any]
    observations: np.ndarray
    actions: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])

    @classmethod
    def load(cls, path: Path) -> WarpParityTrace:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            return cls(
                metadata=metadata,
                observations=np.asarray(data["observations"]),
                actions=np.asarray(data["actions"]),
                qpos=np.asarray(data["qpos"]),
                qvel=np.asarray(data["qvel"]),
                terminated=np.asarray(data["terminated"], dtype=bool),
                truncated=np.asarray(data["truncated"], dtype=bool),
            )

    def validate(self) -> None:
        lengths = {
            getattr(self, field).shape[0] for field in (*PARITY_FIELDS, "terminated", "truncated")
        }
        if len(lengths) != 1:
            raise ValueError(f"Trajectory fields have inconsistent lengths: {sorted(lengths)}")
        if self.observations.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("Observations and actions must be rank-2 arrays")
        if self.qpos.ndim != 2 or self.qvel.ndim != 2:
            raise ValueError("qpos and qvel must be rank-2 arrays")
        if not all(np.isfinite(getattr(self, field)).all() for field in PARITY_FIELDS):
            raise ValueError("Trajectory contains non-finite parity fields")


@dataclass(frozen=True)
class WarpParityMetric:
    """Cumulative maximum absolute differences through one checkpoint."""

    step: int
    action_max_abs: float
    obs_max_abs: float
    qpos_max_abs: float
    qvel_max_abs: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_shapes(torch_trace: WarpParityTrace, warp_trace: WarpParityTrace) -> None:
    for field in PARITY_FIELDS:
        torch_shape = getattr(torch_trace, field).shape
        warp_shape = getattr(warp_trace, field).shape
        if torch_shape != warp_shape:
            raise ValueError(f"{field} shape mismatch: Torch={torch_shape}, Warp={warp_shape}")


def _max_abs_prefix(left: np.ndarray, right: np.ndarray, end: int) -> float:
    return float(np.max(np.abs(left[:end].astype(np.float64) - right[:end].astype(np.float64))))


def interval_metrics(
    torch_trace: WarpParityTrace,
    warp_trace: WarpParityTrace,
    *,
    checkpoints: Iterable[int] | None = None,
) -> list[WarpParityMetric]:
    """Compute cumulative max errors at the requested control-step checkpoints."""

    torch_trace.validate()
    warp_trace.validate()
    if torch_trace.steps != warp_trace.steps:
        raise ValueError(f"Step count mismatch: Torch={torch_trace.steps}, Warp={warp_trace.steps}")
    _check_shapes(torch_trace, warp_trace)
    requested = PARITY_CHECKPOINTS if checkpoints is None else tuple(checkpoints)
    if not requested:
        raise ValueError("At least one parity checkpoint is required")
    if any(step < 1 or step > torch_trace.steps for step in requested):
        raise ValueError(f"Checkpoints must be in [1, {torch_trace.steps}], got {requested}")
    return [
        WarpParityMetric(
            step=int(step),
            action_max_abs=_max_abs_prefix(torch_trace.actions, warp_trace.actions, step),
            obs_max_abs=_max_abs_prefix(torch_trace.observations, warp_trace.observations, step),
            qpos_max_abs=_max_abs_prefix(torch_trace.qpos, warp_trace.qpos, step),
            qvel_max_abs=_max_abs_prefix(torch_trace.qvel, warp_trace.qvel, step),
        )
        for step in requested
    ]


def format_parity_table(
    metrics: Iterable[WarpParityMetric],
    *,
    passed: bool | None = None,
    label: str = "Torch vs upstream Warp",
) -> str:
    """Format cumulative parity metrics as a compact Markdown table."""

    rows = list(metrics)
    if not rows:
        return f"{label} — no parity rows"
    status = "PASS" if passed is True else "FAIL" if passed is False else "DIAGNOSTIC"
    lines = [
        f"{label} — cumulative maximum absolute difference ({status})",
        "| step | action | observation | qpos | qvel |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row.step} | {row.action_max_abs:.6g} | {row.obs_max_abs:.6g} | "
        f"{row.qpos_max_abs:.6g} | {row.qvel_max_abs:.6g} |"
        for row in rows
    )
    return "\n".join(lines)


def write_parity_report(
    path: Path,
    *,
    metadata: dict[str, Any],
    metrics: Iterable[WarpParityMetric],
    passed: bool | None,
    thresholds: dict[str, float],
    error: str | None = None,
) -> Path:
    """Write a human-readable Markdown report and a machine-readable sidecar."""

    rows = list(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if passed is True else "FAIL" if passed is False else "DIAGNOSTIC"
    lines = [
        "# MicroDuck Torch/Warp parity",
        "",
        f"Status: **{status}**",
        "",
        "This is a deterministic paired rollout of the local Torch task and the "
        "upstream `mjlab` MuJoCo-Warp task. Each row is cumulative through the "
        "reported control step.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Thresholds",
        "",
        "```json",
        json.dumps(thresholds, indent=2, sort_keys=True),
        "```",
        "",
    ]
    if error:
        lines.extend(["## Error", "", f"`{error}`", ""])
    if rows:
        lines.extend([format_parity_table(rows, passed=passed), ""])
    path.write_text("\n".join(lines))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "status": status.lower(),
                "metadata": metadata,
                "thresholds": thresholds,
                "metrics": [row.as_dict() for row in rows],
                "error": error,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    return path


def parity_passed(
    metrics: Iterable[WarpParityMetric],
    *,
    thresholds: dict[str, float],
    fail_on: str = "state",
) -> bool:
    """Evaluate the final cumulative row against selected parity thresholds."""

    rows = list(metrics)
    if not rows:
        return False
    final = rows[-1]
    checks = {
        "action": final.action_max_abs <= thresholds["action"],
        "observation": final.obs_max_abs <= thresholds["observation"],
        "qpos": final.qpos_max_abs <= thresholds["qpos"],
        "qvel": final.qvel_max_abs <= thresholds["qvel"],
    }
    if fail_on == "none":
        return True
    if fail_on == "action":
        return checks["action"]
    if fail_on == "state":
        return checks["qpos"] and checks["qvel"]
    if fail_on == "all":
        return all(checks.values())
    raise ValueError(f"Unknown fail_on mode {fail_on!r}")
