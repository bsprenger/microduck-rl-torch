import numpy as np
import pytest

from microduck_rl_torch_verification.warp_parity import (
    WarpParityTrace,
    format_parity_table,
    interval_metrics,
    parity_passed,
    write_parity_report,
)


def _trace(offset: float = 0.0, steps: int = 500) -> WarpParityTrace:
    values = np.zeros((steps, 2), dtype=np.float64)
    return WarpParityTrace(
        metadata={"backend": "test"},
        observations=values + offset,
        actions=values + offset,
        qpos=values + offset,
        qvel=values + offset,
        terminated=np.zeros(steps, dtype=bool),
        truncated=np.zeros(steps, dtype=bool),
    )


def test_interval_metrics_are_cumulative_and_use_requested_checkpoints() -> None:
    left = _trace()
    right = _trace()
    right.observations[2, 0] = 0.25
    right.qpos[4, 1] = -0.5

    metrics = interval_metrics(left, right, checkpoints=(1, 3, 5))

    assert [metric.step for metric in metrics] == [1, 3, 5]
    assert metrics[0].obs_max_abs == 0.0
    assert metrics[1].obs_max_abs == 0.25
    assert metrics[1].qpos_max_abs == 0.0
    assert metrics[2].qpos_max_abs == 0.5


def test_default_table_has_the_500_step_checkpoints() -> None:
    metrics = interval_metrics(_trace(), _trace())

    table = format_parity_table(metrics, passed=True)

    assert "cumulative maximum absolute difference (PASS)" in table
    assert "| step | action | observation | qpos | qvel |" in table
    assert "| 500 | 0 | 0 | 0 | 0 |" in table


def test_parity_threshold_modes_and_report_sidecar(tmp_path) -> None:
    left = _trace()
    right = _trace(offset=0.01)
    metrics = interval_metrics(left, right, checkpoints=(1, 500))
    thresholds = {"action": 0.02, "observation": 0.02, "qpos": 0.02, "qvel": 0.02}
    report = write_parity_report(
        tmp_path / "parity.md",
        metadata={"steps": 500},
        metrics=metrics,
        passed=parity_passed(metrics, thresholds=thresholds, fail_on="all"),
        thresholds=thresholds,
    )

    assert parity_passed(metrics, thresholds=thresholds, fail_on="all")
    assert report.is_file()
    assert report.with_suffix(".json").is_file()
    assert "| 500 | 0.01 | 0.01 | 0.01 | 0.01 |" in report.read_text()


def test_interval_metrics_reject_shape_mismatch() -> None:
    left = _trace(steps=2)
    right = _trace(steps=3)

    with pytest.raises(ValueError, match="Step count mismatch"):
        interval_metrics(left, right, checkpoints=(1,))
