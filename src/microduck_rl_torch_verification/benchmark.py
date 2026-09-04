"""Shared timing and summary helpers for backend benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def summarize_timings(seconds: Sequence[float], *, steps: int) -> dict[str, Any]:
    """Return robust timing and throughput statistics for repeated samples.

    The median is the headline value because a single background scheduler or
    compilation event should not dominate the result. The p05/p95 throughput
    values are derived from the inverse timing percentiles, so the range is
    conservative: p05 is the slower end and p95 is the faster end.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    values = np.asarray(seconds, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("seconds must be a non-empty finite one-dimensional sequence")
    if (values <= 0).any():
        raise ValueError("timings must be positive")

    p05_seconds, median_seconds, p95_seconds = np.percentile(values, [5, 50, 95])
    return {
        "repeat_count": int(values.size),
        "seconds": values.tolist(),
        "mean_seconds": float(values.mean()),
        "median_seconds": float(median_seconds),
        "p05_seconds": float(p05_seconds),
        "p95_seconds": float(p95_seconds),
        "mean_steps_per_second": float(steps / values.mean()),
        "median_steps_per_second": float(steps / median_seconds),
        "p05_steps_per_second": float(steps / p95_seconds),
        "p95_steps_per_second": float(steps / p05_seconds),
    }


def measure_repeated(
    run_sample: Callable[[], None],
    *,
    prepare_sample: Callable[[], None] | None = None,
    synchronize: Callable[[], None] | None = None,
    repeats: int,
) -> dict[str, Any]:
    """Measure repeated samples while keeping setup and synchronization fair.

    ``prepare_sample`` is called before every timed sample. It should reset the
    simulator to the same state and perform any untimed warmup. Synchronization
    runs immediately before and after each timed region, which is required for
    asynchronous accelerators such as CUDA and MPS.
    """

    if repeats < 1:
        raise ValueError("repeats must be positive")
    sync = synchronize or (lambda: None)
    timings: list[float] = []
    import time

    for _ in range(repeats):
        if prepare_sample is not None:
            prepare_sample()
        sync()
        start = time.perf_counter_ns()
        run_sample()
        sync()
        timings.append((time.perf_counter_ns() - start) / 1_000_000_000.0)
    return {"seconds": timings}
