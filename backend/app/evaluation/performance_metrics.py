"""Latency aggregation utilities."""

from typing import Iterable

import numpy as np


def latency_metrics(values: Iterable[float]) -> dict[str, float | int]:
    """Return count, mean, median, p95, min, and max in milliseconds."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {"count": int(array.size), "mean_ms": float(array.mean()), "median_ms": float(np.median(array)), "p95_ms": float(np.percentile(array, 95)), "min_ms": float(array.min()), "max_ms": float(array.max())}
