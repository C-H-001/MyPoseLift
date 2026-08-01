"""Dependency-free latency aggregation for model-only measurements."""

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class LatencyStats:
    samples: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float


def _percentile(sorted_values: list[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def summarize_latencies(latencies_ms: Iterable[float]) -> LatencyStats:
    """Compute mean, p50, p95, and mean-based FPS from non-negative milliseconds."""

    values = [float(value) for value in latencies_ms]
    if not values:
        raise ValueError("latencies_ms must contain at least one sample")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("latency samples must be finite and non-negative")

    ordered = sorted(values)
    mean_ms = sum(values) / len(values)
    fps = float("inf") if mean_ms == 0.0 else 1000.0 / mean_ms
    return LatencyStats(
        samples=len(values),
        mean_ms=mean_ms,
        p50_ms=_percentile(ordered, 50.0),
        p95_ms=_percentile(ordered, 95.0),
        fps=fps,
    )
