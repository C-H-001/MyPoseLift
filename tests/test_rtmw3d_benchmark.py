import pytest

from rtmw3d.benchmark import LatencyStats, summarize_latencies


def test_latency_summary_reports_percentiles_and_fps_from_mean():
    stats = summarize_latencies([10.0, 20.0, 30.0, 40.0])

    assert isinstance(stats, LatencyStats)
    assert stats.samples == 4
    assert stats.mean_ms == pytest.approx(25.0)
    assert stats.p50_ms == pytest.approx(25.0)
    assert stats.p95_ms == pytest.approx(38.5)
    assert stats.fps == pytest.approx(40.0)


def test_latency_summary_rejects_empty_or_invalid_samples():
    with pytest.raises(ValueError, match="at least one"):
        summarize_latencies([])

    with pytest.raises(ValueError, match="finite and non-negative"):
        summarize_latencies([10.0, float("nan")])
