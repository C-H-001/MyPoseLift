import numpy as np
import pytest

from rtmw3d.validation_visualization import compute_sample_metrics


def test_sample_metrics_are_zero_for_identical_pose():
    pose = np.arange(18, dtype=np.float32).reshape(6, 3) / 100

    metrics = compute_sample_metrics(pose, pose)

    assert metrics["mpjpe_mm"] == 0.0
    assert metrics["p_mpjpe_mm"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["valid_keypoints"] == 6


def test_sample_metrics_respect_visibility_mask():
    target = np.zeros((4, 3), dtype=np.float32)
    pred = target.copy()
    pred[0, 0] = 0.1
    pred[1, 0] = 0.2

    metrics = compute_sample_metrics(pred, target, np.array([[1, 0, 0, 0]]))

    assert metrics["mpjpe_mm"] == pytest.approx(100.0)
    assert metrics["valid_keypoints"] == 1
