import numpy as np
import pytest
from types import SimpleNamespace

from rtmw3d.validation_visualization import (
    _sample_from_data,
    compute_sample_metrics,
)


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


def test_sample_uses_projected_prediction_for_2d_panel():
    target = np.zeros((1, 3, 3), dtype=np.float32)
    pred = np.ones((1, 3, 3), dtype=np.float32)
    projected = np.full((1, 3, 2), 7.0, dtype=np.float32)
    sample = SimpleNamespace(
        gt_instances=SimpleNamespace(
            lifting_target=target,
            lifting_target_visible=np.ones((1, 3), dtype=np.float32),
        ),
        metainfo={},
    )
    output = SimpleNamespace(
        pred_instances=SimpleNamespace(
            keypoints=pred,
            transformed_keypoints=projected,
        )
    )

    result = _sample_from_data(sample, output, 0)

    np.testing.assert_array_equal(result["keypoints_2d"], projected[0])
