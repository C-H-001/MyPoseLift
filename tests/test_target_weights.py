import numpy as np
import pytest

from rtmw3d.target_weights import scale_2d_only_target_weights


def test_scales_xy_weights_for_2d_only_targets_and_keeps_depth_disabled():
    results = {
        "with_z_label": [False],
        "keypoint_weights": np.ones((1, 68), dtype=np.float32),
        "weight_z": np.zeros((1, 68), dtype=np.float32),
    }

    scaled = scale_2d_only_target_weights(results, weight=0.25)

    np.testing.assert_allclose(scaled["keypoint_weights"], 0.25)
    np.testing.assert_allclose(scaled["weight_z"], 0.0)


def test_leaves_metric_3d_targets_unchanged():
    weights = np.ones((1, 68), dtype=np.float32)
    results = {
        "with_z_label": [True],
        "keypoint_weights": weights.copy(),
        "weight_z": weights.copy(),
    }

    scaled = scale_2d_only_target_weights(results, weight=0.25)

    np.testing.assert_array_equal(scaled["keypoint_weights"], weights)
    np.testing.assert_array_equal(scaled["weight_z"], weights)


def test_rejects_nonzero_depth_weights_for_2d_only_targets():
    with pytest.raises(ValueError, match="zero depth weights"):
        scale_2d_only_target_weights(
            {
                "with_z_label": [False],
                "keypoint_weights": np.ones((1, 2), dtype=np.float32),
                "weight_z": np.ones((1, 2), dtype=np.float32),
            }
        )
