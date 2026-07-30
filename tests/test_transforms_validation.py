import numpy as np
import pytest

from mypose.data.transforms import make_causal_window, normalize_2d_image
from mypose.data.validation import validate_sample


def test_normalize_2d_image_maps_pixels_to_centered_range_and_preserves_confidence():
    keypoints = np.zeros((133, 3), dtype=np.float32)
    keypoints[0] = [320.0, 240.0, 0.75]
    normalized = normalize_2d_image(keypoints, image_size=(640, 480))
    np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.75])


def test_normalize_2d_image_requires_confidence_channel():
    keypoints = np.zeros((133, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="expected 3 keypoint values"):
        normalize_2d_image(keypoints, image_size=(640, 480))


def test_make_causal_window_left_pads_first_frames_with_first_observation():
    seq = np.arange(5 * 133 * 3, dtype=np.float32).reshape(5, 133, 3)
    window = make_causal_window(seq, frame_idx=1, window=4)
    assert window.shape == (4, 133, 3)
    np.testing.assert_allclose(window[0], seq[0])
    np.testing.assert_allclose(window[1], seq[0])
    np.testing.assert_allclose(window[2], seq[0])
    np.testing.assert_allclose(window[3], seq[1])


def test_make_causal_window_rejects_future_frame_request():
    seq = np.zeros((5, 133, 3), dtype=np.float32)
    with pytest.raises(IndexError, match="frame_idx"):
        make_causal_window(seq, frame_idx=5, window=3)


def test_validate_sample_rejects_nan_target():
    sample = {
        "history_2d": np.zeros((1, 133, 3), dtype=np.float32),
        "target_3d": np.zeros((133, 3), dtype=np.float32),
        "target_mask": np.ones((133,), dtype=bool),
        "meta": {"source": "synthetic"},
    }
    sample["target_3d"][0, 0] = np.nan
    with pytest.raises(ValueError, match="target_3d contains non-finite"):
        validate_sample(sample)


def test_validate_sample_rejects_non_root_relative_target():
    sample = {
        "history_2d": np.zeros((1, 133, 3), dtype=np.float32),
        "target_3d": np.zeros((133, 3), dtype=np.float32),
        "target_mask": np.ones((133,), dtype=bool),
        "meta": {"source": "synthetic"},
    }
    sample["target_3d"][11] = [0.0, 0.0, 0.0]
    sample["target_3d"][12] = [2.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="pelvis-rooted"):
        validate_sample(sample)


def test_validate_sample_rejects_non_binary_mask_values():
    sample = {
        "history_2d": np.zeros((1, 133, 3), dtype=np.float32),
        "target_3d": np.zeros((133, 3), dtype=np.float32),
        "target_mask": np.full((133,), 0.5, dtype=np.float32),
        "meta": {"source": "synthetic"},
    }
    with pytest.raises(ValueError, match="target_mask must be binary"):
        validate_sample(sample)


def test_validate_sample_rejects_sparsely_valid_mask():
    sample = {
        "history_2d": np.zeros((1, 133, 3), dtype=np.float32),
        "target_3d": np.zeros((133, 3), dtype=np.float32),
        "target_mask": np.zeros((133,), dtype=bool),
        "meta": {"source": "synthetic"},
    }
    sample["target_mask"][0] = True
    with pytest.raises(ValueError, match="too few valid keypoints"):
        validate_sample(sample)


def test_validate_sample_requires_exactly_three_target_coordinates():
    sample = {
        "history_2d": np.zeros((1, 133, 3), dtype=np.float32),
        "target_3d": np.zeros((133, 2), dtype=np.float32),
        "target_mask": np.ones((133,), dtype=bool),
        "meta": {"source": "synthetic"},
    }

    with pytest.raises(ValueError, match="target_3d expected 3 coordinate values"):
        validate_sample(sample)


def test_validate_sample_canonicalizes_singleton_column_mask():
    sample = {
        "history_2d": np.zeros((1, 133, 3), dtype=np.float32),
        "target_3d": np.zeros((133, 3), dtype=np.float32),
        "target_mask": np.ones((133, 1), dtype=bool),
        "meta": {"source": "synthetic"},
    }

    validate_sample(sample)

    assert sample["target_mask"].shape == (133,)
