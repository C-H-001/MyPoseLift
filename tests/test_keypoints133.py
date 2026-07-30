import numpy as np
import pytest

from mypose.data.keypoints133 import (
    COCO_WHOLEBODY_EDGES,
    LEFT_HIP,
    NUM_KEYPOINTS,
    PART_INDICES,
    PART_SLICES,
    RIGHT_HIP,
    get_part_indices,
    validate_keypoints_shape,
)


def test_part_slices_cover_all_133_keypoints_once():
    covered = []
    for part in ("body", "foot", "face", "left_hand", "right_hand"):
        covered.extend(range(PART_SLICES[part].start, PART_SLICES[part].stop))
    assert covered == list(range(133))
    assert NUM_KEYPOINTS == 133


def test_known_part_boundaries_match_coco_wholebody():
    assert PART_INDICES["body"] == list(range(0, 17))
    assert PART_INDICES["foot"] == list(range(17, 23))
    assert PART_INDICES["face"] == list(range(23, 91))
    assert PART_INDICES["left_hand"] == list(range(91, 112))
    assert PART_INDICES["right_hand"] == list(range(112, 133))
    assert LEFT_HIP == 11
    assert RIGHT_HIP == 12


def test_validate_keypoints_shape_accepts_expected_shapes():
    validate_keypoints_shape(np.zeros((133, 3), dtype=np.float32), dims=2, name="pose")
    validate_keypoints_shape(np.zeros((8, 133, 3), dtype=np.float32), dims=3, name="history")


def test_validate_keypoints_shape_rejects_wrong_count():
    with pytest.raises(ValueError, match="expected 133 keypoints"):
        validate_keypoints_shape(np.zeros((132, 3), dtype=np.float32), dims=2, name="pose")


def test_get_part_indices_rejects_unknown_part():
    with pytest.raises(KeyError, match="unknown keypoint part"):
        get_part_indices("tail")


def test_graph_edges_are_in_bounds_and_nonempty():
    assert len(COCO_WHOLEBODY_EDGES) > 120
    for a, b in COCO_WHOLEBODY_EDGES:
        assert 0 <= a < 133
        assert 0 <= b < 133
        assert a != b
