import numpy as np

from mypose.data.keypoints65 import (
    COCO65_EDGES,
    NUM_KEYPOINTS,
    ORIGINAL_TO_65,
    get_part_indices,
    remap_133_to_65,
    remap_mask_133_to_65,
)


def test_65_layout_keeps_body_feet_and_hands():
    assert NUM_KEYPOINTS == 65
    assert ORIGINAL_TO_65[:23] == list(range(23))
    assert ORIGINAL_TO_65[23:44] == list(range(91, 112))
    assert ORIGINAL_TO_65[44:65] == list(range(112, 133))


def test_65_parts_include_head3_and_no_dense_face():
    assert get_part_indices("body") == list(range(17))
    assert get_part_indices("foot") == list(range(17, 23))
    assert get_part_indices("head3") == [0, 1, 2]
    assert get_part_indices("left_hand") == list(range(23, 44))
    assert get_part_indices("right_hand") == list(range(44, 65))


def test_65_edges_include_compact_hand_chains_within_bounds():
    assert (23, 24) in COCO65_EDGES
    assert (40, 41) in COCO65_EDGES
    assert (44, 45) in COCO65_EDGES
    assert (63, 64) in COCO65_EDGES
    assert all(0 <= index < NUM_KEYPOINTS for edge in COCO65_EDGES for index in edge)


def test_remap_133_to_65_preserves_expected_indices():
    points = np.arange(133 * 3, dtype=np.float32).reshape(133, 3)
    compact = remap_133_to_65(points)
    assert compact.shape == (65, 3)
    np.testing.assert_array_equal(compact[0], points[0])
    np.testing.assert_array_equal(compact[22], points[22])
    np.testing.assert_array_equal(compact[23], points[91])
    np.testing.assert_array_equal(compact[64], points[132])


def test_remap_helpers_support_batches_and_documented_mask_shapes():
    points = np.arange(2 * 133 * 2, dtype=np.float32).reshape(2, 133, 2)
    mask = np.arange(133, dtype=np.int16)

    compact_points = remap_133_to_65(points)

    assert compact_points.shape == (2, 65, 2)
    np.testing.assert_array_equal(compact_points[:, 23], points[:, 91])
    assert remap_mask_133_to_65(mask).shape == (65,)
    assert remap_mask_133_to_65(mask[:, None]).shape == (65, 1)
    np.testing.assert_array_equal(remap_mask_133_to_65(mask)[23], mask[91])
    np.testing.assert_array_equal(
        remap_mask_133_to_65(np.stack([mask, mask + 1]))[:, 23],
        [mask[91], mask[91] + 1],
    )
