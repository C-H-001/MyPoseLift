from __future__ import annotations

import numpy as np

NUM_KEYPOINTS = 133
LEFT_HIP = 11
RIGHT_HIP = 12

PART_SLICES: dict[str, slice] = {
    "body": slice(0, 17),
    "foot": slice(17, 23),
    "face": slice(23, 91),
    "left_hand": slice(91, 112),
    "right_hand": slice(112, 133),
}

PART_INDICES: dict[str, list[int]] = {
    name: list(range(part_slice.start, part_slice.stop))
    for name, part_slice in PART_SLICES.items()
}

BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]
FOOT_EDGES = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
FACE_EDGES = [(i, i + 1) for i in range(23, 90)]
LEFT_HAND_EDGES = [
    (91, 92), (92, 93), (93, 94), (94, 95),
    (91, 96), (96, 97), (97, 98), (98, 99),
    (91, 100), (100, 101), (101, 102), (102, 103),
    (91, 104), (104, 105), (105, 106), (106, 107),
    (91, 108), (108, 109), (109, 110), (110, 111),
]
RIGHT_HAND_EDGES = [(a + 21, b + 21) for a, b in LEFT_HAND_EDGES]

COCO_WHOLEBODY_EDGES = BODY_EDGES + FOOT_EDGES + FACE_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES


def get_part_indices(part: str) -> list[int]:
    try:
        return PART_INDICES[part]
    except KeyError as exc:
        known = ", ".join(sorted(PART_INDICES))
        raise KeyError(f"unknown keypoint part {part!r}; expected one of {known}") from exc


def validate_keypoints_shape(array: np.ndarray, dims: int, name: str) -> None:
    if array.ndim != dims:
        raise ValueError(f"{name} expected {dims} dims, got shape {array.shape}")
    keypoint_axis = -2
    if array.shape[keypoint_axis] != NUM_KEYPOINTS:
        raise ValueError(f"{name} expected 133 keypoints, got shape {array.shape}")
    if array.shape[-1] not in (2, 3):
        raise ValueError(f"{name} expected coordinate dimension 2 or 3, got shape {array.shape}")
