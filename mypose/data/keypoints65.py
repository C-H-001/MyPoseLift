from __future__ import annotations

import numpy as np

NUM_KEYPOINTS = 65
LEFT_HIP = 11
RIGHT_HIP = 12

ORIGINAL_TO_65 = (
    list(range(23))
    + list(range(91, 112))
    + list(range(112, 133))
)

PART_INDICES: dict[str, list[int]] = {
    "body": list(range(17)),
    "foot": list(range(17, 23)),
    "head3": [0, 1, 2],
    "left_hand": list(range(23, 44)),
    "right_hand": list(range(44, 65)),
}

BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]
FOOT_EDGES = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
LEFT_HAND_EDGES = [
    (23, 24), (24, 25), (25, 26), (26, 27),
    (23, 28), (28, 29), (29, 30), (30, 31),
    (23, 32), (32, 33), (33, 34), (34, 35),
    (23, 36), (36, 37), (37, 38), (38, 39),
    (23, 40), (40, 41), (41, 42), (42, 43),
]
RIGHT_HAND_EDGES = [(a + 21, b + 21) for a, b in LEFT_HAND_EDGES]

COCO65_EDGES = BODY_EDGES + FOOT_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES


def get_part_indices(part: str) -> list[int]:
    try:
        return PART_INDICES[part]
    except KeyError as exc:
        known = ", ".join(sorted(PART_INDICES))
        raise KeyError(f"unknown keypoint part {part!r}; expected one of {known}") from exc


def remap_133_to_65(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points)
    if array.ndim not in (2, 3) or array.shape[-2] != 133:
        raise ValueError(
            f"points expected shape (133, C) or (N, 133, C), got {array.shape}"
        )
    return np.take(array, ORIGINAL_TO_65, axis=-2)


def remap_mask_133_to_65(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 1 and array.shape == (133,):
        return np.take(array, ORIGINAL_TO_65, axis=0)
    if array.ndim == 2 and array.shape == (133, 1):
        return np.take(array, ORIGINAL_TO_65, axis=0)
    if array.ndim == 2 and array.shape[1] == 133:
        return np.take(array, ORIGINAL_TO_65, axis=1)
    raise ValueError(
        f"mask expected shape (133,), (133, 1), or (N, 133), got {array.shape}"
    )


def validate_keypoints_shape(array: np.ndarray, dims: int, name: str) -> None:
    if array.ndim != dims:
        raise ValueError(f"{name} expected {dims} dims, got shape {array.shape}")
    if array.shape[-2] != NUM_KEYPOINTS:
        raise ValueError(f"{name} expected 65 keypoints, got shape {array.shape}")
    if array.shape[-1] not in (2, 3):
        raise ValueError(
            f"{name} expected coordinate dimension 2 or 3, got shape {array.shape}"
        )
