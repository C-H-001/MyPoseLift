"""Reduced COCO-WholeBody layouts used by the 65-point RTMW3D variant."""

from typing import Iterable

from .defaults import KEYPOINT_NAMES


# COCO-WholeBody body points already include nose and both eyes. The extra
# 68-point face block is removed while feet and both hands are retained.
REDUCED_KEYPOINT_INDICES = tuple(range(23)) + tuple(range(91, 133))
REDUCED_NUM_KEYPOINTS = len(REDUCED_KEYPOINT_INDICES)
REDUCED_KEYPOINT_NAMES = tuple(KEYPOINT_NAMES[i] for i in REDUCED_KEYPOINT_INDICES)


def reduced_mapping() -> tuple[tuple[int, int], ...]:
    """Return ``(source_index, reduced_index)`` converter pairs."""

    return tuple(
        (source_index, reduced_index)
        for reduced_index, source_index in enumerate(REDUCED_KEYPOINT_INDICES)
    )


def reduce_joint_parents(parents: Iterable[int]) -> tuple[int, ...]:
    """Remap a full-layout parent list to the reduced layout."""

    source_parents = tuple(parents)
    source_to_reduced = {
        source_index: reduced_index
        for reduced_index, source_index in enumerate(REDUCED_KEYPOINT_INDICES)
    }
    reduced = []
    for source_index in REDUCED_KEYPOINT_INDICES:
        parent = source_parents[source_index]
        if parent not in source_to_reduced:
            raise ValueError(
                f"parent {parent} of retained point {source_index} was removed"
            )
        reduced.append(source_to_reduced[parent])
    return tuple(reduced)


__all__ = [
    "REDUCED_KEYPOINT_INDICES",
    "REDUCED_KEYPOINT_NAMES",
    "REDUCED_NUM_KEYPOINTS",
    "reduce_joint_parents",
    "reduced_mapping",
]
