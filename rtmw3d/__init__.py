"""Lightweight runtime contracts for the RTMW3D integration."""

from .benchmark import LatencyStats, summarize_latencies
from .adapter import RTMW3DAdapter
from .defaults import (
    COCO_WHOLEBODY_GROUPS,
    DEFAULT_DETECTOR_CHECKPOINT_URL,
    DEFAULT_DETECTOR_CONFIG,
    DEFAULT_POSE_CHECKPOINT_URL,
    DEFAULT_POSE_CONFIG,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    SUPPORTED_INPUT_MODES,
    validate_input_mode,
)
from .types import (
    PoseResult,
    RuntimeConfig,
    RuntimeDependencyError,
    require_runtime_dependency,
)
from .reduced_layout import (
    REDUCED_KEYPOINT_INDICES,
    REDUCED_KEYPOINT_NAMES,
    REDUCED_NUM_KEYPOINTS,
    reduce_joint_parents,
    reduced_mapping,
)

__all__ = [
    "COCO_WHOLEBODY_GROUPS",
    "DEFAULT_DETECTOR_CHECKPOINT_URL",
    "DEFAULT_DETECTOR_CONFIG",
    "DEFAULT_POSE_CHECKPOINT_URL",
    "DEFAULT_POSE_CONFIG",
    "KEYPOINT_NAMES",
    "LatencyStats",
    "NUM_KEYPOINTS",
    "PoseResult",
    "RTMW3DAdapter",
    "RuntimeConfig",
    "RuntimeDependencyError",
    "REDUCED_KEYPOINT_INDICES",
    "REDUCED_KEYPOINT_NAMES",
    "REDUCED_NUM_KEYPOINTS",
    "SUPPORTED_INPUT_MODES",
    "require_runtime_dependency",
    "reduce_joint_parents",
    "reduced_mapping",
    "summarize_latencies",
    "validate_input_mode",
]
