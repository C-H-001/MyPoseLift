"""Runtime and result types that do not import the MMPose stack."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Optional

import numpy as np

from .defaults import DEFAULT_DETECTOR_CHECKPOINT_URL
from .defaults import DEFAULT_DETECTOR_CONFIG
from .defaults import DEFAULT_POSE_CHECKPOINT_URL
from .defaults import DEFAULT_POSE_CONFIG
from .defaults import RTMW3D_INPUT_SIZE
from .defaults import NUM_KEYPOINTS


class RuntimeDependencyError(ImportError):
    """Raised when an optional inference dependency is unavailable."""


def require_runtime_dependency(module_name: str, feature_name: str) -> Any:
    """Import an optional runtime package with an actionable feature-level error."""

    try:
        return import_module(module_name)
    except ImportError as exc:
        raise RuntimeDependencyError(
            f"{feature_name} requires optional package {module_name!r}. "
            "Install the documented RTMW3D runtime dependencies first."
        ) from exc


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration shared by image, video, and webcam inference."""

    detector_config: str = DEFAULT_DETECTOR_CONFIG
    detector_checkpoint: str = DEFAULT_DETECTOR_CHECKPOINT_URL
    pose_config: str = DEFAULT_POSE_CONFIG
    pose_checkpoint: str = DEFAULT_POSE_CHECKPOINT_URL
    device: str = "cpu"
    bbox_thr: float = 0.3
    max_instances: int = 1
    input_size: tuple[int, int] = RTMW3D_INPUT_SIZE
    use_full_frame: bool = False

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("device must be a non-empty string")
        if not 0.0 <= self.bbox_thr <= 1.0:
            raise ValueError("bbox_thr must be between 0 and 1")
        if self.max_instances < 1:
            raise ValueError("max_instances must be at least 1")
        if len(self.input_size) != 2 or any(size <= 0 for size in self.input_size):
            raise ValueError("input_size must contain two positive dimensions")


@dataclass(frozen=True)
class PoseResult:
    """A stable representation of one frame's 133-point 3D predictions."""

    keypoints_3d: np.ndarray
    keypoints_2d: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    bboxes: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints_3d, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (NUM_KEYPOINTS, 3):
            raise ValueError(
                "keypoints_3d must have shape (N, 133, 3); "
                f"got {keypoints.shape}"
            )
        object.__setattr__(self, "keypoints_3d", keypoints)

        if self.keypoints_2d is not None:
            keypoints_2d = np.asarray(self.keypoints_2d, dtype=np.float32)
            expected_2d = (keypoints.shape[0], NUM_KEYPOINTS, 2)
            if keypoints_2d.shape != expected_2d:
                raise ValueError(
                    f"keypoints_2d must have shape {expected_2d}; "
                    f"got {keypoints_2d.shape}"
                )
            object.__setattr__(self, "keypoints_2d", keypoints_2d)

        if self.scores is not None:
            scores = np.asarray(self.scores, dtype=np.float32)
            expected = (keypoints.shape[0], NUM_KEYPOINTS)
            if scores.shape != expected:
                raise ValueError(f"scores must have shape {expected}; got {scores.shape}")
            object.__setattr__(self, "scores", scores)

        if self.bboxes is not None:
            bboxes = np.asarray(self.bboxes, dtype=np.float32)
            if bboxes.ndim != 2 or bboxes.shape[0] != keypoints.shape[0]:
                raise ValueError(
                    "bboxes must have shape (N, 4) or (N, 5); "
                    f"got {bboxes.shape}"
                )
            if bboxes.shape[1] not in (4, 5):
                raise ValueError(
                    "bboxes must have shape (N, 4) or (N, 5); "
                    f"got {bboxes.shape}"
                )
            object.__setattr__(self, "bboxes", bboxes)

    @classmethod
    def from_arrays(
        cls,
        keypoints_3d: np.ndarray,
        *,
        keypoints_2d: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        bboxes: Optional[np.ndarray] = None,
    ) -> "PoseResult":
        """Normalize one-person arrays and validate the multi-person contract."""

        keypoints = np.asarray(keypoints_3d, dtype=np.float32)
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
        if keypoints.ndim != 3 or keypoints.shape[1:] != (NUM_KEYPOINTS, 3):
            raise ValueError(
                "keypoints_3d must have shape (N, 133, 3); "
                f"got {keypoints.shape}"
            )

        person_count = keypoints.shape[0]
        normalized_keypoints_2d = None
        if keypoints_2d is not None:
            keypoints_2d_array = np.asarray(keypoints_2d, dtype=np.float32)
            if keypoints_2d_array.ndim == 2 and person_count == 1:
                keypoints_2d_array = keypoints_2d_array[None, ...]
            normalized_keypoints_2d = keypoints_2d_array

        normalized_scores = None
        if scores is not None:
            score_array = np.asarray(scores, dtype=np.float32)
            if score_array.ndim == 1 and person_count == 1:
                score_array = score_array[None, ...]
            normalized_scores = score_array

        normalized_bboxes = None
        if bboxes is not None:
            bbox_array = np.asarray(bboxes, dtype=np.float32)
            if bbox_array.ndim == 1 and person_count == 1:
                bbox_array = bbox_array[None, ...]
            normalized_bboxes = bbox_array

        return cls(
            keypoints_3d=keypoints,
            keypoints_2d=normalized_keypoints_2d,
            scores=normalized_scores,
            bboxes=normalized_bboxes,
        )
