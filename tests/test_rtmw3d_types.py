import numpy as np
import pytest

from rtmw3d.types import (
    PoseResult,
    RuntimeConfig,
    RuntimeDependencyError,
    require_runtime_dependency,
)


def test_pose_result_normalizes_single_pose_and_preserves_optional_fields():
    keypoints = np.zeros((133, 3), dtype=np.float32)
    scores = np.ones(133, dtype=np.float32)
    bbox = np.array([1, 2, 100, 200, 0.9], dtype=np.float32)
    keypoints_2d = np.zeros((133, 2), dtype=np.float32)

    result = PoseResult.from_arrays(
        keypoints, keypoints_2d=keypoints_2d, scores=scores, bboxes=bbox
    )

    assert result.keypoints_3d.shape == (1, 133, 3)
    assert result.keypoints_3d.dtype == np.float32
    assert result.keypoints_2d.shape == (1, 133, 2)
    assert result.scores.shape == (1, 133)
    assert result.bboxes.shape == (1, 5)
    np.testing.assert_array_equal(result.scores[0], scores)


def test_pose_result_rejects_unexpected_shapes():
    with pytest.raises(ValueError, match="keypoints_3d must have shape"):
        PoseResult.from_arrays(np.zeros((17, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="scores must have shape"):
        PoseResult.from_arrays(
            np.zeros((2, 133, 3), dtype=np.float32),
            scores=np.zeros((2, 17), dtype=np.float32),
        )


def test_runtime_config_uses_cpu_safe_defaults():
    config = RuntimeConfig()

    assert config.device == "cpu"
    assert config.bbox_thr == 0.3
    assert config.max_instances == 1
    assert config.input_size == (288, 384)


def test_pose_result_supports_reduced_65_point_layout():
    result = PoseResult.from_arrays(
        np.zeros((65, 3), dtype=np.float32),
        keypoints_2d=np.zeros((65, 2), dtype=np.float32),
        scores=np.ones(65, dtype=np.float32),
        keypoint_count=65,
    )

    assert result.keypoint_count == 65
    assert result.keypoints_3d.shape == (1, 65, 3)


def test_missing_optional_dependency_has_actionable_error():
    with pytest.raises(RuntimeDependencyError, match="MMPose runtime"):
        require_runtime_dependency("module_that_does_not_exist_for_rtmw3d", "MMPose runtime")
