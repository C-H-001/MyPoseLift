import numpy as np
import pytest

from src.augmentation.sequence_augmentation import (
    AugmentationConfig,
    CameraModel,
    ResidualBank,
    augment_sequence,
)


def _camera(projection="weak"):
    return CameraModel(
        focal_length=100.0,
        image_size=(200, 100),
        camera_distance=10.0,
        projection=projection,
    )


def _pose_clip():
    pose = np.zeros((5, 3, 3), dtype=np.float32)
    pose[:, 1, 0] = np.arange(5, dtype=np.float32)
    pose[:, 2, 1] = 1.0
    return pose


def _fixed_config(**kwargs):
    values = dict(
        rotation_degrees=(0.0, 0.0, 0.0),
        scale_range=(1.0, 1.0),
        root_index=0,
    )
    values.update(kwargs)
    return AugmentationConfig(**values)


def test_augment_sequence_returns_video_pose_training_shapes():
    result = augment_sequence(
        _pose_clip(), _camera(), config=_fixed_config(), rng=np.random.default_rng(0)
    )

    assert result.keypoints_2d.shape == (5, 3, 2)
    assert result.clean_keypoints_2d.shape == (5, 3, 2)
    assert result.target_3d.shape == (3, 3)
    assert result.visibility.shape == (5, 3)
    assert result.pose_3d.shape == (5, 3, 3)
    assert result.center_frame == 2


def test_weak_projection_is_centered_and_uses_one_camera_for_all_frames():
    result = augment_sequence(
        _pose_clip(), _camera(), config=_fixed_config(), rng=np.random.default_rng(1)
    )

    # x_pixel = 100 * x / 10 + 100, then normalized to [-1, 1].
    expected_x = 2.0 * (np.arange(5, dtype=np.float32) * 10.0 + 100.0) / 200.0 - 1.0
    np.testing.assert_allclose(result.clean_keypoints_2d[:, 1, 0], expected_x)
    np.testing.assert_allclose(
        result.clean_keypoints_2d[:, 2, 1], 2.0 * 60.0 / 100.0 - 1.0, atol=1e-6
    )
    np.testing.assert_allclose(result.clean_keypoints_2d, result.keypoints_2d)


def test_target_is_center_frame_root_relative_after_transform():
    result = augment_sequence(
        _pose_clip(), _camera(), config=_fixed_config(), rng=np.random.default_rng(2)
    )

    expected = result.pose_3d[result.center_frame].copy()
    expected -= expected[0]
    np.testing.assert_allclose(result.target_3d, expected)


def test_perspective_projection_rejects_non_positive_projected_depth():
    pose = _pose_clip()
    camera = CameraModel(
        focal_length=100.0,
        image_size=(200, 100),
        camera_distance=0.5,
        projection="perspective",
    )
    pose[:, 0, 2] = -1.0

    with pytest.raises(ValueError, match="positive depth"):
        augment_sequence(pose, camera, config=_fixed_config())


def test_residual_bank_is_replayed_without_changing_3d_target():
    bank = ResidualBank.from_array(np.full((2, 3, 2), 0.1, dtype=np.float32))
    result = augment_sequence(
        _pose_clip(),
        _camera(),
        config=_fixed_config(residual_rho=0.0, residual_scale=1.0),
        residual_bank=bank,
        rng=np.random.default_rng(3),
    )

    np.testing.assert_allclose(
        result.keypoints_2d - result.clean_keypoints_2d, 0.1, atol=1e-6
    )
    np.testing.assert_allclose(result.target_3d, result.pose_3d[2] - result.pose_3d[2, 0])


def test_rho_one_keeps_sampled_residual_constant_across_time():
    bank = ResidualBank.from_array(
        np.array([[[-1.0, -2.0]], [[1.0, 2.0]]], dtype=np.float32)
    )

    sampled = bank.sample(6, np.random.default_rng(4), rho=1.0, scale=1.0)

    np.testing.assert_allclose(sampled, np.broadcast_to(sampled[0:1], sampled.shape))


def test_dropout_marks_contiguous_missing_spans_and_keeps_coordinates():
    clean = augment_sequence(
        _pose_clip(),
        _camera(),
        config=_fixed_config(
            dropout_probability=1.0,
            max_dropout_span=2,
        ),
        rng=np.random.default_rng(5),
    )

    assert not np.any(clean.visibility)
    np.testing.assert_allclose(clean.keypoints_2d, clean.clean_keypoints_2d)


def test_augmentation_does_not_mutate_input():
    pose = _pose_clip()
    original = pose.copy()
    augment_sequence(
        pose,
        _camera(),
        config=_fixed_config(
            rotation_degrees=(10.0, 10.0, 10.0),
            scale_range=(0.8, 1.2),
            residual_rho=0.8,
            residual_scale=1.0,
            dropout_probability=0.5,
            max_dropout_span=2,
        ),
        residual_bank=ResidualBank.from_array(np.zeros((1, 3, 2), dtype=np.float32)),
        rng=np.random.default_rng(6),
    )

    np.testing.assert_array_equal(pose, original)


def test_same_seed_reproduces_the_complete_augmented_sample():
    config = _fixed_config(
        rotation_degrees=(10.0, 5.0, 3.0),
        scale_range=(0.8, 1.2),
        residual_rho=0.7,
        residual_scale=1.0,
        dropout_probability=0.25,
        max_dropout_span=2,
    )
    bank = ResidualBank.from_array(np.arange(12, dtype=np.float32).reshape(2, 3, 2))
    first = augment_sequence(
        _pose_clip(), _camera("perspective"), config, bank, np.random.default_rng(7)
    )
    second = augment_sequence(
        _pose_clip(), _camera("perspective"), config, bank, np.random.default_rng(7)
    )

    np.testing.assert_array_equal(first.keypoints_2d, second.keypoints_2d)
    np.testing.assert_array_equal(first.target_3d, second.target_3d)
    np.testing.assert_array_equal(first.visibility, second.visibility)


def test_invalid_inputs_fail_with_clear_errors():
    with pytest.raises(ValueError, match=r"shape \(T, J, 3\)"):
        augment_sequence(np.zeros((3, 2, 2)), _camera())

    with pytest.raises(ValueError, match="focal_length"):
        CameraModel(focal_length=0.0, image_size=(200, 100))

    with pytest.raises(ValueError, match="residuals must have shape"):
        ResidualBank.from_array(np.zeros((3, 2)))
