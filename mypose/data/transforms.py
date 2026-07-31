from __future__ import annotations

import numpy as np

from mypose.data.keypoints65 import LEFT_HIP, RIGHT_HIP, validate_keypoints_shape


def normalize_2d_image(keypoints_xyc: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    keypoints = np.asarray(keypoints_xyc, dtype=np.float32).copy()
    validate_keypoints_shape(keypoints, dims=2, name="keypoints_xyc")
    if keypoints.shape[1] != 3:
        raise ValueError(f"keypoints_xyc expected 3 keypoint values (x, y, confidence), got shape {keypoints.shape}")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    keypoints[:, 0] = (keypoints[:, 0] / float(width) - 0.5) * 2.0
    keypoints[:, 1] = (keypoints[:, 1] / float(height) - 0.5) * 2.0
    return keypoints


def compute_pelvis_root(points_3d: np.ndarray) -> np.ndarray:
    points = np.asarray(points_3d, dtype=np.float32)
    validate_keypoints_shape(points, dims=2, name="points_3d")
    return (points[LEFT_HIP] + points[RIGHT_HIP]) * 0.5


def make_root_relative(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_3d, dtype=np.float32)
    root = compute_pelvis_root(points)
    return points - root[None, :], root


def make_causal_window(sequence: np.ndarray, frame_idx: int, window: int) -> np.ndarray:
    seq = np.asarray(sequence, dtype=np.float32)
    validate_keypoints_shape(seq, dims=3, name="sequence")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if frame_idx < 0 or frame_idx >= seq.shape[0]:
        raise IndexError(f"frame_idx {frame_idx} outside sequence length {seq.shape[0]}")
    start = frame_idx - window + 1
    indices = [max(0, idx) for idx in range(start, frame_idx + 1)]
    return seq[indices]
