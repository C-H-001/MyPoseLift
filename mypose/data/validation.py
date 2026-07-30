from __future__ import annotations

import numpy as np

from mypose.data.transforms import compute_pelvis_root
from mypose.data.keypoints133 import NUM_KEYPOINTS, validate_keypoints_shape


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _validate_target_mask(mask: np.ndarray) -> None:
    if not np.isfinite(mask).all():
        raise ValueError("target_mask contains non-finite values")

    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError("target_mask must be binary 0 or 1 values")


def validate_sample(sample: dict) -> None:
    required = {"history_2d", "target_3d", "target_mask", "meta"}
    missing = required.difference(sample)
    if missing:
        raise ValueError(f"sample missing keys: {sorted(missing)}")

    history = np.asarray(sample["history_2d"])
    target = np.asarray(sample["target_3d"])
    mask = np.asarray(sample["target_mask"])

    validate_keypoints_shape(history, dims=3, name="history_2d")
    validate_keypoints_shape(target, dims=2, name="target_3d")
    if mask.shape not in ((NUM_KEYPOINTS,), (NUM_KEYPOINTS, 1)):
        raise ValueError(f"target_mask expected shape (133,) or (133, 1), got {mask.shape}")
    _validate_target_mask(mask)
    if mask.astype(bool).sum() == 0:
        raise ValueError("target_mask has no valid keypoints")
    _require_finite("history_2d", history)
    _require_finite("target_3d", target)
    root = compute_pelvis_root(target)
    if not np.allclose(root, np.zeros_like(root), atol=1e-6):
        raise ValueError("target_3d is expected to be pelvis-rooted (pelvis midpoint at origin)")
