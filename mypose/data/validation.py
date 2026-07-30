from __future__ import annotations

import numpy as np

from mypose.data.keypoints133 import NUM_KEYPOINTS, validate_keypoints_shape


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


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
    if mask.astype(bool).sum() == 0:
        raise ValueError("target_mask has no valid keypoints")
    _require_finite("history_2d", history)
    _require_finite("target_3d", target)
