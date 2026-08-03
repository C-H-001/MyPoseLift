"""Pure helpers for validating and scaling RTMW3D supervision weights."""

from __future__ import annotations

from typing import MutableMapping, Any

import numpy as np


def scale_2d_only_target_weights(
    results: MutableMapping[str, Any], weight: float = 0.25
) -> MutableMapping[str, Any]:
    """Down-weight XY labels for samples without metric depth labels.

    ``SimCC3DLabel`` emits ``with_z_label=False`` and a zero ``weight_z`` for
    COCO-style 2D samples.  The regular target-weight channel is still used
    for XY, so it is the correct place to apply a dataset-level loss weight.
    """

    if not 0.0 < weight <= 1.0:
        raise ValueError(f"weight must be in (0, 1], got {weight}")

    flags = np.asarray(results.get("with_z_label", []), dtype=bool).reshape(-1)
    if flags.size != 1:
        raise ValueError(
            "with_z_label must contain exactly one sample-level flag, "
            f"got shape {flags.shape}"
        )
    if flags[0]:
        return results

    weight_z = np.asarray(results.get("weight_z"))
    if weight_z.size == 0:
        raise KeyError("2D-only targets must contain weight_z")
    if not np.allclose(weight_z, 0.0):
        raise ValueError("2D-only targets must have zero depth weights")

    keypoint_weights = results.get("keypoint_weights")
    if keypoint_weights is None:
        raise KeyError("encoded targets must contain keypoint_weights")
    results["keypoint_weights"] = np.asarray(keypoint_weights).copy() * weight
    return results
