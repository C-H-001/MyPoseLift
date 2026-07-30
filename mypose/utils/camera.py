from __future__ import annotations

import numpy as np


def meters_to_millimeters(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) * 1000.0
