"""Convert the official H3WB training NPZ to MMPose's bbox-augmented NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _bbox(points: np.ndarray, margin_ratio: float) -> dict[str, float]:
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        raise ValueError("H3WB frame has no finite 2D keypoints")
    valid = points[finite]
    x_min, y_min = valid.min(axis=0)
    x_max, y_max = valid.max(axis=0)
    margin = max(x_max - x_min, y_max - y_min) * margin_ratio
    return {
        "x_min": float(max(0.0, x_min - margin)),
        "y_min": float(max(0.0, y_min - margin)),
        "x_max": float(x_max + margin),
        "y_max": float(y_max + margin),
    }


def prepare(input_path: Path, output_path: Path, margin_ratio: float = 0.10) -> int:
    source = np.load(input_path, allow_pickle=True)
    train_data = source["train_data"].item()
    metadata = source["metadata"].item()
    bboxes: dict[tuple[str, str, str, str], dict[str, float]] = {}
    count = 0

    for subject, subject_data in train_data.items():
        for action, action_data in subject_data.items():
            frames = action_data["frame_id"]
            for camera, camera_data in action_data.items():
                if not isinstance(camera_data, dict) or "pose_2d" not in camera_data:
                    continue
                poses_2d = np.asarray(camera_data["pose_2d"])
                if len(poses_2d) != len(frames):
                    raise ValueError(
                        f"frame/keypoint length mismatch for {subject}/{action}/{camera}"
                    )
                for frame, points in zip(frames, poses_2d):
                    bboxes[(subject, action, camera, str(frame))] = _bbox(
                        points, margin_ratio
                    )
                    count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata=metadata,
        train_data=train_data,
        bbox=bboxes,
    )
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin-ratio", type=float, default=0.10)
    args = parser.parse_args()
    if args.margin_ratio < 0:
        raise ValueError("--margin-ratio must be non-negative")
    count = prepare(args.input, args.output, args.margin_ratio)
    print(f"wrote {count} bboxes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
