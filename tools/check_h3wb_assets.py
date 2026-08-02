"""Check that H3WB annotations resolve to extracted Human3.6M images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def expected_image_path(
    h36m_root: Path, subject: str, action: str, camera: str, frame: str
) -> Path:
    return (
        h36m_root
        / "original"
        / subject
        / "Images"
        / f"{action}.{camera}"
        / f"frame_{frame}.jpg"
    )


def check_assets(
    ann_file: Path, h36m_root: Path, subjects: set[str] | None = None
) -> tuple[int, int, list[Path]]:
    source = np.load(ann_file, allow_pickle=True)
    train_data = source["train_data"].item()
    missing: list[Path] = []
    total = 0
    present = 0

    for subject, subject_data in train_data.items():
        if subjects is not None and subject not in subjects:
            continue
        for action, action_data in subject_data.items():
            frames = action_data["frame_id"]
            for camera, camera_data in action_data.items():
                if not isinstance(camera_data, dict) or "pose_2d" not in camera_data:
                    continue
                for frame in frames:
                    total += 1
                    path = expected_image_path(
                        h36m_root, subject, action, camera, str(frame)
                    )
                    if path.is_file():
                        present += 1
                    elif len(missing) < 20:
                        missing.append(path)
    return total, present, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument("--h36m-root", required=True, type=Path)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-missing", type=int, default=0)
    args = parser.parse_args()

    selected = None if args.subjects is None else set(args.subjects)
    total, present, missing = check_assets(args.ann, args.h36m_root, selected)
    print(f"images: {present}/{total} present")
    missing_count = total - present
    if missing:
        print("missing examples:")
        for path in missing:
            print(f"  {path}")
    if missing_count > args.max_missing:
        return 1
    print("H3WB annotations and Human3.6M images are ready for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
