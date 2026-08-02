"""Normalize flat H36W JPEGs into the directory layout expected by H3WB."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def source_image(
    images_root: Path, subject: str, action: str, camera: str, frame: str
) -> Path:
    # The H36W archive uses one-based image names; H3WB frame ids are zero-based.
    archive_frame = int(frame) + 1
    archive_action = action.replace(" ", "_")
    return images_root / (
        f"{subject}_{archive_action}.{camera}_{archive_frame:06d}.jpg"
    )


def target_image(
    output_root: Path, subject: str, action: str, camera: str, frame: str
) -> Path:
    return (
        output_root
        / "original"
        / subject
        / "Images"
        / f"{action}.{camera}"
        / f"frame_{frame}.jpg"
    )


def prepare(ann_file: Path, images_root: Path, output_root: Path) -> tuple[int, list[Path]]:
    source = np.load(ann_file, allow_pickle=True)
    train_data = source["train_data"].item()
    linked = 0
    missing: list[Path] = []

    for subject, subject_data in train_data.items():
        for action, action_data in subject_data.items():
            frames = action_data["frame_id"]
            for camera, camera_data in action_data.items():
                if not isinstance(camera_data, dict) or "pose_2d" not in camera_data:
                    continue
                for frame in frames:
                    frame_text = str(frame)
                    src = source_image(images_root, subject, action, camera, frame_text)
                    dst = target_image(output_root, subject, action, camera, frame_text)
                    if not src.is_file():
                        if len(missing) < 20:
                            missing.append(src)
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        if os.path.samefile(src, dst):
                            linked += 1
                            continue
                        raise FileExistsError(f"target already exists and differs: {dst}")
                    os.link(src, dst)
                    linked += 1
    return linked, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    linked, missing = prepare(args.ann, args.images_root, args.output_root)
    print(f"linked {linked} images under {args.output_root}")
    if missing:
        print("missing examples:")
        for path in missing:
            print(f"  {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
