from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mypose.data.keypoints65 import NUM_KEYPOINTS, remap_133_to_65
from mypose.data.transforms import normalize_2d_image


def _reshape_part(values: list[float], expected_points: int) -> np.ndarray:
    if not values:
        return np.zeros((expected_points, 3), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    if arr.shape[0] != expected_points:
        raise ValueError(f"expected {expected_points} points, got {arr.shape[0]}")
    return arr


def _annotation_to_133(annotation: dict[str, Any]) -> np.ndarray:
    body = _reshape_part(annotation.get("keypoints", []), 17)
    foot = _reshape_part(annotation.get("foot_kpts", []), 6)
    face = _reshape_part(annotation.get("face_kpts", []), 68)
    left = _reshape_part(annotation.get("lefthand_kpts", []), 21)
    right = _reshape_part(annotation.get("righthand_kpts", []), 21)
    return np.concatenate([body, foot, face, left, right], axis=0).astype(np.float32)


def load_coco_wholebody_annotations(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in payload["images"]}
    samples = []
    for ann in payload["annotations"]:
        image = images[ann["image_id"]]
        keypoints = remap_133_to_65(_annotation_to_133(ann))
        samples.append({
            "keypoints_2d": keypoints,
            "image_size": (int(image["width"]), int(image["height"])),
            "bbox": ann.get("bbox"),
            "meta": {
                "source": "coco-wholebody",
                "image_id": ann["image_id"],
                "annotation_id": ann["id"],
                "file_name": image["file_name"],
            },
        })
    return samples


class CocoWholeBodyDataset:
    def __init__(self, annotation_file: Path, image_root: Path | None = None) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_root = image_root
        self.samples = load_coco_wholebody_annotations(self.annotation_file)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        raw = self.samples[index]
        keypoints = normalize_2d_image(raw["keypoints_2d"], raw["image_size"])
        sample = {
            "history_2d": keypoints[None, :, :],
            "target_3d": np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32),
            "target_mask": np.zeros((NUM_KEYPOINTS,), dtype=bool),
            "meta": raw["meta"],
        }
        sample["meta"]["source"] = "coco-wholebody"
        return sample
