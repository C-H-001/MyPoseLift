from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from mypose.data.transforms import make_causal_window, make_root_relative, normalize_2d_image
from mypose.data.validation import validate_sample


def _dict_points(entry: dict[str, Any] | None, dims: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros((133, dims), dtype=np.float32)
    mask = np.zeros((133,), dtype=bool)
    for key, value in (entry or {}).items():
        idx = int(key)
        if idx < 0 or idx >= 133:
            raise ValueError(f"keypoint index {idx} outside 0..132")
        names = ("x", "y") if dims == 2 else ("x", "y", "z")
        points[idx] = [float(value[name]) for name in names]
        mask[idx] = True
    return points, mask


def load_h3wb_json(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = []
    for sample_id, item in payload.items():
        xy, mask2d = _dict_points(item.get("keypoint_2d") or item.get("keypont_2d"), 2)
        xyz, mask3d = _dict_points(item.get("keypoint_3d") or item.get("keypont_3d"), 3)
        xyc = np.concatenate([xy, mask2d[:, None].astype(np.float32)], axis=1)
        bbox = item.get("bbox", [0, 0, 1, 1])
        width = max(1, int(bbox[2] if len(bbox) == 4 else 1))
        height = max(1, int(bbox[3] if len(bbox) == 4 else 1))
        norm_2d = normalize_2d_image(xyc, (width, height))
        rel_3d, _ = make_root_relative(xyz)
        sample = {
            "history_2d": norm_2d[None, :, :],
            "target_3d": rel_3d,
            "target_mask": mask3d,
            "meta": {
                "source": "h3wb",
                "sample_id": sample_id,
                "frame_id": item.get("frame_id", sample_id),
                "image_path": item.get("image_path", ""),
            },
        }
        validate_sample(sample)
        samples.append(sample)
    return samples


def write_h3wb_cache(annotation_file: Path, out_file: Path) -> None:
    samples = load_h3wb_json(annotation_file)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        inputs_2d=np.stack([sample["history_2d"][0] for sample in samples]),
        targets_3d=np.stack([sample["target_3d"] for sample in samples]),
        target_masks=np.stack([sample["target_mask"] for sample in samples]),
        frame_ids=np.asarray([sample["meta"]["frame_id"] for sample in samples]),
        metas=np.asarray([sample["meta"] for sample in samples], dtype=object),
    )


class H3WBDataset(Dataset):
    def __init__(self, cache_file: Path, window: int) -> None:
        with np.load(cache_file, allow_pickle=True) as payload:
            self.inputs_2d = payload["inputs_2d"].astype(np.float32)
            self.targets_3d = payload["targets_3d"].astype(np.float32)
            self.target_masks = payload["target_masks"].astype(bool)
            self.frame_ids = payload["frame_ids"]
            self.metas = payload["metas"]
        self.window = int(window)

    def __len__(self) -> int:
        return self.inputs_2d.shape[0]

    def __getitem__(self, index: int) -> dict:
        sample = {
            "history_2d": make_causal_window(self.inputs_2d, index, self.window),
            "target_3d": self.targets_3d[index],
            "target_mask": self.target_masks[index],
            "meta": dict(self.metas[index]),
        }
        validate_sample(sample)
        return sample
