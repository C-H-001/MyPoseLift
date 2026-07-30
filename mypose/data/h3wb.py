from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from mypose.data.transforms import make_root_relative, normalize_2d_image
from mypose.data.validation import validate_sample


def _sequence_id(item: dict[str, Any], sample_id: str) -> str:
    explicit = item.get("sequence_id")
    if explicit is not None:
        return str(explicit)
    fields = [item.get(name) for name in ("subject", "action", "camera")]
    if all(value is not None for value in fields):
        return "/".join(str(value) for value in fields)
    image_path = str(item.get("image_path", "")).replace("\\", "/")
    parent = image_path.rsplit("/", 1)[0] if "/" in image_path else ""
    return parent or str(sample_id)


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
        if not mask3d[11] or not mask3d[12]:
            raise ValueError("target_3d requires valid left and right hip annotations at indices 11 and 12")
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
                "frame_id": item.get("frame_id", item.get("frame_idx", sample_id)),
                "sequence_id": _sequence_id(item, sample_id),
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
        sequence_ids=np.asarray([sample["meta"]["sequence_id"] for sample in samples]),
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
            if "sequence_ids" in payload.files:
                self.sequence_ids = payload["sequence_ids"]
            else:
                self.sequence_ids = np.asarray([dict(meta).get("sequence_id", "") for meta in self.metas])
        self.window = int(window)
        if self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}")
        self._ordered_by_sequence = {}
        for item_index, sequence_id in enumerate(self.sequence_ids):
            self._ordered_by_sequence.setdefault(str(sequence_id), []).append(item_index)
        for sequence_id, indices in self._ordered_by_sequence.items():
            indices.sort(key=lambda item_index: self._frame_sort_key(self.frame_ids[item_index]))

    @staticmethod
    def _frame_sort_key(frame_id: object) -> tuple[int, object]:
        try:
            return (0, float(frame_id))
        except (TypeError, ValueError):
            return (1, str(frame_id))

    def __len__(self) -> int:
        return self.inputs_2d.shape[0]

    def __getitem__(self, index: int) -> dict:
        sequence_id = str(self.sequence_ids[index])
        ordered = self._ordered_by_sequence[sequence_id]
        current_key = self._frame_sort_key(self.frame_ids[index])
        eligible = [item_index for item_index in ordered if self._frame_sort_key(self.frame_ids[item_index]) <= current_key]
        history_indices = eligible[-self.window:]
        history = self.inputs_2d[history_indices]
        if history.shape[0] < self.window:
            history = np.concatenate([np.repeat(history[:1], self.window - history.shape[0], axis=0), history], axis=0)
        sample = {
            "history_2d": history,
            "target_3d": self.targets_3d[index],
            "target_mask": self.target_masks[index],
            "meta": dict(self.metas[index]),
        }
        validate_sample(sample)
        return sample
