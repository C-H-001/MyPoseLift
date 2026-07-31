from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from mypose.data.keypoints65 import (
    NUM_KEYPOINTS,
    remap_133_to_65,
    remap_mask_133_to_65,
)
from mypose.data.transforms import make_root_relative, normalize_2d_image
from mypose.data.validation import validate_sample


H36M_CAMERA_RESOLUTIONS = {
    "54138969": (1000, 1002),
    "55011271": (1000, 1000),
    "58860488": (1000, 1000),
    "60457274": (1000, 1002),
}


def _sequence_id(item: dict[str, Any]) -> str | None:
    for field in ("sequence_id", "video_id"):
        explicit = item.get(field)
        if explicit not in (None, ""):
            return str(explicit)
    fields = [item.get(name) for name in ("subject", "action", "camera")]
    if all(value not in (None, "") for value in fields):
        return "/".join(str(value) for value in fields)
    image_path = str(item.get("image_path", "")).replace("\\", "/")
    parent = image_path.rsplit("/", 1)[0] if "/" in image_path else ""
    return parent or None


def _frame_id(item: dict[str, Any]) -> object | None:
    for field in ("frame_id", "frame_idx", "frame_index"):
        value = item.get(field)
        if value not in (None, ""):
            return value
    image_path = str(item.get("image_path", "")).replace("\\", "/")
    stem = image_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if stem:
        try:
            return int(stem)
        except ValueError:
            return stem
    return None


def _positive_image_size(width: object, height: object, source: str) -> tuple[int, int]:
    try:
        size = (int(width), int(height))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must contain numeric image width and height") from exc
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"{source} must contain positive image width and height, got {size}")
    return size


def _image_size(item: dict[str, Any]) -> tuple[int, int]:
    for width_field, height_field in (
        ("image_width", "image_height"),
        ("width", "height"),
    ):
        if width_field in item or height_field in item:
            if width_field not in item or height_field not in item:
                raise ValueError(
                    f"{width_field} and {height_field} must be provided together"
                )
            return _positive_image_size(
                item[width_field], item[height_field], f"{width_field}/{height_field}"
            )

    image_size = item.get("image_size")
    if image_size is not None:
        if isinstance(image_size, dict):
            width = image_size.get("width", image_size.get("w"))
            height = image_size.get("height", image_size.get("h"))
        elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            width, height = image_size
        else:
            raise ValueError(
                "image_size must be [width, height] or a width/height mapping"
            )
        return _positive_image_size(width, height, "image_size")

    camera = item.get("camera", item.get("camera_id"))
    if camera is not None:
        camera_id = str(camera)
        for known_id, resolution in H36M_CAMERA_RESOLUTIONS.items():
            if camera_id == known_id or known_id in camera_id:
                return resolution

    raise ValueError(
        "cannot determine image width and height: provide image_width/image_height, "
        "width/height, image_size, or a recognized Human3.6M camera ID"
    )


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
    if not isinstance(payload, dict):
        raise ValueError("H3WB annotation JSON must be a sample-id mapping")
    samples = []
    for sample_id, item in payload.items():
        if not isinstance(item, dict):
            raise ValueError(f"H3WB sample {sample_id!r} must be an object")
        missing_fields = [
            field for field in ("keypoints_2d", "keypoints_3d") if field not in item
        ]
        if missing_fields:
            raise ValueError(
                f"H3WB sample {sample_id!r} missing official field(s): "
                f"{', '.join(missing_fields)}"
            )
        xy, mask2d = _dict_points(item["keypoints_2d"], 2)
        xyz, mask3d = _dict_points(item["keypoints_3d"], 3)
        if not mask3d[11] or not mask3d[12]:
            raise ValueError("target_3d requires valid left and right hip annotations at indices 11 and 12")
        xyc = np.concatenate([xy, mask2d[:, None].astype(np.float32)], axis=1)
        norm_2d = normalize_2d_image(remap_133_to_65(xyc), _image_size(item))
        rel_3d, _ = make_root_relative(remap_133_to_65(xyz))
        sample = {
            "history_2d": norm_2d[None, :, :],
            "target_3d": rel_3d,
            "target_mask": remap_mask_133_to_65(mask3d),
            "meta": {
                "source": "h3wb",
                "sample_id": sample_id,
                "frame_id": _frame_id(item),
                "sequence_id": _sequence_id(item),
                "image_path": item.get("image_path", ""),
                "subject": item.get("subject"),
                "action": item.get("action"),
                "camera": item.get("camera", item.get("camera_id")),
            },
        }
        validate_sample(sample)
        samples.append(sample)
    return samples


def load_h3wb_npz(path: Path) -> list[dict]:
    samples = []
    with np.load(path, allow_pickle=True) as payload:
        if "train_data" not in payload.files:
            raise ValueError("H3WB NPZ must contain official 'train_data'")
        train_data = payload["train_data"].item()
    if not isinstance(train_data, dict):
        raise ValueError("H3WB NPZ train_data must be a nested mapping")
    for subject, actions in train_data.items():
        for action, item in actions.items():
            if not isinstance(item, dict) or "frame_id" not in item:
                raise ValueError(f"H3WB sequence {subject}/{action} missing frame_id")
            frame_ids = item["frame_id"]
            camera_ids = [
                key for key, value in item.items()
                if isinstance(value, dict) and "pose_2d" in value and "camera_3d" in value
            ]
            if not camera_ids:
                raise ValueError(f"H3WB sequence {subject}/{action} has no camera data")
            for camera in sorted(camera_ids):
                camera_item = item[camera]
                pose_2d = np.asarray(camera_item["pose_2d"], dtype=np.float32)
                camera_3d = np.asarray(camera_item["camera_3d"], dtype=np.float32)
                if pose_2d.ndim != 3 or pose_2d.shape[1:] != (133, 2):
                    raise ValueError(
                        f"H3WB {subject}/{action}/{camera} pose_2d must have shape (N, 133, 2)"
                    )
                if camera_3d.shape != (pose_2d.shape[0], 133, 3):
                    raise ValueError(
                        f"H3WB {subject}/{action}/{camera} camera_3d must have shape (N, 133, 3)"
                    )
                if len(frame_ids) != pose_2d.shape[0]:
                    raise ValueError(
                        f"H3WB {subject}/{action}/{camera} frame count mismatch"
                    )
                width, height = _image_size({"camera": camera})
                sequence_id = f"{subject}/{action}/{camera}"
                sample_ids = camera_item.get("sample_id")
                for index in range(pose_2d.shape[0]):
                    mask = np.ones((133,), dtype=bool)
                    xyc = np.concatenate(
                        [pose_2d[index], np.ones((133, 1), dtype=np.float32)],
                        axis=1,
                    )
                    norm_2d = normalize_2d_image(
                        remap_133_to_65(xyc), (width, height)
                    )
                    rel_3d, _ = make_root_relative(
                        remap_133_to_65(camera_3d[index])
                    )
                    sample_id = (
                        int(sample_ids[index])
                        if sample_ids is not None and len(sample_ids) > index
                        else f"{sequence_id}/{frame_ids[index]}"
                    )
                    sample = {
                        "history_2d": norm_2d[None, :, :],
                        "target_3d": rel_3d,
                        "target_mask": remap_mask_133_to_65(mask),
                        "meta": {
                            "source": "h3wb_release_npz",
                            "sample_id": sample_id,
                            "frame_id": frame_ids[index],
                            "sequence_id": sequence_id,
                            "image_path": "",
                            "subject": subject,
                            "action": action,
                            "camera": camera,
                        },
                    }
                    validate_sample(sample)
                    samples.append(sample)
    return samples


def load_h3wb_annotations(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return load_h3wb_npz(path)
    return load_h3wb_json(path)


def _write_samples_cache(samples: list[dict], out_file: Path) -> None:
    if not samples:
        raise ValueError("cannot write an empty H3WB cache")
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


def write_h3wb_cache(annotation_file: Path, out_file: Path) -> None:
    _write_samples_cache(load_h3wb_annotations(annotation_file), out_file)


def _fold_group(sample: dict) -> tuple[str, ...]:
    meta = sample["meta"]
    subject = meta.get("subject")
    action = meta.get("action")
    if subject not in (None, "") and action not in (None, ""):
        return ("motion", str(subject), str(action))
    return ("sequence", str(meta["sequence_id"]))


def write_h3wb_fold_caches(
    annotation_file: Path,
    train_out: Path,
    val_out: Path,
    *,
    num_folds: int = 5,
    val_fold: int = 0,
) -> None:
    samples = load_h3wb_annotations(annotation_file)
    if any(sample["meta"]["sequence_id"] in (None, "") for sample in samples):
        raise ValueError("fold preparation requires sequence metadata for every sample")
    fold_groups = sorted({_fold_group(sample) for sample in samples})
    if num_folds < 2 or num_folds > len(fold_groups):
        raise ValueError(
            f"num_folds must be between 2 and the {len(fold_groups)} available motion groups"
        )
    if val_fold < 0 or val_fold >= num_folds:
        raise ValueError(f"val_fold must be in [0, {num_folds}), got {val_fold}")
    fold_by_group = {
        group: index % num_folds
        for index, group in enumerate(fold_groups)
    }
    train_samples = [
        sample
        for sample in samples
        if fold_by_group[_fold_group(sample)] != val_fold
    ]
    val_samples = [
        sample
        for sample in samples
        if fold_by_group[_fold_group(sample)] == val_fold
    ]
    _write_samples_cache(train_samples, train_out)
    _write_samples_cache(val_samples, val_out)


class H3WBDataset(Dataset):
    def __init__(self, cache_file: Path, window: int) -> None:
        self.cache_file = Path(cache_file)
        with np.load(self.cache_file, allow_pickle=True) as payload:
            self.inputs_2d = payload["inputs_2d"].astype(np.float32)
            self.targets_3d = payload["targets_3d"].astype(np.float32)
            self.target_masks = payload["target_masks"].astype(bool)
            self.metas = payload["metas"]
            self.frame_ids = (
                payload["frame_ids"]
                if "frame_ids" in payload.files
                else np.asarray([dict(meta).get("frame_id") for meta in self.metas], dtype=object)
            )
            self.sequence_ids = (
                payload["sequence_ids"]
                if "sequence_ids" in payload.files
                else np.asarray(
                    [dict(meta).get("sequence_id") for meta in self.metas], dtype=object
                )
            )
        if self.target_masks.ndim == 3 and self.target_masks.shape[-1] == 1:
            self.target_masks = self.target_masks[..., 0]
        sample_count = self.inputs_2d.shape[0]
        if (
            self.inputs_2d.shape[1:] != (NUM_KEYPOINTS, 3)
            or self.targets_3d.shape != (sample_count, NUM_KEYPOINTS, 3)
            or self.target_masks.shape != (sample_count, NUM_KEYPOINTS)
            or len(self.metas) != sample_count
            or len(self.frame_ids) != sample_count
            or len(self.sequence_ids) != sample_count
        ):
            raise ValueError("H3WB cache arrays have inconsistent sample or keypoint shapes")
        self.window = int(window)
        if self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}")
        self._ordered_by_sequence: dict[str, list[int]] = {}
        self._position_by_index: dict[int, int] = {}
        if self.window == 1:
            return
        has_temporal_metadata = all(
            value not in (None, "") for value in self.sequence_ids
        ) and all(value not in (None, "") for value in self.frame_ids)
        if not has_temporal_metadata:
            raise ValueError(
                "window > 1 requires explicit sequence and frame metadata for every sample"
            )
        for item_index, sequence_id in enumerate(self.sequence_ids):
            self._ordered_by_sequence.setdefault(str(sequence_id), []).append(item_index)
        for sequence_id, indices in self._ordered_by_sequence.items():
            indices.sort(key=lambda item_index: self._frame_sort_key(self.frame_ids[item_index]))
            frame_keys = [self._frame_sort_key(self.frame_ids[item_index]) for item_index in indices]
            if len(frame_keys) != len(set(frame_keys)):
                raise ValueError(f"sequence {sequence_id!r} contains duplicate frame metadata")
            self._position_by_index.update(
                {item_index: position for position, item_index in enumerate(indices)}
            )

    @staticmethod
    def _frame_sort_key(frame_id: object) -> tuple[int, object]:
        try:
            return (0, float(frame_id))
        except (TypeError, ValueError):
            return (1, str(frame_id))

    def __len__(self) -> int:
        return self.inputs_2d.shape[0]

    def __getitem__(self, index: int) -> dict:
        if self.window == 1:
            history = self.inputs_2d[index:index + 1]
        else:
            sequence_id = str(self.sequence_ids[index])
            ordered = self._ordered_by_sequence[sequence_id]
            position = self._position_by_index[index]
            history_indices = ordered[max(0, position - self.window + 1):position + 1]
            history = self.inputs_2d[history_indices]
            if history.shape[0] < self.window:
                history = np.concatenate(
                    [
                        np.repeat(history[:1], self.window - history.shape[0], axis=0),
                        history,
                    ],
                    axis=0,
                )
        sample = {
            "history_2d": history,
            "target_3d": self.targets_3d[index],
            "target_mask": self.target_masks[index],
            "meta": dict(self.metas[index]),
        }
        validate_sample(sample)
        return sample
