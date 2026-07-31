from pathlib import Path
import json

import numpy as np
import pytest

from mypose.data.h3wb import (
    H3WBDataset,
    load_h3wb_json,
    write_h3wb_cache,
    write_h3wb_fold_caches,
)


def _pose(
    frame_x: float,
    *,
    subject: str = "S1",
    action: str = "Walk",
    camera: str = "54138969",
    frame_idx: int = 1,
    omit_hips: bool = False,
) -> dict:
    points_3d = {
        str(index): {"x": float(index), "y": float(index + 100), "z": float(index + 1000)}
        for index in range(68)
        if index != 25 and not (omit_hips and index == 11)
    }
    points_3d["11"] = {"x": -100.0, "y": 0.0, "z": 1000.0}
    points_3d["12"] = {"x": 100.0, "y": 0.0, "z": 1000.0}
    if omit_hips:
        points_3d.pop("11")
    return {
        "image_path": f"{subject}/{action}/{camera}/{frame_idx:06d}.jpg",
        "image_width": 640,
        "image_height": 480,
        "bbox": [50, 60, 100, 120],
        "subject": subject,
        "action": action,
        "camera": camera,
        "frame_idx": frame_idx,
        "keypoints_2d": {
            "0": {"x": frame_x, "y": 240},
            "11": {"x": 300, "y": 300},
            "12": {"x": 340, "y": 300},
        },
        "keypoints_3d": points_3d,
    }


def test_load_h3wb_json_fills_missing_points_and_root_relative_target():
    samples = load_h3wb_json(Path("tests/fixtures/h3wb_tiny.json"))
    assert len(samples) == 1
    sample = samples[0]
    assert sample["history_2d"].shape == (1, 133, 3)
    assert sample["target_3d"].shape == (133, 3)
    np.testing.assert_allclose((sample["target_3d"][11] + sample["target_3d"][12]) / 2.0, [0.0, 0.0, 0.0])
    assert sample["target_mask"][0]
    assert not sample["target_mask"][25]
    np.testing.assert_allclose(sample["history_2d"][0, 0, :2], [0.0, 0.0])


@pytest.mark.parametrize("field", ["keypoints_2d", "keypoints_3d"])
def test_load_h3wb_json_requires_official_keypoint_fields(tmp_path, field):
    item = _pose(320)
    item.pop(field)
    annotation_file = tmp_path / "missing_field.json"
    annotation_file.write_text(json.dumps({"sample": item}), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_h3wb_json(annotation_file)


def test_load_h3wb_json_uses_known_h36m_camera_resolution_without_image_metadata(tmp_path):
    item = _pose(500)
    item.pop("image_width")
    item.pop("image_height")
    item["keypoints_2d"]["0"]["y"] = 501
    annotation_file = tmp_path / "camera_resolution.json"
    annotation_file.write_text(json.dumps({"sample": item}), encoding="utf-8")

    sample = load_h3wb_json(annotation_file)[0]

    np.testing.assert_allclose(sample["history_2d"][0, 0, :2], [0.0, 0.0])


def test_load_h3wb_json_rejects_missing_normalization_basis(tmp_path):
    item = _pose(320)
    for field in ("image_width", "image_height", "camera", "image_path"):
        item.pop(field)
    annotation_file = tmp_path / "missing_image_size.json"
    annotation_file.write_text(json.dumps({"sample": item}), encoding="utf-8")

    with pytest.raises(ValueError, match="image width and height"):
        load_h3wb_json(annotation_file)


def test_h3wb_dataset_reads_npz_cache_with_causal_window(tmp_path):
    cache = tmp_path / "h3wb.npz"
    write_h3wb_cache(Path("tests/fixtures/h3wb_tiny.json"), cache)
    with np.load(cache, allow_pickle=True) as payload:
        assert {"inputs_2d", "targets_3d", "target_masks", "frame_ids", "sequence_ids", "metas"} <= set(payload.files)
    dataset = H3WBDataset(cache, window=3)
    sample = dataset[0]
    assert sample["history_2d"].shape == (3, 133, 3)
    assert sample["target_3d"].shape == (133, 3)


def test_h3wb_dataset_windows_are_sequence_local_and_sorted_by_frame(tmp_path):
    annotation_file = tmp_path / "interleaved.json"
    annotation_file.write_text(
        json.dumps({
            "a_frame_2": _pose(12, frame_idx=2),
            "b_frame_1": _pose(102, subject="S2", frame_idx=1),
            "a_frame_1": _pose(11, frame_idx=1),
            "b_frame_2": _pose(112, subject="S2", frame_idx=2),
        }),
        encoding="utf-8",
    )
    cache = tmp_path / "interleaved.npz"
    write_h3wb_cache(annotation_file, cache)
    dataset = H3WBDataset(cache, window=3)

    a_frame_2 = dataset[0]["history_2d"][:, 0, 0]
    b_frame_1 = dataset[1]["history_2d"][:, 0, 0]
    np.testing.assert_allclose(a_frame_2, [(11 / 640 - 0.5) * 2, (11 / 640 - 0.5) * 2, (12 / 640 - 0.5) * 2])
    np.testing.assert_allclose(b_frame_1, [(102 / 640 - 0.5) * 2] * 3)


def test_h3wb_dataset_t27_uses_prior_frames_from_same_sequence(tmp_path):
    annotation_file = tmp_path / "sequence.json"
    items = {
        f"frame_{frame_idx:02d}": _pose(float(frame_idx), frame_idx=frame_idx)
        for frame_idx in range(30)
    }
    annotation_file.write_text(json.dumps(items), encoding="utf-8")
    cache = tmp_path / "sequence.npz"
    write_h3wb_cache(annotation_file, cache)

    history = H3WBDataset(cache, window=27)[26]["history_2d"][:, 0, 0]

    expected = np.asarray([(frame_idx / 640 - 0.5) * 2 for frame_idx in range(27)])
    np.testing.assert_allclose(history, expected)
    assert np.unique(history).size == 27


def test_h3wb_dataset_allows_unordered_cache_for_window_one(tmp_path):
    cache = tmp_path / "unordered.npz"
    np.savez_compressed(
        cache,
        inputs_2d=np.zeros((1, 133, 3), dtype=np.float32),
        targets_3d=np.zeros((1, 133, 3), dtype=np.float32),
        target_masks=np.ones((1, 133), dtype=bool),
        metas=np.asarray([{"source": "synthetic"}], dtype=object),
    )

    assert H3WBDataset(cache, window=1)[0]["history_2d"].shape == (1, 133, 3)


def test_h3wb_dataset_requires_sequence_and_frame_metadata_for_temporal_window(tmp_path):
    cache = tmp_path / "unordered.npz"
    np.savez_compressed(
        cache,
        inputs_2d=np.zeros((1, 133, 3), dtype=np.float32),
        targets_3d=np.zeros((1, 133, 3), dtype=np.float32),
        target_masks=np.ones((1, 133), dtype=bool),
        metas=np.asarray([{"source": "synthetic"}], dtype=object),
    )

    with pytest.raises(ValueError, match="sequence and frame metadata"):
        H3WBDataset(cache, window=27)


def test_load_h3wb_json_rejects_missing_hip_annotation(tmp_path):
    annotation_file = tmp_path / "missing_hip.json"
    annotation_file.write_text(json.dumps({"missing_hip": _pose(10, omit_hips=True)}), encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "valid left and right hip"):
        load_h3wb_json(annotation_file)


def test_write_h3wb_fold_caches_is_deterministic_and_sequence_disjoint(tmp_path):
    annotation_file = tmp_path / "folds.json"
    annotation_file.write_text(
        json.dumps(
            {
                f"{subject}_frame": _pose(320, subject=subject)
                for subject in ("S1", "S2", "S3", "S4")
            }
        ),
        encoding="utf-8",
    )
    train_cache = tmp_path / "train.npz"
    val_cache = tmp_path / "val.npz"
    write_h3wb_fold_caches(
        annotation_file,
        train_cache,
        val_cache,
        num_folds=2,
        val_fold=0,
    )

    with np.load(train_cache, allow_pickle=True) as train_payload:
        train_sequences = set(train_payload["sequence_ids"].tolist())
    with np.load(val_cache, allow_pickle=True) as val_payload:
        val_sequences = set(val_payload["sequence_ids"].tolist())

    assert train_sequences
    assert val_sequences
    assert train_sequences.isdisjoint(val_sequences)
    assert train_sequences | val_sequences == {
        f"{subject}/Walk/54138969" for subject in ("S1", "S2", "S3", "S4")
    }

    second_train = tmp_path / "train_again.npz"
    second_val = tmp_path / "val_again.npz"
    write_h3wb_fold_caches(
        annotation_file,
        second_train,
        second_val,
        num_folds=2,
        val_fold=0,
    )
    with np.load(second_val, allow_pickle=True) as second_payload:
        assert set(second_payload["sequence_ids"].tolist()) == val_sequences
