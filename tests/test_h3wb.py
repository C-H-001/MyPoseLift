from pathlib import Path
import json

import numpy as np

from mypose.data.h3wb import H3WBDataset, load_h3wb_json, write_h3wb_cache


def _pose(frame_x: float, *, subject: str = "S1", action: str = "Walk", camera: str = "54138969", frame_idx: int = 1, omit_hips: bool = False) -> dict:
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
        "bbox": [0, 0, 640, 480],
        "subject": subject,
        "action": action,
        "camera": camera,
        "frame_idx": frame_idx,
        "keypoint_2d": {
            "0": {"x": frame_x, "y": 240},
            "11": {"x": 300, "y": 300},
            "12": {"x": 340, "y": 300},
        },
        "keypoint_3d": points_3d,
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


def test_load_h3wb_json_rejects_missing_hip_annotation(tmp_path):
    annotation_file = tmp_path / "missing_hip.json"
    annotation_file.write_text(json.dumps({"missing_hip": _pose(10, omit_hips=True)}), encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "valid left and right hip"):
        load_h3wb_json(annotation_file)
