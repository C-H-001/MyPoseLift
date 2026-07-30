from pathlib import Path

import numpy as np

from mypose.data.h3wb import H3WBDataset, load_h3wb_json, write_h3wb_cache


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
        assert {"inputs_2d", "targets_3d", "target_masks", "frame_ids", "metas"} <= set(payload.files)
    dataset = H3WBDataset(cache, window=3)
    sample = dataset[0]
    assert sample["history_2d"].shape == (3, 133, 3)
    assert sample["target_3d"].shape == (133, 3)
