from pathlib import Path

import numpy as np

from mypose.data.coco_wholebody import CocoWholeBodyDataset, load_coco_wholebody_annotations


def test_load_coco_wholebody_annotations_pads_missing_optional_parts():
    samples = load_coco_wholebody_annotations(Path("tests/fixtures/coco_wholebody_tiny.json"))
    assert len(samples) == 1
    sample = samples[0]
    assert sample["keypoints_2d"].shape == (133, 3)
    assert sample["image_size"] == (640, 480)
    assert sample["meta"]["image_id"] == 7
    np.testing.assert_allclose(sample["keypoints_2d"][0], [0.0, 0.0, 0.0])


def test_dataset_returns_normalized_history_sample():
    dataset = CocoWholeBodyDataset(Path("tests/fixtures/coco_wholebody_tiny.json"))
    sample = dataset[0]
    assert sample["history_2d"].shape == (1, 133, 3)
    assert sample["meta"]["source"] == "coco-wholebody"
