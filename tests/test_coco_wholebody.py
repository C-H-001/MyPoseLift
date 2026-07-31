from pathlib import Path

import numpy as np
import pytest

from mypose.data.coco_wholebody import CocoWholeBodyDataset, load_coco_wholebody_annotations
from tools import download_datasets


def test_load_coco_wholebody_annotations_pads_missing_optional_parts():
    samples = load_coco_wholebody_annotations(Path("tests/fixtures/coco_wholebody_tiny.json"))
    assert len(samples) == 1
    sample = samples[0]
    assert sample["keypoints_2d"].shape == (65, 3)
    assert sample["image_size"] == (640, 480)
    assert sample["meta"]["image_id"] == 7
    np.testing.assert_allclose(sample["keypoints_2d"][0], [0.0, 0.0, 0.0])


def test_dataset_returns_normalized_history_sample():
    dataset = CocoWholeBodyDataset(Path("tests/fixtures/coco_wholebody_tiny.json"))
    sample = dataset[0]
    assert sample["history_2d"].shape == (1, 65, 3)
    assert sample["target_3d"].shape == (65, 3)
    assert sample["target_mask"].shape == (65,)
    assert sample["meta"]["source"] == "coco-wholebody"


def test_coco_downloader_requires_expected_annotation_files(tmp_path, capsys):
    with pytest.raises(FileNotFoundError, match="missing COCO-WholeBody annotation files"):
        download_datasets.download_coco_wholebody(tmp_path, with_images=False, annotation_source="manual")

    output = capsys.readouterr().out
    assert "https://github.com/jin-s13/COCO-WholeBody#download" in output
    assert (tmp_path / "annotations").is_dir()


def test_coco_openxlab_download_builds_official_command(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, check):
        calls.append((command, check))
        annotations = tmp_path / "annotations"
        annotations.mkdir(exist_ok=True)
        for filename in download_datasets.COCO_ANNOTATION_FILENAMES:
            (annotations / filename).write_text('{"images": [], "annotations": []}', encoding="utf-8")

    monkeypatch.setattr(download_datasets.shutil, "which", lambda name: "openxlab.exe")
    monkeypatch.setattr(download_datasets.subprocess, "run", fake_run)
    download_datasets.download_coco_wholebody(tmp_path, with_images=False, annotation_source="openxlab")

    assert calls == [([
        "openxlab.exe",
        "dataset",
        "get",
        "--dataset-repo",
        "OpenDataLab/COCO-WholeBody",
        "--target-path",
        str(tmp_path),
    ], True)]


def test_coco_openxlab_missing_cli_explains_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(download_datasets.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="pip install openxlab.*openxlab login.*OpenDataLab/COCO-WholeBody"):
        download_datasets.download_coco_wholebody(tmp_path, with_images=False, annotation_source="openxlab")


def test_h3wb_downloader_accepts_nonempty_expected_annotation_files(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for filename in download_datasets.H3WB_ANNOTATION_FILENAMES:
        (annotations / filename).write_text("{}", encoding="utf-8")

    download_datasets.download_h3wb(tmp_path)


def test_download_file_replaces_destination_only_after_stream_completes(tmp_path, monkeypatch):
    destination = tmp_path / "archive.zip"

    class Response:
        headers = {"content-length": "7"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"partial"
            raise RuntimeError("stream interrupted")

    monkeypatch.setattr(download_datasets.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="stream interrupted"):
        download_datasets.download_file("https://example.test/archive.zip", destination)

    assert not destination.exists()
    assert not destination.with_name("archive.zip.part").exists()
