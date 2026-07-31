import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from mypose.engine import build_model_from_config
from mypose.engine.checkpoint import save_checkpoint


def _tiny_config(tmp_path: Path, cache: Path, out_dir: Path) -> Path:
    config = {
        "data": {
            "train_cache": str(cache),
            "val_cache": str(cache),
            "window": 3,
        },
        "model": {
            "type": "causal_tcn",
            "hidden_channels": 16,
            "num_blocks": 1,
            "kernel_size": 3,
            "dropout": 0.0,
        },
        "loss": {
            "part_weights": {
                "body": 1.0,
                "foot": 1.5,
                "head3": 1.0,
                "left_hand": 2.5,
                "right_hand": 2.5,
            },
            "local_weights": {
                "left_hand": 1.0,
                "right_hand": 1.0,
            },
        },
        "train": {
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "lr": 0.001,
            "weight_decay": 0.0,
            "epochs": 1,
            "seed": 0,
            "out_dir": str(out_dir),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _tiny_cache(tmp_path: Path) -> Path:
    inputs = np.zeros((4, 65, 3), dtype=np.float32)
    targets = np.zeros((4, 65, 3), dtype=np.float32)
    masks = np.ones((4, 65), dtype=bool)
    for index in range(4):
        inputs[index, :, 0] = np.linspace(-1.0, 1.0, 65) + index * 0.01
        inputs[index, :, 1] = np.linspace(1.0, -1.0, 65)
        inputs[index, :, 2] = 1.0
        targets[index, :, 0] = np.linspace(-100.0, 100.0, 65)
        targets[index, :, 1] = index * 10.0
        targets[index, :, 2] = np.linspace(50.0, -50.0, 65)
        targets[index, 11] = 0.0
        targets[index, 12] = 0.0
    metas = np.asarray(
        [
            {
                "sequence_id": "S1/Walking/CameraA",
                "frame_id": str(index),
                "subject": "S1",
                "action": "Walking",
                "camera": "CameraA",
            }
            for index in range(4)
        ],
        dtype=object,
    )
    cache = tmp_path / "cache.npz"
    np.savez_compressed(
        cache,
        inputs_2d=inputs,
        targets_3d=targets,
        target_masks=masks,
        frame_ids=np.asarray([str(index) for index in range(4)], dtype=object),
        sequence_ids=np.asarray(["S1/Walking/CameraA"] * 4, dtype=object),
        metas=metas,
    )
    return cache


def _tiny_checkpoint(config_path: Path, tmp_path: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_model_from_config(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer=optimizer,
        epoch=0,
        metrics={"MPJPE_whole": 0.0},
        best_metric=0.0,
    )
    return checkpoint


def test_export_sequence_viewer_help_runs():
    result = subprocess.run(
        [sys.executable, "tools/export_sequence_viewer.py", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--frames-per-sequence" in result.stdout


def test_export_sequence_viewer_writes_slider_html(tmp_path):
    cache = _tiny_cache(tmp_path)
    config = _tiny_config(tmp_path, cache, tmp_path / "checkpoints")
    checkpoint = _tiny_checkpoint(config, tmp_path)
    out = tmp_path / "viewer.html"

    subprocess.run(
        [
            sys.executable,
            "tools/export_sequence_viewer.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--cache",
            str(cache),
            "--num-sequences",
            "1",
            "--frames-per-sequence",
            "2",
            "--out",
            str(out),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    html = out.read_text(encoding="utf-8")
    assert 'type="range"' in html
    assert "Ground truth" in html
    assert "Prediction" in html
    assert "sequenceViewerData" in html
    assert "S1/Walking/CameraA" in html
