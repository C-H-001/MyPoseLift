import ast
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from mypose.data.h3wb import H3WBDataset
from mypose.engine.checkpoint import load_checkpoint, save_checkpoint
from mypose.engine.evaluate import evaluate, main as evaluate_main
from mypose.engine.train import build_model_from_config, train_from_config
from mypose.models.causal_tcn_lifter import CausalTCNLifter
from mypose.models.hrgcn_lifter import HRGCNLifter


class TinyPoseDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "history_2d": torch.zeros(1, 65, 3),
            "target_3d": torch.zeros(65, 3),
            "target_mask": torch.ones(65, dtype=torch.bool),
            "meta": {"index": index},
        }


def test_checkpoint_roundtrip(tmp_path):
    model = HRGCNLifter(hidden_channels=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, epoch=3, metrics={"MPJPE_whole": 1.5})
    state = load_checkpoint(path, model, optimizer)
    assert state["epoch"] == 3
    assert state["metrics"]["MPJPE_whole"] == 1.5


def test_evaluate_returns_part_metrics():
    loader = DataLoader(TinyPoseDataset(), batch_size=2)
    model = IdentityPoseModel()
    metrics = evaluate(model, loader, device=torch.device("cpu"))
    assert "MPJPE_whole" in metrics
    assert "MPJPE_hands_wrist_aligned" in metrics


class UnevenMetricDataset:
    def __len__(self):
        return 3

    def __getitem__(self, index):
        error = 1.0 if index < 2 else 3.0
        history = torch.zeros(1, 65, 3)
        history[..., 0] = error
        mask = torch.zeros(65, dtype=torch.bool)
        mask[: 2 if index == 0 else 1] = True
        return {
            "history_2d": history,
            "target_3d": torch.zeros(65, 3),
            "target_mask": mask,
        }


class IdentityPoseModel(torch.nn.Module):
    def forward(self, history_2d):
        return history_2d[:, -1]


def test_evaluate_weights_batches_by_valid_points():
    loader = DataLoader(UnevenMetricDataset(), batch_size=2)
    metrics = evaluate(IdentityPoseModel(), loader, device=torch.device("cpu"))
    assert metrics["MPJPE_whole"] == pytest.approx(6.0 / 4.0)


class Head3Dataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        history = torch.zeros(1, 65, 3)
        history[:, :3, 0] = 2.0
        return {
            "history_2d": history,
            "target_3d": torch.zeros(65, 3),
            "target_mask": torch.ones(65, dtype=torch.bool),
        }


def test_evaluate_reports_head3_metric():
    loader = DataLoader(Head3Dataset(), batch_size=1)

    metrics = evaluate(IdentityPoseModel(), loader, device=torch.device("cpu"))

    assert metrics["MPJPE_head3"] == 2.0


def _write_pose_cache(path: Path, target_offset: float) -> None:
    target = np.full((1, 65, 3), target_offset, dtype=np.float32)
    target[:, 11] = [-1.0, 0.0, 0.0]
    target[:, 12] = [1.0, 0.0, 0.0]
    np.savez_compressed(
        path,
        inputs_2d=np.zeros((1, 65, 3), dtype=np.float32),
        targets_3d=target,
        target_masks=np.ones((1, 65), dtype=bool),
        metas=np.asarray([{"source": "synthetic"}], dtype=object),
    )


def _training_config(tmp_path: Path) -> dict:
    train_cache = tmp_path / "train.npz"
    val_cache = tmp_path / "val.npz"
    _write_pose_cache(train_cache, target_offset=0.0)
    _write_pose_cache(val_cache, target_offset=10.0)
    return {
        "data": {
            "train_cache": str(train_cache),
            "val_cache": str(val_cache),
            "window": 1,
        },
        "model": {
            "type": "hrgcn",
            "hidden_channels": 4,
            "use_temporal": False,
            "temporal_kernel_size": 1,
            "temporal_dilation": 1,
        },
        "loss": {
            "part_weights": {},
            "local_weights": {},
        },
        "train": {
            "device": "cpu",
            "batch_size": 1,
            "num_workers": 0,
            "lr": 0.0,
            "weight_decay": 0.0,
            "epochs": 1,
            "seed": 7,
            "out_dir": str(tmp_path / "checkpoints"),
        },
    }


class TinyTrainablePoseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(()))

    def forward(self, history_2d):
        return history_2d[:, -1] + self.offset


def _use_tiny_training_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "mypose.engine.train.build_model_from_config",
        lambda cfg: TinyTrainablePoseModel(),
    )


def test_training_evaluates_validation_cache_and_saves_last_and_best(
    tmp_path, monkeypatch
):
    cfg = _training_config(tmp_path)
    _use_tiny_training_model(monkeypatch)

    train_from_config(cfg)

    out_dir = Path(cfg["train"]["out_dir"])
    assert (out_dir / "last.pt").is_file()
    assert (out_dir / "best.pt").is_file()
    model = TinyTrainablePoseModel()
    state = load_checkpoint(out_dir / "last.pt", model)
    val_metrics = evaluate(
        model,
        DataLoader(H3WBDataset(Path(cfg["data"]["val_cache"]), window=1)),
        torch.device("cpu"),
    )
    train_metrics = evaluate(
        model,
        DataLoader(H3WBDataset(Path(cfg["data"]["train_cache"]), window=1)),
        torch.device("cpu"),
    )
    assert state["metrics"]["MPJPE_whole"] == pytest.approx(
        val_metrics["MPJPE_whole"]
    )
    assert state["metrics"]["MPJPE_whole"] != pytest.approx(
        train_metrics["MPJPE_whole"]
    )


def test_training_resume_continues_after_checkpoint_epoch(tmp_path, monkeypatch):
    cfg = _training_config(tmp_path)
    _use_tiny_training_model(monkeypatch)
    train_from_config(cfg)
    checkpoint = Path(cfg["train"]["out_dir"]) / "last.pt"
    cfg["train"]["epochs"] = 2

    train_from_config(cfg, resume=checkpoint)

    model = TinyTrainablePoseModel()
    state = load_checkpoint(checkpoint, model)
    assert state["epoch"] == 1


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/h3wb_hrgcn_t1.yaml"),
        Path("configs/h3wb_hrgcn_causal_t27.yaml"),
        Path("configs/h3wb_tcn_t81.yaml"),
        Path("configs/h3wb_tcn_t27.yaml"),
    ],
)
def test_configs_use_separate_train_and_validation_caches(config_path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert cfg["data"]["train_cache"] != cfg["data"]["val_cache"]


def test_t27_config_builds_27_frame_temporal_adapter():
    cfg = yaml.safe_load(
        Path("configs/h3wb_hrgcn_causal_t27.yaml").read_text(encoding="utf-8")
    )
    model = HRGCNLifter(
        hidden_channels=4,
        use_temporal=cfg["model"]["use_temporal"],
        temporal_kernel_size=cfg["model"]["temporal_kernel_size"],
        temporal_dilation=cfg["model"]["temporal_dilation"],
    )

    assert model.temporal.receptive_field == cfg["data"]["window"] == 27


@pytest.mark.parametrize(
    ("config_path", "window", "out_dir"),
    [
        (
            Path("configs/h3wb_tcn_t81.yaml"),
            81,
            "checkpoints/h3wb_tcn_t81",
        ),
        (
            Path("configs/h3wb_tcn_t27.yaml"),
            27,
            "checkpoints/h3wb_tcn_t27",
        ),
    ],
)
def test_tcn_configs_build_causal_model(config_path, window, out_dir):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model = build_model_from_config(cfg)

    assert isinstance(model, CausalTCNLifter)
    assert cfg["model"]["type"] == "causal_tcn"
    assert cfg["data"]["window"] == window
    assert cfg["train"]["out_dir"] == out_dir


def test_model_factory_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="unknown model type"):
        build_model_from_config({"model": {"type": "transformer"}})


def test_evaluate_cli_defaults_to_validation_cache(tmp_path, monkeypatch, capsys):
    cfg = _training_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    model = TinyTrainablePoseModel()
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(checkpoint, model, optimizer, epoch=0, metrics={})
    expected = evaluate(
        model,
        DataLoader(H3WBDataset(Path(cfg["data"]["val_cache"]), window=1)),
        torch.device("cpu"),
    )
    monkeypatch.setattr(
        "mypose.engine.evaluate.build_model_from_config",
        lambda cfg: TinyTrainablePoseModel(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mypose.engine.evaluate",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
        ],
    )

    evaluate_main()

    actual = ast.literal_eval(capsys.readouterr().out.strip())
    assert actual["MPJPE_whole"] == pytest.approx(expected["MPJPE_whole"])
