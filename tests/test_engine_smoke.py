import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mypose.engine.checkpoint import load_checkpoint, save_checkpoint
from mypose.engine.evaluate import evaluate
from mypose.models.hrgcn_lifter import HRGCNLifter


class TinyPoseDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "history_2d": torch.zeros(1, 133, 3),
            "target_3d": torch.zeros(133, 3),
            "target_mask": torch.ones(133, dtype=torch.bool),
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
    model = HRGCNLifter(hidden_channels=16)
    metrics = evaluate(model, loader, device=torch.device("cpu"))
    assert "MPJPE_whole" in metrics
    assert "MPJPE_hands_wrist_aligned" in metrics


class UnevenMetricDataset:
    def __len__(self):
        return 3

    def __getitem__(self, index):
        error = 1.0 if index < 2 else 3.0
        history = torch.zeros(1, 133, 3)
        history[..., 0] = error
        mask = torch.zeros(133, dtype=torch.bool)
        mask[: 2 if index == 0 else 1] = True
        return {
            "history_2d": history,
            "target_3d": torch.zeros(133, 3),
            "target_mask": mask,
        }


class IdentityPoseModel(torch.nn.Module):
    def forward(self, history_2d):
        return history_2d[:, -1]


def test_evaluate_weights_batches_by_valid_points():
    loader = DataLoader(UnevenMetricDataset(), batch_size=2)
    metrics = evaluate(IdentityPoseModel(), loader, device=torch.device("cpu"))
    assert metrics["MPJPE_whole"] == pytest.approx(6.0 / 4.0)
