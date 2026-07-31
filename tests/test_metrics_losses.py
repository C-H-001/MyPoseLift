import pytest
import torch

from mypose.engine.evaluate import evaluate
from mypose.models.losses import WholeBodyLoss
from mypose.utils.metrics import aligned_mpjpe, mpjpe, part_mpjpe


def test_mpjpe_masks_invalid_points():
    pred = torch.zeros(1, 65, 3)
    target = torch.zeros(1, 65, 3)
    target[:, 0] = 10.0
    mask = torch.zeros(1, 65, dtype=torch.bool)
    assert mpjpe(pred, target, mask).item() == 0.0
    mask[:, 0] = True
    assert mpjpe(pred, target, mask).item() > 0.0


def test_part_mpjpe_uses_requested_indices():
    pred = torch.zeros(1, 65, 3)
    target = torch.zeros(1, 65, 3)
    target[:, 23:44] = 2.0
    assert part_mpjpe(pred, target, "left_hand").item() > 0.0
    assert part_mpjpe(pred, target, "body").item() == 0.0


def test_aligned_mpjpe_removes_anchor_translation():
    pred = torch.zeros(1, 65, 3)
    target = torch.zeros(1, 65, 3)
    pred[:, 23:44] = 5.0
    target[:, 23:44] = 7.0
    assert aligned_mpjpe(pred, target, list(range(23, 44)), anchor_index=23).item() == 0.0


def test_aligned_mpjpe_requires_valid_anchor():
    pred = torch.ones(1, 65, 3)
    target = torch.zeros(1, 65, 3)
    pred[:, 23:44] = 2.0
    target[:, 23:44] = 4.0
    pred[:, 30, :] = 12.0
    target[:, 30, :] = 1.0
    mask = torch.ones(1, 65, dtype=torch.bool)
    mask[:, 30] = False
    assert aligned_mpjpe(pred, target, list(range(23, 44)), anchor_index=30, mask=mask).item() == 0.0


def test_wholebody_loss_hand_weight_changes_total_loss():
    pred = torch.zeros(1, 65, 3, requires_grad=True)
    target = torch.zeros(1, 65, 3)
    target[:, 23:44] = 1.0
    mask = torch.ones(1, 65, dtype=torch.bool)
    low = WholeBodyLoss(part_weights={"left_hand": 1.0})
    high = WholeBodyLoss(part_weights={"left_hand": 10.0})
    assert high(pred, target, mask)["total"].item() > low(pred, target, mask)["total"].item()


def test_boneloss_respects_target_mask():
    pred = torch.tensor([[[3.0, 0.0, 0.0]] * 65], dtype=torch.float32)
    target = torch.zeros(1, 65, 3, dtype=torch.float32)
    mask = torch.zeros(1, 65, dtype=torch.bool)
    losses = WholeBodyLoss()(pred, target, mask)
    assert losses["bone"].item() == 0.0
    assert losses["total"].item() == 0.0


def test_metrics_and_loss_canonicalize_batched_singleton_mask():
    pred = torch.zeros(2, 65, 3)
    target = torch.zeros(2, 65, 3)
    mask = torch.ones(2, 65, 1, dtype=torch.bool)

    assert mpjpe(pred, target, mask).item() == 0.0
    assert WholeBodyLoss()(pred, target, mask)["total"].item() == 0.0


def test_hand_local_losses_align_to_compact_wrist_indices():
    pred = torch.zeros(1, 65, 3)
    target = torch.zeros(1, 65, 3)
    pred[:, 23:44, 0] = 5.0
    pred[:, 44:65, 0] = 7.0
    pred[:, 23, 0] = 2.0
    pred[:, 44, 0] = 3.0

    loss = WholeBodyLoss()(pred, target, torch.ones(1, 65, dtype=torch.bool))

    assert loss["left_hand_local"].item() == pytest.approx(60.0 / 21.0)
    assert loss["right_hand_local"].item() == pytest.approx(80.0 / 21.0)


def test_whole_body_loss_uses_65_point_parts_without_dense_face():
    pred = torch.zeros(2, 65, 3)
    target = torch.ones(2, 65, 3)
    mask = torch.ones(2, 65, dtype=torch.bool)

    losses = WholeBodyLoss()(pred, target, mask)

    assert "head3_mpjpe" in losses
    assert "face_mpjpe" not in losses
    assert "face_local" not in losses


class _LastFrameModel(torch.nn.Module):
    def forward(self, history_2d):
        return history_2d[:, -1]


def test_evaluate_reports_65_point_metrics():
    batch = {
        "history_2d": torch.zeros(2, 1, 65, 3),
        "target_3d": torch.zeros(2, 65, 3),
        "target_mask": torch.ones(2, 65, dtype=torch.bool),
    }

    metrics = evaluate(_LastFrameModel(), [batch], torch.device("cpu"))

    assert set(metrics) == {
        "MPJPE_whole",
        "MPJPE_body",
        "MPJPE_feet",
        "MPJPE_head3",
        "MPJPE_left_hand",
        "MPJPE_right_hand",
        "MPJPE_hands_wrist_aligned",
    }
    assert "MPJPE_face" not in metrics
    assert "MPJPE_face_nose_aligned" not in metrics
