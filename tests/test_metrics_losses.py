import torch

from mypose.models.losses import WholeBodyLoss
from mypose.utils.metrics import aligned_mpjpe, mpjpe, part_mpjpe


def test_mpjpe_masks_invalid_points():
    pred = torch.zeros(1, 133, 3)
    target = torch.zeros(1, 133, 3)
    target[:, 0] = 10.0
    mask = torch.zeros(1, 133, dtype=torch.bool)
    assert mpjpe(pred, target, mask).item() == 0.0
    mask[:, 0] = True
    assert mpjpe(pred, target, mask).item() > 0.0


def test_part_mpjpe_uses_requested_indices():
    pred = torch.zeros(1, 133, 3)
    target = torch.zeros(1, 133, 3)
    target[:, 91:112] = 2.0
    assert part_mpjpe(pred, target, "left_hand").item() > 0.0
    assert part_mpjpe(pred, target, "body").item() == 0.0


def test_aligned_mpjpe_removes_anchor_translation():
    pred = torch.zeros(1, 133, 3)
    target = torch.zeros(1, 133, 3)
    pred[:, 91:112] = 5.0
    target[:, 91:112] = 7.0
    assert aligned_mpjpe(pred, target, list(range(91, 112)), anchor_index=91).item() == 0.0


def test_wholebody_loss_hand_weight_changes_total_loss():
    pred = torch.zeros(1, 133, 3, requires_grad=True)
    target = torch.zeros(1, 133, 3)
    target[:, 91:112] = 1.0
    mask = torch.ones(1, 133, dtype=torch.bool)
    low = WholeBodyLoss(part_weights={"left_hand": 1.0})
    high = WholeBodyLoss(part_weights={"left_hand": 10.0})
    assert high(pred, target, mask)["total"].item() > low(pred, target, mask)["total"].item()
