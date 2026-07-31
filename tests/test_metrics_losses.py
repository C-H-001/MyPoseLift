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


def test_aligned_mpjpe_requires_valid_anchor():
    pred = torch.ones(1, 133, 3)
    target = torch.zeros(1, 133, 3)
    pred[:, 23:91] = 2.0
    target[:, 23:91] = 4.0
    pred[:, 30, :] = 12.0
    target[:, 30, :] = 1.0
    mask = torch.ones(1, 133, dtype=torch.bool)
    mask[:, 30] = False
    assert aligned_mpjpe(pred, target, list(range(23, 91)), anchor_index=30, mask=mask).item() == 0.0


def test_wholebody_loss_hand_weight_changes_total_loss():
    pred = torch.zeros(1, 133, 3, requires_grad=True)
    target = torch.zeros(1, 133, 3)
    target[:, 91:112] = 1.0
    mask = torch.ones(1, 133, dtype=torch.bool)
    low = WholeBodyLoss(part_weights={"left_hand": 1.0})
    high = WholeBodyLoss(part_weights={"left_hand": 10.0})
    assert high(pred, target, mask)["total"].item() > low(pred, target, mask)["total"].item()


def test_boneloss_respects_target_mask():
    pred = torch.tensor([[[3.0, 0.0, 0.0]] * 133], dtype=torch.float32)
    target = torch.zeros(1, 133, 3, dtype=torch.float32)
    mask = torch.zeros(1, 133, dtype=torch.bool)
    losses = WholeBodyLoss()(pred, target, mask)
    assert losses["bone"].item() == 0.0
    assert losses["total"].item() == 0.0


def test_metrics_and_loss_canonicalize_batched_singleton_mask():
    pred = torch.zeros(2, 133, 3)
    target = torch.zeros(2, 133, 3)
    mask = torch.ones(2, 133, 1, dtype=torch.bool)

    assert mpjpe(pred, target, mask).item() == 0.0
    assert WholeBodyLoss()(pred, target, mask)["total"].item() == 0.0


def test_face_local_loss_aligns_to_coco_body_nose_index_zero():
    pred = torch.zeros(1, 133, 3)
    target = torch.zeros(1, 133, 3)
    pred[:, 0, 0] = 2.0
    pred[:, 23:91, 0] = 5.0

    loss = WholeBodyLoss()(pred, target, torch.ones(1, 133, dtype=torch.bool))

    assert loss["face_local"].item() == 3.0
