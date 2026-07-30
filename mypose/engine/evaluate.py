from __future__ import annotations

import torch

from mypose.data.keypoints133 import get_part_indices


def _distance_sum_count(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.linalg.norm(pred - target, dim=-1)
    valid = mask.to(device=pred.device, dtype=torch.bool)
    return distances[valid].sum(), valid.sum()


def _metric_sum_count(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    indices: list[int],
    anchor_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    metric_pred = pred[:, indices]
    metric_target = target[:, indices]
    metric_mask = mask[:, indices]
    if anchor_index is not None:
        metric_pred = metric_pred - pred[:, anchor_index:anchor_index + 1]
        metric_target = metric_target - target[:, anchor_index:anchor_index + 1]
        metric_mask = metric_mask & mask[:, anchor_index:anchor_index + 1]
    return _distance_sum_count(metric_pred, metric_target, metric_mask)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {
        name: [0.0, 0]
        for name in (
            "MPJPE_whole",
            "MPJPE_body",
            "MPJPE_feet",
            "MPJPE_face",
            "MPJPE_face_nose_aligned",
            "MPJPE_left_hand",
            "MPJPE_right_hand",
            "MPJPE_hands_wrist_aligned",
        )
    }

    def accumulate(name: str, values: tuple[torch.Tensor, torch.Tensor]) -> None:
        distance_sum, valid_count = values
        totals[name][0] += distance_sum.item()
        totals[name][1] += int(valid_count.item())

    for batch in dataloader:
        history = batch["history_2d"].to(device=device, dtype=torch.float32)
        target = batch["target_3d"].to(device=device, dtype=torch.float32)
        mask = batch["target_mask"].to(device=device, dtype=torch.bool)
        pred = model(history)
        accumulate("MPJPE_whole", _metric_sum_count(pred, target, mask, list(range(133))))
        accumulate("MPJPE_body", _metric_sum_count(pred, target, mask, get_part_indices("body")))
        accumulate("MPJPE_feet", _metric_sum_count(pred, target, mask, get_part_indices("foot")))
        accumulate("MPJPE_face", _metric_sum_count(pred, target, mask, get_part_indices("face")))
        accumulate(
            "MPJPE_face_nose_aligned",
            _metric_sum_count(pred, target, mask, list(range(23, 91)), anchor_index=30),
        )
        accumulate(
            "MPJPE_left_hand",
            _metric_sum_count(pred, target, mask, get_part_indices("left_hand")),
        )
        accumulate(
            "MPJPE_right_hand",
            _metric_sum_count(pred, target, mask, get_part_indices("right_hand")),
        )
        left = _metric_sum_count(pred, target, mask, list(range(91, 112)), anchor_index=91)
        right = _metric_sum_count(pred, target, mask, list(range(112, 133)), anchor_index=112)
        accumulate(
            "MPJPE_hands_wrist_aligned",
            (left[0] + right[0], left[1] + right[1]),
        )
    return {
        name: distance_sum / valid_count if valid_count else 0.0
        for name, (distance_sum, valid_count) in totals.items()
    }
