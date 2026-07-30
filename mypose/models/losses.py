from __future__ import annotations

import torch
from torch import nn

from mypose.data.keypoints133 import COCO_WHOLEBODY_EDGES
from mypose.utils.metrics import aligned_mpjpe, mpjpe, part_mpjpe


DEFAULT_PART_WEIGHTS = {
    "body": 1.0,
    "foot": 1.5,
    "face": 2.0,
    "left_hand": 2.5,
    "right_hand": 2.5,
}


class WholeBodyLoss(nn.Module):
    def __init__(
        self,
        part_weights: dict[str, float] | None = None,
        local_weights: dict[str, float] | None = None,
        bone_weight: float = 0.01,
    ) -> None:
        super().__init__()
        weights = DEFAULT_PART_WEIGHTS.copy()
        if part_weights:
            weights.update(part_weights)
        self.part_weights = weights
        self.local_weights = {"face": 1.0, "left_hand": 1.0, "right_hand": 1.0}
        if local_weights:
            self.local_weights.update(local_weights)
        self.bone_weight = float(bone_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        losses["whole"] = mpjpe(pred, target, target_mask)
        total = losses["whole"]
        for part, weight in self.part_weights.items():
            value = part_mpjpe(pred, target, part, target_mask)
            losses[f"{part}_mpjpe"] = value
            total = total + float(weight) * value
        losses["face_local"] = aligned_mpjpe(
            pred, target, list(range(23, 91)), anchor_index=30, mask=target_mask
        )
        losses["left_hand_local"] = aligned_mpjpe(
            pred, target, list(range(91, 112)), anchor_index=91, mask=target_mask
        )
        losses["right_hand_local"] = aligned_mpjpe(
            pred, target, list(range(112, 133)), anchor_index=112, mask=target_mask
        )
        for name, weight in self.local_weights.items():
            total = total + float(weight) * losses[f"{name}_local"]
        losses["bone"] = self._bone_loss(pred, target, target_mask)
        losses["total"] = total + self.bone_weight * losses["bone"]
        return losses

    def _bone_loss(
        self, pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        edge_index = torch.tensor(COCO_WHOLEBODY_EDGES, dtype=torch.long, device=pred.device)
        pred_len = torch.linalg.norm(pred[:, edge_index[:, 0]] - pred[:, edge_index[:, 1]], dim=-1)
        target_len = torch.linalg.norm(target[:, edge_index[:, 0]] - target[:, edge_index[:, 1]], dim=-1)
        if target_mask is None:
            return torch.mean(torch.abs(pred_len - target_len))
        target_mask = target_mask.to(device=pred.device, dtype=torch.bool)
        valid_mask = target_mask[:, edge_index[:, 0]] & target_mask[:, edge_index[:, 1]]
        if valid_mask.sum() == 0:
            return pred_len.sum() * 0.0
        return torch.mean(torch.abs(pred_len[valid_mask] - target_len[valid_mask]))
