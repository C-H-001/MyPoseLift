from __future__ import annotations

from collections.abc import Sequence

import torch

from mypose.data.keypoints65 import get_part_indices


def canonicalize_mask(mask: torch.Tensor | None, pred: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(pred.shape[:-1], dtype=torch.bool, device=pred.device)
    valid = mask.to(device=pred.device, dtype=torch.bool)
    if valid.shape == pred.shape[:-1] + (1,):
        valid = valid.squeeze(-1)
    if valid.shape != pred.shape[:-1]:
        raise ValueError(
            f"mask expected shape {tuple(pred.shape[:-1])}, got {tuple(mask.shape)}"
        )
    return valid


def mpjpe(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    distances = torch.linalg.norm(pred - target, dim=-1)
    valid = canonicalize_mask(mask, pred)
    if valid.sum() == 0:
        return distances.sum() * 0.0
    return distances[valid].mean()


def part_mpjpe(
    pred: torch.Tensor, target: torch.Tensor, part: str, mask: torch.Tensor | None = None
) -> torch.Tensor:
    indices = get_part_indices(part)
    part_mask = None if mask is None else mask[:, indices]
    return mpjpe(pred[:, indices], target[:, indices], part_mask)


def aligned_mpjpe(
    pred: torch.Tensor,
    target: torch.Tensor,
    indices: Sequence[int],
    anchor_index: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    anchor_pred = pred[:, anchor_index:anchor_index + 1]
    anchor_target = target[:, anchor_index:anchor_index + 1]
    local_pred = pred[:, indices] - anchor_pred
    local_target = target[:, indices] - anchor_target
    local_mask = None if mask is None else canonicalize_mask(mask, pred)[:, indices]
    if mask is not None:
        anchor_valid = canonicalize_mask(mask, pred)[:, anchor_index]
        local_mask = local_mask & anchor_valid[:, None]
    return mpjpe(local_pred, local_target, local_mask)
