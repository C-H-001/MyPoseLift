from __future__ import annotations

import torch

from mypose.utils.metrics import aligned_mpjpe, mpjpe, part_mpjpe


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {
        "MPJPE_whole": 0.0,
        "MPJPE_body": 0.0,
        "MPJPE_feet": 0.0,
        "MPJPE_face": 0.0,
        "MPJPE_face_nose_aligned": 0.0,
        "MPJPE_left_hand": 0.0,
        "MPJPE_right_hand": 0.0,
        "MPJPE_hands_wrist_aligned": 0.0,
    }
    count = 0
    for batch in dataloader:
        history = batch["history_2d"].to(device=device, dtype=torch.float32)
        target = batch["target_3d"].to(device=device, dtype=torch.float32)
        mask = batch["target_mask"].to(device=device, dtype=torch.bool)
        pred = model(history)
        totals["MPJPE_whole"] += mpjpe(pred, target, mask).item()
        totals["MPJPE_body"] += part_mpjpe(pred, target, "body", mask).item()
        totals["MPJPE_feet"] += part_mpjpe(pred, target, "foot", mask).item()
        totals["MPJPE_face"] += part_mpjpe(pred, target, "face", mask).item()
        totals["MPJPE_face_nose_aligned"] += aligned_mpjpe(pred, target, list(range(23, 91)), 30, mask).item()
        totals["MPJPE_left_hand"] += part_mpjpe(pred, target, "left_hand", mask).item()
        totals["MPJPE_right_hand"] += part_mpjpe(pred, target, "right_hand", mask).item()
        left = aligned_mpjpe(pred, target, list(range(91, 112)), 91, mask)
        right = aligned_mpjpe(pred, target, list(range(112, 133)), 112, mask)
        totals["MPJPE_hands_wrist_aligned"] += ((left + right) * 0.5).item()
        count += 1
    return {name: value / max(1, count) for name, value in totals.items()}
