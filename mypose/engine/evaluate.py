from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from mypose.data.h3wb import H3WBDataset
from mypose.data.keypoints65 import NUM_KEYPOINTS, get_part_indices
from mypose.engine import build_model_from_config
from mypose.engine.checkpoint import load_checkpoint
from mypose.utils.metrics import canonicalize_mask


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
            "MPJPE_head3",
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
        mask = canonicalize_mask(batch["target_mask"], target)
        pred = model(history)
        accumulate(
            "MPJPE_whole",
            _metric_sum_count(pred, target, mask, list(range(NUM_KEYPOINTS))),
        )
        accumulate("MPJPE_body", _metric_sum_count(pred, target, mask, get_part_indices("body")))
        accumulate("MPJPE_feet", _metric_sum_count(pred, target, mask, get_part_indices("foot")))
        accumulate(
            "MPJPE_head3",
            _metric_sum_count(pred, target, mask, get_part_indices("head3")),
        )
        accumulate(
            "MPJPE_left_hand",
            _metric_sum_count(pred, target, mask, get_part_indices("left_hand")),
        )
        accumulate(
            "MPJPE_right_hand",
            _metric_sum_count(pred, target, mask, get_part_indices("right_hand")),
        )
        left = _metric_sum_count(
            pred, target, mask, get_part_indices("left_hand"), anchor_index=23
        )
        right = _metric_sum_count(
            pred, target, mask, get_part_indices("right_hand"), anchor_index=44
        )
        accumulate(
            "MPJPE_hands_wrist_aligned",
            (left[0] + right[0], left[1] + right[1]),
        )
    return {
        name: distance_sum / valid_count if valid_count else 0.0
        for name, (distance_sum, valid_count) in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained pose lifter checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        help="override data.val_cache for an explicit smoke-test cache",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    requested_device = cfg["train"]["device"]
    selected_device = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested_device == "auto"
        else requested_device
    )
    device = torch.device(selected_device)
    cache_file = args.cache or Path(cfg["data"]["val_cache"])
    dataset = H3WBDataset(cache_file, window=int(cfg["data"]["window"]))
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
    )
    model = build_model_from_config(cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    print(evaluate(model, dataloader, device))


if __name__ == "__main__":
    main()
