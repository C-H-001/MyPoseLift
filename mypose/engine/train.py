from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from mypose.data.h3wb import H3WBDataset
from mypose.engine.checkpoint import load_checkpoint, save_checkpoint
from mypose.engine.evaluate import evaluate
from mypose.models.hrgcn_lifter import HRGCNLifter
from mypose.models.losses import WholeBodyLoss


def _device_from_config(cfg: dict) -> torch.device:
    requested_device = cfg["train"]["device"]
    selected_device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else "cpu" if requested_device == "auto" else requested_device
    return torch.device(selected_device)


def _model_from_config(cfg: dict) -> HRGCNLifter:
    model_cfg = cfg["model"]
    return HRGCNLifter(
        hidden_channels=int(model_cfg["hidden_channels"]),
        use_temporal=bool(model_cfg["use_temporal"]),
        temporal_kernel_size=int(model_cfg.get("temporal_kernel_size", 3)),
        temporal_dilation=int(model_cfg.get("temporal_dilation", 1)),
    )


def train_from_config(cfg: dict, resume: Path | None = None) -> dict[str, float]:
    seed = int(cfg["train"].get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device_from_config(cfg)
    train_set = H3WBDataset(Path(cfg["data"]["train_cache"]), window=int(cfg["data"]["window"]))
    val_set = H3WBDataset(Path(cfg["data"]["val_cache"]), window=int(cfg["data"]["window"]))
    train_loader = DataLoader(train_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"]["num_workers"]))
    val_loader = DataLoader(val_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"]["num_workers"]))
    model = _model_from_config(cfg).to(device)
    criterion = WholeBodyLoss(
        part_weights=cfg["loss"]["part_weights"],
        local_weights=cfg["loss"]["local_weights"],
        bone_weight=float(cfg["loss"].get("bone_weight", 0.01)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    out_dir = Path(cfg["train"]["out_dir"])
    resume_path = Path(resume) if resume is not None else None
    start_epoch = 0
    best_metric = float("inf")
    metrics: dict[str, float] = {}
    if resume_path is not None:
        state = load_checkpoint(resume_path, model, optimizer)
        start_epoch = int(state["epoch"]) + 1
        metrics = dict(state.get("metrics", {}))
        saved_best = state.get("best_metric")
        if saved_best is not None:
            best_metric = float(saved_best)
        elif "MPJPE_whole" in metrics:
            best_metric = float(metrics["MPJPE_whole"])
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch in train_loader:
            history = batch["history_2d"].to(device=device, dtype=torch.float32)
            target = batch["target_3d"].to(device=device, dtype=torch.float32)
            mask = batch["target_mask"].to(device=device, dtype=torch.bool)
            losses = criterion(model(history), target, mask)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        is_best = metrics["MPJPE_whole"] < best_metric
        if is_best:
            best_metric = metrics["MPJPE_whole"]
        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            metrics=metrics,
            best_metric=best_metric,
        )
        if is_best:
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                epoch=epoch,
                metrics=metrics,
                best_metric=best_metric,
            )
        print({"epoch": epoch, **metrics})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_from_config(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
