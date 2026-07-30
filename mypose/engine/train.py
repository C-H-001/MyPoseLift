from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from mypose.data.h3wb import H3WBDataset
from mypose.engine.checkpoint import save_checkpoint
from mypose.engine.evaluate import evaluate
from mypose.models.hrgcn_lifter import HRGCNLifter
from mypose.models.losses import WholeBodyLoss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    requested_device = cfg["train"]["device"]
    selected_device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else "cpu" if requested_device == "auto" else requested_device
    device = torch.device(selected_device)
    train_set = H3WBDataset(Path(cfg["data"]["train_cache"]), window=int(cfg["data"]["window"]))
    train_loader = DataLoader(train_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"]["num_workers"]))
    model = HRGCNLifter(
        hidden_channels=int(cfg["model"]["hidden_channels"]),
        use_temporal=bool(cfg["model"]["use_temporal"]),
    ).to(device)
    criterion = WholeBodyLoss(part_weights=cfg["loss"]["part_weights"], local_weights=cfg["loss"]["local_weights"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    out_dir = Path(cfg["train"]["out_dir"])
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            history = batch["history_2d"].to(device=device, dtype=torch.float32)
            target = batch["target_3d"].to(device=device, dtype=torch.float32)
            mask = batch["target_mask"].to(device=device, dtype=torch.bool)
            losses = criterion(model(history), target, mask)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
        metrics = evaluate(model, train_loader, device)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch=epoch, metrics=metrics)
        print({"epoch": epoch, **metrics})


if __name__ == "__main__":
    main()
