from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    best_metric: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "best_metric": best_metric,
    }, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state
