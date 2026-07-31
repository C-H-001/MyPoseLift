from __future__ import annotations

import torch

from mypose.models.causal_tcn_lifter import CausalTCNLifter


def build_model_from_config(cfg: dict) -> torch.nn.Module:
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    if model_type == "causal_tcn":
        return CausalTCNLifter(
            num_keypoints=int(model_cfg.get("num_keypoints", 65)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            hidden_channels=int(model_cfg["hidden_channels"]),
            num_blocks=int(model_cfg.get("num_blocks", 4)),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_history=int(cfg["data"]["window"]),
        )
    if model_type == "hrgcn":
        raise ValueError(
            "Legacy HRGCN only supports 133 keypoints and is unavailable "
            "for the 65-point migration"
        )
    raise ValueError(f"unknown model type: {model_type!r}")
