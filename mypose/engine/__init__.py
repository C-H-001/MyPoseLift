from __future__ import annotations

import torch

from mypose.models.causal_tcn_lifter import CausalTCNLifter
from mypose.models.hrgcn_lifter import HRGCNLifter


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
        )
    if model_type == "hrgcn":
        return HRGCNLifter(
            hidden_channels=int(model_cfg["hidden_channels"]),
            use_temporal=bool(model_cfg["use_temporal"]),
            temporal_kernel_size=int(model_cfg.get("temporal_kernel_size", 3)),
            temporal_dilation=int(model_cfg.get("temporal_dilation", 1)),
        )
    raise ValueError(f"unknown model type: {model_type!r}")
