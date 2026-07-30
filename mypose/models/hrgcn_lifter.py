from __future__ import annotations

import torch
from torch import nn

from mypose.data.keypoints133 import COCO_WHOLEBODY_EDGES, NUM_KEYPOINTS
from mypose.models.temporal_adapter import CausalTemporalAdapter


class GraphBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(channels, channels)
        self.neighbor_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        edges = torch.tensor(COCO_WHOLEBODY_EDGES, dtype=torch.long)
        reverse = edges[:, [1, 0]]
        self.register_buffer("edges", torch.cat([edges, reverse], dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = self.edges[:, 0]
        dst = self.edges[:, 1]
        messages = torch.zeros_like(x)
        messages.index_add_(1, dst, x[:, src])
        degree = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        messages = messages / degree.clamp_min(1.0)[None, :, None]
        return self.norm(x + self.act(self.self_proj(x) + self.neighbor_proj(messages)))


class HRGCNLifter(nn.Module):
    def __init__(
        self,
        num_keypoints: int = NUM_KEYPOINTS,
        in_channels: int = 3,
        hidden_channels: int = 128,
        use_temporal: bool = False,
        temporal_kernel_size: int = 3,
        temporal_dilation: int = 1,
    ) -> None:
        if num_keypoints != NUM_KEYPOINTS:
            raise ValueError(f"num_keypoints must be {NUM_KEYPOINTS}, got {num_keypoints}")
        super().__init__()
        self.num_keypoints = num_keypoints
        self.use_temporal = use_temporal
        self.temporal = (
            CausalTemporalAdapter(
                in_channels,
                hidden_channels,
                kernel_size=temporal_kernel_size,
                dilation=temporal_dilation,
            )
            if use_temporal
            else None
        )
        self.input = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([GraphBlock(hidden_channels), GraphBlock(hidden_channels), GraphBlock(hidden_channels)])
        self.body_head = nn.Linear(hidden_channels, 3)
        self.fine_head = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.GELU(), nn.Linear(hidden_channels, 3))

    def forward(self, history_2d: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = history_2d
        if self.temporal is not None:
            x = self.temporal(x)
        frame = x[:, -1]
        return self.forward_frame(frame)

    def forward_frame(self, frame_2d: torch.Tensor) -> torch.Tensor:
        h = self.input(frame_2d)
        for block in self.blocks:
            h = block(h)
        out = self.body_head(h)
        fine_indices = list(range(23, 133))
        out[:, fine_indices] = self.fine_head(h[:, fine_indices])
        return out

    def reset_stream(self) -> None:
        if self.temporal is not None:
            self.temporal.reset_stream()

    def step(self, frame_2d: torch.Tensor) -> torch.Tensor:
        if self.temporal is None:
            return self.forward_frame(frame_2d)
        adapted = self.temporal.step(frame_2d)
        return self.forward_frame(adapted)
