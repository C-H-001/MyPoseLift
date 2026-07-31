from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalTemporalAdapter(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = 0
        self.input = nn.Linear(in_channels, hidden_channels)
        self.depthwise = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,
            padding=padding,
        )
        self.pointwise = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.output = nn.Linear(hidden_channels, in_channels)
        self.activation = nn.GELU()
        self._stream: list[torch.Tensor] = []

    @property
    def receptive_field(self) -> int:
        return (self.kernel_size - 1) * self.dilation + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, joints, channels = x.shape
        h = self.input(x).permute(0, 2, 3, 1).reshape(bsz * joints, -1, steps)
        left_pad = self.receptive_field - 1
        h = F.pad(h, (left_pad, 0))
        h = self.depthwise(h)
        h = self.activation(self.pointwise(h))
        h = h.reshape(bsz, joints, -1, steps).permute(0, 3, 1, 2)
        return x + self.output(h)

    def reset_stream(self) -> None:
        self._stream = []

    def step(self, frame: torch.Tensor) -> torch.Tensor:
        self._stream.append(frame.detach().clone())
        if len(self._stream) > self.receptive_field:
            self._stream = self._stream[-self.receptive_field:]
        seq = torch.stack(self._stream, dim=1)
        return self.forward(seq)[:, -1]
