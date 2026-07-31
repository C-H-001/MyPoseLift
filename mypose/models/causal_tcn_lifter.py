from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mypose.data.keypoints65 import NUM_KEYPOINTS


class _CausalResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self.left_padding, 0))
        x = self.temporal(x)
        x = self.pointwise(self.dropout(self.activation(x)))
        return self.norm((x + residual).transpose(1, 2)).transpose(1, 2)


class CausalTCNLifter(nn.Module):
    def __init__(
        self,
        num_keypoints: int = NUM_KEYPOINTS,
        in_channels: int = 3,
        hidden_channels: int = 512,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        if num_keypoints != NUM_KEYPOINTS:
            raise ValueError(
                f"num_keypoints must be {NUM_KEYPOINTS}, got {num_keypoints}"
            )
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_channels <= 0:
            raise ValueError(
                f"hidden_channels must be positive, got {hidden_channels}"
            )
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.num_keypoints = num_keypoints
        self.in_channels = in_channels
        self.flatten = nn.Linear(num_keypoints * in_channels, hidden_channels)
        dilations = [kernel_size**index for index in range(num_blocks)]
        self.blocks = nn.ModuleList(
            [
                _CausalResidualBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, num_keypoints * 3),
        )
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        self._stream: list[torch.Tensor] = []

    def _validate_history(self, history_2d: torch.Tensor) -> None:
        expected = (self.num_keypoints, self.in_channels)
        if history_2d.ndim != 4 or tuple(history_2d.shape[-2:]) != expected:
            raise ValueError(
                "history_2d expected shape "
                f"(B, T, {self.num_keypoints}, {self.in_channels}), "
                f"got {tuple(history_2d.shape)}"
            )
        if history_2d.shape[1] == 0:
            raise ValueError("history_2d must contain at least one frame")

    def forward_sequence(self, history_2d: torch.Tensor) -> torch.Tensor:
        self._validate_history(history_2d)
        batch_size, steps = history_2d.shape[:2]
        x = history_2d.reshape(batch_size, steps, -1)
        x = self.flatten(x).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.head(x.transpose(1, 2))
        return x.reshape(batch_size, steps, self.num_keypoints, 3)

    def forward(self, history_2d: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(history_2d)[:, -1]

    def reset_stream(self) -> None:
        self._stream = []

    def step(self, frame_2d: torch.Tensor) -> torch.Tensor:
        expected = (self.num_keypoints, self.in_channels)
        if frame_2d.ndim != 3 or tuple(frame_2d.shape[-2:]) != expected:
            raise ValueError(
                "frame_2d expected shape "
                f"(B, {self.num_keypoints}, {self.in_channels}), "
                f"got {tuple(frame_2d.shape)}"
            )
        self._stream.append(frame_2d.detach().clone())
        self._stream = self._stream[-self.receptive_field :]
        return self.forward(torch.stack(self._stream, dim=1))
