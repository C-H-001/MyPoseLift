from __future__ import annotations

import numpy as np
import torch

from mypose.data.keypoints65 import NUM_KEYPOINTS


class PoseStream:
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.model.reset_stream()

    @torch.no_grad()
    def step(self, frame_2d: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(frame_2d, np.ndarray):
            if frame_2d.shape != (NUM_KEYPOINTS, 3):
                raise ValueError(
                    f"frame_2d expected shape ({NUM_KEYPOINTS}, 3), "
                    f"got {tuple(frame_2d.shape)}"
                )
            if not np.isfinite(frame_2d).all():
                raise ValueError("frame_2d must contain only finite values")
            frame = torch.from_numpy(frame_2d).to(device=self.device, dtype=torch.float32)
        else:
            if frame_2d.ndim != 2 or tuple(frame_2d.shape[-2:]) != (
                NUM_KEYPOINTS,
                3,
            ):
                raise ValueError(
                    f"frame_2d expected shape ({NUM_KEYPOINTS}, 3), "
                    f"got {tuple(frame_2d.shape)}"
                )
            if not torch.isfinite(frame_2d).all():
                raise ValueError("frame_2d must contain only finite values")
            frame = frame_2d.to(device=self.device, dtype=torch.float32)
        pred = self.model.step(frame.unsqueeze(0))
        return pred.squeeze(0).cpu().numpy()
