from __future__ import annotations

import numpy as np
import torch

from mypose.data.keypoints133 import validate_keypoints_shape
from mypose.models.hrgcn_lifter import HRGCNLifter


class PoseStream:
    def __init__(self, model: HRGCNLifter, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.model.reset_stream()

    @torch.no_grad()
    def step(self, frame_2d: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(frame_2d, np.ndarray):
            validate_keypoints_shape(frame_2d, dims=2, name="frame_2d")
            frame = torch.from_numpy(frame_2d).to(device=self.device, dtype=torch.float32)
        else:
            frame = frame_2d.to(device=self.device, dtype=torch.float32)
        pred = self.model.step(frame.unsqueeze(0))
        return pred.squeeze(0).cpu().numpy()
