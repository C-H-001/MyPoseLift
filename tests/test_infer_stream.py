import numpy as np
import pytest
import torch

from mypose.engine.infer_stream import PoseStream
from mypose.models.causal_tcn_lifter import CausalTCNLifter


def test_pose_stream_accepts_numpy_frame_and_returns_numpy_pose():
    model = CausalTCNLifter(hidden_channels=16, num_blocks=1)
    stream = PoseStream(model, device=torch.device("cpu"))
    frame = np.zeros((65, 3), dtype=np.float32)
    pred = stream.step(frame)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (65, 3)


def test_pose_stream_reset_keeps_api_usable():
    model = CausalTCNLifter(hidden_channels=16, num_blocks=1)
    stream = PoseStream(model, device=torch.device("cpu"))
    stream.step(np.zeros((65, 3), dtype=np.float32))
    stream.reset()
    pred = stream.step(np.zeros((65, 3), dtype=np.float32))
    assert pred.shape == (65, 3)


def test_pose_stream_rejects_tensor_with_wrong_shape():
    model = CausalTCNLifter(hidden_channels=16, num_blocks=1)
    stream = PoseStream(model, device=torch.device("cpu"))
    with pytest.raises(ValueError, match=r"frame_2d expected shape \(65, 3\)"):
        stream.step(torch.zeros((64, 3)))


def test_pose_stream_rejects_non_finite_tensor_values():
    model = CausalTCNLifter(hidden_channels=16, num_blocks=1)
    stream = PoseStream(model, device=torch.device("cpu"))
    frame = torch.zeros((65, 3))
    frame[0, 0] = torch.nan
    with pytest.raises(ValueError, match="frame_2d must contain only finite values"):
        stream.step(frame)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((65, 2), dtype=np.float32),
        torch.zeros((65, 2), dtype=torch.float32),
    ],
)
def test_pose_stream_rejects_frames_without_confidence_channel(frame):
    model = CausalTCNLifter(hidden_channels=16, num_blocks=1)
    stream = PoseStream(model, device=torch.device("cpu"))

    with pytest.raises(ValueError, match=r"frame_2d expected shape \(65, 3\)"):
        stream.step(frame)
