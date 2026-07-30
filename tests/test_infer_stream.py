import numpy as np
import pytest
import torch

from mypose.engine.infer_stream import PoseStream
from mypose.models.hrgcn_lifter import HRGCNLifter


def test_pose_stream_accepts_numpy_frame_and_returns_numpy_pose():
    model = HRGCNLifter(hidden_channels=16, use_temporal=True)
    stream = PoseStream(model, device=torch.device("cpu"))
    frame = np.zeros((133, 3), dtype=np.float32)
    pred = stream.step(frame)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (133, 3)


def test_pose_stream_reset_keeps_api_usable():
    model = HRGCNLifter(hidden_channels=16, use_temporal=True)
    stream = PoseStream(model, device=torch.device("cpu"))
    stream.step(np.zeros((133, 3), dtype=np.float32))
    stream.reset()
    pred = stream.step(np.zeros((133, 3), dtype=np.float32))
    assert pred.shape == (133, 3)


def test_pose_stream_rejects_tensor_with_wrong_shape():
    model = HRGCNLifter(hidden_channels=16, use_temporal=True)
    stream = PoseStream(model, device=torch.device("cpu"))
    with pytest.raises(ValueError, match=r"frame_2d expected shape \(133, 3\)"):
        stream.step(torch.zeros((132, 3)))


def test_pose_stream_rejects_non_finite_tensor_values():
    model = HRGCNLifter(hidden_channels=16, use_temporal=True)
    stream = PoseStream(model, device=torch.device("cpu"))
    frame = torch.zeros((133, 3))
    frame[0, 0] = torch.nan
    with pytest.raises(ValueError, match="frame_2d must contain only finite values"):
        stream.step(frame)
