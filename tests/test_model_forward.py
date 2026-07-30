import torch

from mypose.models.hrgcn_lifter import HRGCNLifter


def test_framewise_lifter_outputs_133_3():
    model = HRGCNLifter(use_temporal=False, hidden_channels=32)
    x = torch.randn(2, 1, 133, 3)
    y = model(x)
    assert y.shape == (2, 133, 3)


def test_temporal_lifter_outputs_133_3_on_cpu():
    model = HRGCNLifter(use_temporal=True, hidden_channels=32)
    x = torch.randn(2, 5, 133, 3)
    y = model(x)
    assert y.shape == (2, 133, 3)


def test_streaming_lifter_step_outputs_current_pose():
    model = HRGCNLifter(use_temporal=True, hidden_channels=32)
    model.reset_stream()
    y = model.step(torch.randn(1, 133, 3))
    assert y.shape == (1, 133, 3)
