import torch
import pytest

from mypose.models.hrgcn_lifter import HRGCNLifter


def test_lifter_rejects_non_133_keypoint_count():
    with pytest.raises(ValueError, match="num_keypoints must be 133"):
        HRGCNLifter(num_keypoints=17)


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


def test_streaming_lifter_matches_batch_current_pose():
    torch.manual_seed(2)
    model = HRGCNLifter(use_temporal=True, hidden_channels=32)
    model.eval()
    history = torch.randn(1, 5, 133, 3)
    with torch.no_grad():
        batch = model(history)
        model.reset_stream()
        stepped = None
        for idx in range(history.shape[1]):
            stepped = model.step(history[:, idx])
    torch.testing.assert_close(batch, stepped)


def test_lifter_output_at_frame_does_not_depend_on_future_frames():
    torch.manual_seed(3)
    model = HRGCNLifter(use_temporal=True, hidden_channels=32)
    model.eval()
    history = torch.randn(1, 5, 133, 3)
    changed = history.clone()
    changed[:, 4] = changed[:, 4] + 1000.0
    with torch.no_grad():
        before_future = model.temporal(history)[:, :4]
        after_future = model.temporal(changed)[:, :4]
    torch.testing.assert_close(before_future, after_future)
