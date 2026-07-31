import copy

import torch
import pytest

from mypose.models.hrgcn_lifter import HRGCNLifter
from mypose.models.temporal_adapter import CausalTemporalAdapter


def test_lifter_rejects_non_133_keypoint_count():
    with pytest.raises(ValueError, match="num_keypoints must be 133"):
        HRGCNLifter(num_keypoints=17)


def test_framewise_lifter_outputs_133_3():
    model = HRGCNLifter(use_temporal=False, hidden_channels=32)
    x = torch.randn(2, 1, 133, 3)
    y = model(x)
    assert y.shape == (2, 133, 3)


def test_lifter_has_learned_joint_identity():
    model = HRGCNLifter(use_temporal=False, hidden_channels=8)

    assert tuple(model.joint_embedding.shape) == (133, 8)
    assert model.joint_embedding.requires_grad


def test_lifter_uses_nonlinear_output_head_for_all_joints():
    model = HRGCNLifter(use_temporal=False, hidden_channels=8)

    assert isinstance(model.output_head, torch.nn.Sequential)
    assert any(isinstance(layer, torch.nn.GELU) for layer in model.output_head)


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


def test_t27_lifter_has_27_frame_receptive_field():
    model = HRGCNLifter(
        use_temporal=True,
        hidden_channels=16,
        temporal_kernel_size=27,
    )

    assert model.temporal.receptive_field == 27


def test_causal_adapter_current_output_depends_on_oldest_frame_in_receptive_field():
    adapter = CausalTemporalAdapter(
        in_channels=1,
        hidden_channels=1,
        kernel_size=27,
    )
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.fill_(1.0)
        baseline = torch.zeros(1, 27, 1, 1)
        changed = baseline.clone()
        changed[:, 0] = 1.0

        baseline_current = adapter(baseline)[:, -1]
        changed_current = adapter(changed)[:, -1]

    assert not torch.allclose(baseline_current, changed_current)


def test_causal_adapter_stream_history_is_not_aliased_to_input_buffer():
    torch.manual_seed(5)
    adapter = CausalTemporalAdapter(3, 4, kernel_size=3)
    reference = copy.deepcopy(adapter)
    first = torch.randn(1, 133, 3)
    original_first = first.clone()
    second = torch.randn(1, 133, 3)

    adapter.step(first)
    first.fill_(1000.0)
    actual = adapter.step(second)
    reference.step(original_first)
    expected = reference.step(second)

    torch.testing.assert_close(actual, expected)
