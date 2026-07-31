import pytest
import torch

from mypose.models.causal_tcn_lifter import CausalTCNLifter


def test_causal_tcn_outputs_current_65_point_pose():
    model = CausalTCNLifter(hidden_channels=64, num_blocks=2)
    y = model(torch.zeros(2, 81, 65, 3))
    assert y.shape == (2, 65, 3)


def test_causal_tcn_rejects_non_65_points():
    model = CausalTCNLifter(hidden_channels=32, num_blocks=1)
    with pytest.raises(ValueError, match="65"):
        model(torch.zeros(1, 27, 133, 3))


def test_causal_tcn_current_output_does_not_depend_on_future_frames():
    model = CausalTCNLifter(hidden_channels=32, num_blocks=2, kernel_size=3)
    model.eval()
    history = torch.randn(1, 81, 65, 3)
    changed = history.clone()
    changed[:, 80] = changed[:, 80] + 1000.0
    with torch.no_grad():
        before_future = model.forward_sequence(history)[:, :80]
        after_future = model.forward_sequence(changed)[:, :80]
    torch.testing.assert_close(before_future, after_future)


def test_causal_tcn_stream_matches_batch_last_frame():
    torch.manual_seed(7)
    model = CausalTCNLifter(hidden_channels=32, num_blocks=2)
    model.eval()
    history = torch.randn(1, 27, 65, 3)
    with torch.no_grad():
        batch = model(history)
        model.reset_stream()
        stepped = None
        for t in range(history.shape[1]):
            stepped = model.step(history[:, t])
    torch.testing.assert_close(batch, stepped)
