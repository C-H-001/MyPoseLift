from pathlib import Path

import pytest
import torch
import yaml

from mypose.engine import build_model_from_config
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


def test_causal_tcn_first_stream_step_matches_repeated_first_frame_window():
    torch.manual_seed(17)
    model = CausalTCNLifter(hidden_channels=16, num_blocks=2, max_history=5).eval()
    first_frame = torch.randn(1, 65, 3)

    with torch.no_grad():
        expected = model(first_frame[:, None].repeat(1, 5, 1, 1))
        model.reset_stream()
        actual = model.step(first_frame)

    torch.testing.assert_close(actual, expected)


def test_causal_tcn_partial_stream_matches_repeated_first_frame_history():
    torch.manual_seed(19)
    model = CausalTCNLifter(hidden_channels=16, num_blocks=2, max_history=5).eval()
    observed = torch.randn(1, 3, 65, 3)
    expected_history = torch.cat(
        [observed[:, :1].repeat(1, 2, 1, 1), observed], dim=1
    )

    with torch.no_grad():
        expected = model(expected_history)
        model.reset_stream()
        for index in range(observed.shape[1]):
            actual = model.step(observed[:, index])

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/h3wb_tcn_t27.yaml"),
        Path("configs/h3wb_tcn_t81.yaml"),
    ],
)
def test_configured_stream_matches_batch_after_exceeding_window(config_path):
    torch.manual_seed(11)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["model"]["hidden_channels"] = 8
    model = build_model_from_config(cfg).eval()
    window = cfg["data"]["window"]
    history = torch.randn(1, window + 2, 65, 3)

    with torch.no_grad():
        batch = model(history[:, -window:])
        model.reset_stream()
        for index in range(history.shape[1]):
            stepped = model.step(history[:, index])

    torch.testing.assert_close(batch, stepped)
