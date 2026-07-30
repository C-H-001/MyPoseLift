import torch

from mypose.models.temporal_adapter import CausalTemporalAdapter


def test_temporal_adapter_output_does_not_depend_on_future_frames():
    torch.manual_seed(0)
    model = CausalTemporalAdapter(in_channels=3, hidden_channels=8, kernel_size=3)
    model.eval()
    x1 = torch.randn(1, 5, 133, 3)
    x2 = x1.clone()
    x2[:, 4] = x2[:, 4] + 1000.0
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    torch.testing.assert_close(y1[:, :4], y2[:, :4])


def test_temporal_adapter_stream_step_matches_batch_last_frame():
    torch.manual_seed(1)
    model = CausalTemporalAdapter(in_channels=3, hidden_channels=8, kernel_size=3)
    model.eval()
    seq = torch.randn(1, 4, 133, 3)
    with torch.no_grad():
        batch = model(seq)[:, -1]
        model.reset_stream()
        stepped = None
        for idx in range(seq.shape[1]):
            stepped = model.step(seq[:, idx])
    torch.testing.assert_close(batch, stepped)
