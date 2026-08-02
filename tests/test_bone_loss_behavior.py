"""Document the upstream BoneLoss batch-mean failure mode."""

import sys
from pathlib import Path

import torch

# Use the same checked-in MMPose and compatibility modules as the training
# entrypoint, rather than whichever partial mmpose wheel happens to be active.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'compat'))
sys.path.insert(0, str(ROOT / 'external' / 'mmpose'))

from mmpose.models.losses import BoneLoss


def test_bone_loss_has_autograd_gradient_for_a_non_cancelled_batch():
    output = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]],
        requires_grad=True,
    )
    target = torch.zeros_like(output)
    target[:, 1, 0] = 1.0
    weights = torch.ones((2, 2))

    loss = BoneLoss([0, 0], use_target_weight=True)(output, target, weights)
    loss.backward()

    assert torch.isfinite(loss)
    assert output.grad is not None
    assert torch.isfinite(output.grad).all()
    assert output.grad.abs().sum() > 0


def test_bone_loss_can_cancel_wrong_per_sample_bones():
    # One sample is twice as long and the other has zero length. Their mean is
    # the target length, so upstream BoneLoss returns zero despite both samples
    # being wrong. Gradient accumulation cannot repair this batch-mean objective.
    output = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    target = torch.zeros_like(output)
    target[:, 1, 0] = 1.0
    weights = torch.ones((2, 2))

    loss = BoneLoss([0, 0], use_target_weight=True)(output, target, weights)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.0))
    assert output.grad is not None
    assert torch.isclose(output.grad.abs().sum(), torch.tensor(0.0))
