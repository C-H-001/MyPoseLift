"""Loss 模块测试"""
import torch
from src.losses import mpjpe_loss, weighted_mpjpe_loss


def test_mpjpe_zero():
    pred = torch.randn(4, 17, 3)
    assert mpjpe_loss(pred, pred.clone()).item() < 1e-5


def test_mpjpe_magnitude():
    # 仅 x 轴偏差 10mm -> 每关节欧氏距离 = 10
    pred = torch.randn(2, 17, 3)
    gt = pred.clone()
    gt[:, :, 0] += 10.0
    loss = mpjpe_loss(pred, gt)
    assert abs(loss.item() - 10.0) < 1e-3


def test_weighted_mask():
    # 缺失关节权重 0
    pred = torch.randn(2, 17, 3)
    gt = pred.clone()
    gt[:, 1:5] = 999  # 未监督关节任意偏差
    mask = torch.ones(17)
    mask[1:5] = 0
    loss = weighted_mpjpe_loss(pred, gt, mask)
    assert loss.item() < 1e-4


def test_weighted_mask_12_joints():
    # 12 关节监督场景: 加权结果与子集 mpjpe 一致
    pred = torch.randn(3, 17, 3)
    gt = pred.clone()
    gt[:, :, 0] += 5.0
    mask = torch.zeros(17)
    mask[5:] = 1
    lw = weighted_mpjpe_loss(pred, gt, mask)
    lm = mpjpe_loss(pred[:, 5:], gt[:, 5:])
    assert abs(lw.item() - lm.item()) < 1e-4


def test_grad_flows():
    pred = torch.randn(2, 17, 3, requires_grad=True)
    gt = torch.randn(2, 17, 3)
    mask = torch.ones(17)
    loss = weighted_mpjpe_loss(pred, gt, mask)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
