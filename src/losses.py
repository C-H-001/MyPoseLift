"""3D 姿态 loss。数值在归一化空间 (mm / torso_scale)。"""
import torch


def mpjpe_loss(pred, gt):
    """MPJPE: 平均每关节欧氏距离。pred/gt: (B, J, 3)"""
    return torch.mean(torch.norm(pred - gt, dim=-1))


def weighted_mpjpe_loss(pred, gt, joint_mask):
    """加权 MPJPE: 支持全局 (J,) 或 per-sample (B, J) mask。
    只统计有监督关节, 返回监督关节的平均每关节误差 (对 batch 平均)。
    """
    d = torch.norm(pred - gt, dim=-1)  # (B, J)
    m = joint_mask.float().to(d.device)
    if m.dim() == 1:
        m = m.unsqueeze(0)             # (J,) -> (1, J)
    m = m.expand_as(d)                 # 广播到 (B, J)
    denom = (m > 0).sum()
    if denom == 0:
        return torch.zeros((), device=d.device)
    return torch.sum(d * m) / denom
