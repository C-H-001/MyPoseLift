"""3D 姿态 loss。数值在归一化空间 (mm / torso_scale)。"""
import torch


def mpjpe_loss(pred, gt):
    """MPJPE: 平均每关节欧氏距离。pred/gt: (B, J, 3)"""
    return torch.mean(torch.norm(pred - gt, dim=-1))


def weighted_mpjpe_loss(pred, gt, joint_mask):
    """加权 MPJPE: joint_mask:(J,) float/bool -> 只统计有监督关节。
    返回: 监督关节的【每样本平均】每关节误差 (对 batch 平均)。
    分母 = 监督关节数 x batch 大小, 保证与 mpjpe_loss 数值一致。
    """
    d = torch.norm(pred - gt, dim=-1)  # (B, J)
    m = joint_mask.float().to(d.device)
    denom = torch.sum(m) * d.shape[0]
    if denom == 0:
        return torch.zeros((), device=d.device)
    return torch.sum(d * m) / denom
