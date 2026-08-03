"""因果时序数据集: 输入 81 帧 2D 序列 -> 预测最后一帧 3D。

数据源: T3WB 缓存 npz (pose2d_coco17 + cam3d_coco17)。
窗口构建: 对每个 (subject, action, camera) 的帧序列滑动窗口。
T3WB 是稀疏采样 (帧间隔 5-50 不等), 这里按帧序构建窗口 (因果: 只用 <= 当前帧)。

归一化协议:
- 2D: 像素 -> pelvis 相对 -> 除以 torso 像素长度
- 3D: mm -> pelvis 相对 -> 除以当前帧 2D 尺度 (2D/3D 单位一致)
- 缺失关节 (nose/eyes/ears) 填 root 值 -> 归一化后为 0, 不引入噪声
- scale 下限保护: torso 长度 < MIN_SCALE_PX 的帧用中位数替代 (防数值爆炸)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.normalize import center_at_root, normalize_scale

MIN_SCALE_PX = 10.0  # torso 像素长度下限


def build_window_indices(T, rf):
    """T 帧序列 -> 因果窗口列表。每个窗口 [i-rf+1, ..., i], 以 i 结尾。"""
    return [list(range(i - rf + 1, i + 1)) for i in range(rf - 1, T)]


def _fill_missing_with_root(pts):
    """pts: (..., J, D) 可能含 NaN。
    缺失关节填该样本的 root (l/r_hip 中点)。
    返回 (filled, root)。
    """
    pts = pts.copy()
    lh = pts[..., 11, :]
    rh = pts[..., 12, :]
    root = np.where(np.isnan(lh), 0.0, lh) + np.where(np.isnan(rh), 0.0, rh)
    root = root / 2.0
    # 如果 root 本身 NaN (l/r_hip 都缺失), 用全样本中位数兜底
    if np.isnan(root).any():
        med = np.nanmedian(pts.reshape(-1, pts.shape[-1]), axis=0)
        root = np.where(np.isnan(root), med, root)
    mask = np.isnan(pts)
    root_b = root[..., None, :]
    # 广播: root_b 形状 (..., 1, D)
    filled = pts
    filled[mask] = np.broadcast_to(root_b, pts.shape)[mask]
    return filled, root


class TemporalPoseDataset(Dataset):
    def __init__(self, npz_path, subjects=None, rf=81, stride=1):
        data = np.load(npz_path, allow_pickle=True)["data"].item()
        self.rf = rf
        self.samples = []      # (subject, action, camera, center_idx, window)
        self.cache = {}        # (subject, action, camera) -> item
        for subj, actions in data.items():
            if subjects is not None and subj not in subjects:
                continue
            for act, cams in actions.items():
                for ck, item in cams.items():
                    N = len(item["frame_id"])
                    windows = build_window_indices(N, rf)[::stride]
                    for w in windows:
                        self.samples.append((subj, act, ck, w[-1], w))
                    self.cache[(subj, act, ck)] = item

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj, act, ck, center, w = self.samples[idx]
        item = self.cache[(subj, act, ck)]
        pose2d = item["pose2d_coco17"][w]     # (rf,17,2) 像素
        cam3d = item["cam3d_coco17"][center]  # (17,3) 相机系 mm

        # ---- 2D: 缺失填 root, pelvis 相对, torso 缩放 ----
        p2d, _ = _fill_missing_with_root(np.asarray(pose2d, dtype=np.float64))
        centered2d, _ = center_at_root(p2d)
        # scale 下限保护
        raw_scale = np.linalg.norm(centered2d[:, 5, :] - centered2d[:, 11, :], axis=-1)
        safe_scale = np.where(raw_scale < MIN_SCALE_PX, np.median(raw_scale), raw_scale)
        safe_scale = np.where(safe_scale < 1e-6, 1.0, safe_scale)
        normed2d = centered2d / safe_scale[:, None, None]

        # ---- 3D: 缺失填 root, pelvis 相对, 用当前帧 2D 尺度 ----
        c3d, _ = _fill_missing_with_root(np.asarray(cam3d, dtype=np.float64))
        centered3d, _ = center_at_root(c3d)
        normed3d = centered3d / safe_scale[-1]

        x = torch.from_numpy(normed2d.reshape(self.rf, 34)).float()
        y = torch.from_numpy(normed3d.reshape(17, 3)).float()
        return x, y
