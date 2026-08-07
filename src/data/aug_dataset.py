"""增强数据集: 3D clip -> augment_sequence -> (noisy 2D, 3D target)
与现有 TemporalPoseDataset 输出格式一致: (x: rf×34, y: 17×3, m: 17)
"""
import sys
sys.path.insert(0, "/home/user/ch/MyPoseLift")
import numpy as np
import torch
from torch.utils.data import Dataset

from src.augmentation.sequence_augmentation import (
    CameraModel, AugmentationConfig, ResidualBank, augment_sequence)
from src.data.dataset import build_window_indices


class AugmentedPoseDataset(Dataset):
    """从 3D 全序列用增强器生成 (2D noisy, 3D target)。

    2D 输出: [-1,1] 归一化 (与 RTMW 检测一致)
    3D 输出: root(Hip)相对, 米制 (与 lifting 训练一致)
    """

    def __init__(self, npz_path, rf=27, subjects=None, stride=1, stride_aug=(1, 2, 3),
                 camera=None, config=None, residual_bank=None, seed=0):
        _npz = np.load(npz_path, allow_pickle=True)
        self.supervision_mask = np.asarray(_npz["supervision_mask"], dtype=bool)
        data = _npz["data"].item()
        self.rf = rf
        self.samples = []      # (subj, act, cam, center, window, stride_in)
        self.cache = {}
        for subj, actions in data.items():
            if subjects is not None and subj not in subjects:
                continue
            for act, cams in actions.items():
                for ck, item in cams.items():
                    N = len(item["frame_id"])
                    for sin in stride_aug:
                        windows = build_window_indices(N, rf, sin)[::stride]
                        for w in windows:
                            self.samples.append((subj, act, ck, w[-1], w, sin))
                    self.cache[(subj, act, ck)] = item

        # 相机 (H36M 默认: fx~1145, 1000x1002, 距离~4500mm)
        self.camera = camera or CameraModel(
            focal_length=1145.0, image_size=(1000, 1002),
            camera_distance=4500.0, projection="perspective")
        # 增强配置 (修正后: 噪声幅度匹配真实检测水平)
        # 实测 RTMW 残差 mean|r|=0.13 归一化, residual_scale=0.2 -> ~2% 图像宽 (合理)
        self.config = config or AugmentationConfig(
            rotation_degrees=(10.0, 5.0, 3.0),
            scale_range=(0.95, 1.05),
            residual_rho=0.8,
            residual_scale=0.2,
            dropout_probability=0.03,
            max_dropout_span=2,
            root_index=0,
        )
        self.bank = residual_bank
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj, act, ck, center, w, _ = self.samples[idx]
        item = self.cache[(subj, act, ck)]
        cam3d = item["cam3d_coco17"][w]  # (rf,17,3) mm (H36M 17 原序)

        # 3D 增强 (clip 级旋转/缩放 + 投影 + 噪声 + dropout)
        sample = augment_sequence(
            cam3d, self.camera, config=self.config,
            residual_bank=self.bank, rng=self.rng)

        x = sample.keypoints_2d.reshape(self.rf, 34).astype(np.float32)  # [-1,1]
        y = (sample.target_3d / 1000.0).astype(np.float32)               # mm->米 root相对
        # mask: 中心帧可见性 & 全局监督
        m = self.supervision_mask & sample.visibility[sample.center_frame]
        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).float(),
                torch.from_numpy(m.astype(np.float32)).float())
