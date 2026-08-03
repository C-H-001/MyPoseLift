"""归一化: pelvis(l/r_hip 中点)相对 + torso 长度缩放。

所有数据集统一此协议, 2D 与 3D 共用根节点定义 (l/r_hip 中点)。
- 2D: 像素 -> 根节点相对 -> 除以 torso 像素长度
- 3D: mm -> 根节点相对 -> 除以 torso 毫米长度
- 2D/3D 使用各自尺度, 训练时 loss 在归一化空间计算
"""
import numpy as np

# COCO 17 索引
L_HIP, R_HIP = 11, 12
L_SHOULDER = 5


def center_at_root(X, root_idx=None):
    """X:(...,J,D) -> 减去根节点。默认 l/r_hip 中点。返回 (centered, root)
    root 形状与 X 去掉最后一维 (关节维) 相同。
    """
    X = np.asarray(X, dtype=np.float64)
    if root_idx is None:
        root = (X[..., L_HIP, :] + X[..., R_HIP, :]) / 2.0
    else:
        root = X[..., root_idx, :]
    return X - root[..., None, :], root


def compute_torso_length(X):
    """torso = |l_shoulder(5) - mid_hip|。X:(...,J,D) -> (...,1)"""
    X = np.asarray(X, dtype=np.float64)
    mid_hip = (X[..., L_HIP, :] + X[..., R_HIP, :]) / 2.0
    return np.linalg.norm(X[..., L_SHOULDER, :] - mid_hip, axis=-1, keepdims=True)


def normalize_scale(X):
    """除以 torso 长度。返回 (normalized, scale)。scale 形状同 torso (...,1)"""
    X = np.asarray(X, dtype=np.float64)
    s = compute_torso_length(X)
    s = np.where(s < 1e-6, 1.0, s)  # 防除零
    return X / s[..., None], s


def denormalize_scale(Xn, s):
    """乘回 scale。"""
    return np.asarray(Xn, dtype=np.float64) * np.asarray(s, dtype=np.float64)[..., None]
