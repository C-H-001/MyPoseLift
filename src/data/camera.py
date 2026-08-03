"""相机工具: T3WB K 为归一化内参, 需要与像素坐标换算。

约定 (H36M/T3WB 标准):
- 世界系 (mm) -> 相机系: Xc = R @ Xw + T
- 相机系 -> 像素: x_pix = (fx_norm * X/Z + cx_norm) * W, y 同理 * H
  (K_norm 的 fx/cx 已除以 W/H, 故乘回 W/H 得像素)
- T3WB 的 pose_2d 是像素坐标 (图像 1000x1000)
- 相机系 3D 单位: 毫米 (mm)
"""
import numpy as np


def normalize_K(K_norm, img_w, img_h):
    """归一化内参 -> 像素内参。
    K_norm: fx_norm, fy_norm, cx_norm, cy_norm (fx_norm = fx_pix / W)。
    若 fx 已是像素单位 (>10), 直接返回。
    """
    K = np.asarray(K_norm, dtype=np.float64)
    if K.ndim == 3:
        K = K[0]
    fx, fy = K[0, 0], K[1, 1]
    if fx > 10:
        return K
    out = np.eye(3, dtype=np.float64)
    out[0, 0], out[1, 1] = fx * img_w, fy * img_h
    out[0, 2], out[1, 2] = K[0, 2] * img_w, K[1, 2] * img_h
    return out


def world_to_camera(Xw, R, T):
    """世界坐标 (N,3) -> 相机坐标 (N,3)。Xc = R @ Xw + T
    R:(3,3), T:(3,) 或 (1,3) 或 (1,1,3)。T 与 Xw 单位一致 (mm)。
    (T3WB 的原始 T 为米, 已在 t3wb.get_camera_params 中转换)
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    orig_shape = Xw.shape[:-1]
    flat = Xw.reshape(-1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3)
    out = flat @ R.T + T
    return out.reshape(*orig_shape, 3).astype(np.float32)


def camera_to_world(Xc, R, T):
    """相机坐标 (N,3) -> 世界坐标 (N,3)。Xw = R.T @ (Xc - T)
    """
    Xc = np.asarray(Xc, dtype=np.float64)
    orig_shape = Xc.shape[:-1]
    flat = Xc.reshape(-1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3)
    out = (flat - T) @ R
    return out.reshape(*orig_shape, 3).astype(np.float32)


def project_to_pixel(Xc, K_norm, img_w, img_h):
    """相机系 (N,3) mm -> 像素 (N,2)。返回 float32 像素坐标。
    假设无畸变 (T3WB 提供 Distortion, 但投影一致性检查用无畸变模型验证,
    若误差大需引入畸变校正, 见 check_data.py 阶段)。
    """
    Xc = np.asarray(Xc, dtype=np.float64)
    orig_shape = Xc.shape[:-1]
    flat = Xc.reshape(-1, 3)
    z = flat[:, 2:3]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    K = normalize_K(K_norm, img_w, img_h)
    x = K[0, 0] * flat[:, 0:1] / z + K[0, 2]
    y = K[1, 1] * flat[:, 1:2] / z + K[1, 2]
    out = np.concatenate([x, y], axis=1)
    return out.reshape(*orig_shape, 2).astype(np.float32)
