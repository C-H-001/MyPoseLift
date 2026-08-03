"""相机模块测试: T3WB K 为归一化内参, 需与像素换算; 世界/相机系转换"""
import numpy as np
from src.data.camera import normalize_K, project_to_pixel, world_to_camera, camera_to_world


def test_normalize_K_with_image_size():
    # T3WB K 是归一化内参 (fx~1.145, cx~0.515), 图像 1000x1000
    K_norm = np.array([[1.1455114, 0, 0.5149682],
                       [0, 1.144774, 0.501882],
                       [0, 0, 1]])
    K_pix = normalize_K(K_norm, 1000, 1000)
    assert abs(K_pix[0, 0] - 1.1455114 * 1000) < 1e-3
    assert abs(K_pix[1, 1] - 1.144774 * 1000) < 1e-3
    assert abs(K_pix[0, 2] - 0.5149682 * 1000) < 1e-3
    assert abs(K_pix[1, 2] - 0.501882 * 1000) < 1e-3


def test_normalize_K_already_pixel():
    # 已是像素内参 (fx~1000) 时原样返回
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=np.float64)
    out = normalize_K(K, 1000, 1000)
    np.testing.assert_allclose(out, K)


def test_world_to_camera_roundtrip():
    R = np.eye(3)
    t = np.array([0.05, 0.38, 4.4])
    Xw = np.random.rand(10, 3).astype(np.float32)
    Xc = world_to_camera(Xw, R, t)
    Xw_back = camera_to_world(Xc, R, t)
    np.testing.assert_allclose(Xw_back, Xw, atol=1e-4)


def test_world_to_camera_with_rotation():
    # Xc = R @ Xw + T (T3WB 约定)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    t = np.array([1, 2, 3], dtype=np.float64)
    Xw = np.array([[10, 20, 30]], dtype=np.float64)
    Xc = world_to_camera(Xw, R, t)
    expected = R @ Xw.T + t[:, None]
    np.testing.assert_allclose(Xc[0], expected[:, 0], atol=1e-5)


def test_project_to_pixel():
    # 相机系 (x,y,z) mm, 归一化 K
    K = np.array([[1.1455114, 0, 0.5149682],
                  [0, 1.144774, 0.501882],
                  [0, 0, 1]], dtype=np.float32)
    Xc = np.array([[100, 50, 4000]], dtype=np.float32)
    uv = project_to_pixel(Xc, K, 1000, 1000)
    # 手动计算: x_pix = (x/z * fx + cx) * W
    x_manual = (100 / 4000 * 1.1455114 + 0.5149682) * 1000
    y_manual = (50 / 4000 * 1.144774 + 0.501882) * 1000
    assert abs(uv[0, 0] - x_manual) < 0.01
    assert abs(uv[0, 1] - y_manual) < 0.01


def test_project_returns_float32():
    K = np.eye(3, dtype=np.float32)
    Xc = np.random.rand(5, 3).astype(np.float32)
    uv = project_to_pixel(Xc, K, 1000, 1000)
    assert uv.dtype == np.float32
