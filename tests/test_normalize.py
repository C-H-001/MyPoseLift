"""归一化模块测试: pelvis 相对 + torso 长度缩放"""
import numpy as np
from src.data.normalize import center_at_root, normalize_scale, denormalize_scale, compute_torso_length

L_HIP, R_HIP = 11, 12
L_SHOULDER = 5


def test_center_at_root_2d():
    X = np.random.rand(17, 2) * 1000
    Xc, root = center_at_root(X)
    mid = (X[L_HIP] + X[R_HIP]) / 2
    np.testing.assert_allclose(root, mid, atol=1e-6)
    np.testing.assert_allclose(Xc[L_HIP] + root, X[L_HIP], atol=1e-6)  # 反变换一致
    np.testing.assert_allclose((Xc[L_HIP] + Xc[R_HIP]) / 2, 0.0, atol=1e-6)  # 根节点归零


def test_center_at_root_3d_batch():
    X = np.random.rand(4, 17, 3) * 100
    Xc, root = center_at_root(X)
    assert root.shape == (4, 3)
    mid = (X[:, L_HIP] + X[:, R_HIP]) / 2
    np.testing.assert_allclose(root, mid, atol=1e-6)


def test_torso_length():
    X = np.random.rand(17, 3) * 100
    s = compute_torso_length(X)
    expected = np.linalg.norm(X[L_SHOULDER] - (X[L_HIP] + X[R_HIP]) / 2)
    assert abs(s - expected) < 1e-5


def test_normalize_scale():
    X = np.random.rand(17, 3) * 100
    Xn, s = normalize_scale(X)
    assert abs(s - compute_torso_length(X)) < 1e-6
    # 归一化后 torso 长度 = 1
    assert abs(compute_torso_length(Xn) - 1.0) < 1e-4


def test_denormalize_roundtrip():
    X = np.random.rand(17, 3) * 100
    Xn, s = normalize_scale(X)
    Xr = denormalize_scale(Xn, s)
    np.testing.assert_allclose(Xr, X, atol=1e-4)


def test_zero_torso_safe():
    # 全零输入不应产生 NaN (防除零)
    X = np.zeros((17, 3))
    Xn, s = normalize_scale(X)
    assert np.isfinite(Xn).all()
