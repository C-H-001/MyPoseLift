"""时序窗口数据集测试"""
import numpy as np
import torch
from src.data.dataset import TemporalPoseDataset, build_window_indices


def test_build_window_indices():
    # T=200, rf=81 -> 窗口数 = 200-80 = 120 (中心从 80 到 199)
    idx = build_window_indices(200, 81)
    assert len(idx) == 120
    assert idx[0][-1] == 80
    assert idx[-1][-1] == 199


def test_window_is_causal():
    idx = build_window_indices(200, 81)
    for w in idx:
        assert len(w) == 81
        assert np.all(np.diff(w) == 1)
        assert w[-1] == max(w)  # 最后一帧是窗口中心


def test_dataset_len_and_shapes():
    ds = TemporalPoseDataset("data/cache/t3wb_train.npz",
                             subjects=["S5"], rf=81, stride=10)
    assert len(ds) > 100
    x, y = ds[0]
    assert x.shape == (81, 34)
    assert y.shape == (17, 3)
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32


def test_dataset_values_reasonable():
    ds = TemporalPoseDataset("data/cache/t3wb_train.npz",
                             subjects=["S5"], rf=81, stride=50)
    x, y = ds[0]
    xn = x.numpy()
    # 归一化后 2D 应在合理范围 ([-5,5])
    assert np.abs(xn).max() < 10, f"2D 输入异常: max={np.abs(xn).max():.2f}"
    # 3D 归一化后 torso=1, 数值范围合理
    yn = y.numpy()
    assert np.abs(yn).max() < 10, f"3D 输出异常: max={np.abs(yn).max():.2f}"
    # 2D 首帧和末帧都不应为全零
    assert np.abs(xn[0]).sum() > 0
    assert np.abs(xn[-1]).sum() > 0


def test_dataset_all_subjects():
    ds = TemporalPoseDataset("data/cache/t3wb_train.npz",
                             subjects=["S1", "S5", "S6", "S7"], rf=81, stride=100)
    assert len(ds) > 500
