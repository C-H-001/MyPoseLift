"""可视化工具模块测试"""
import numpy as np
from src.visualize import plot_skeleton_3d, plot_skeleton_2d, COCO17_SKELETON
from src.data.joint_mapping import COCO17_SKELETON as MAP_SKELETON


def test_skeleton_connectivity():
    # 关键连接存在
    assert (5, 7) in COCO17_SKELETON   # l_shoulder->l_elbow
    assert (11, 13) in COCO17_SKELETON  # l_hip->l_knee
    assert (0, 5) in COCO17_SKELETON    # nose->l_shoulder
    assert (11, 12) in COCO17_SKELETON  # l_hip->r_hip


def test_plot_3d_saves_file(tmp_path):
    X = np.random.rand(17, 3) * 2
    out = tmp_path / "sk3d.png"
    plot_skeleton_3d(X, out, title="test")
    assert out.exists() and out.stat().st_size > 1000


def test_plot_3d_batch_and_nan(tmp_path):
    # 支持 (N,17,3) 输入 + NaN 关节不报错
    X = np.full((5, 17, 3), np.nan)
    X[:, 5:, :] = np.random.rand(5, 12, 3)
    out = tmp_path / "sk3d_batch.png"
    plot_skeleton_3d(X, out)
    assert out.exists()


def test_plot_2d_saves_file(tmp_path):
    P = np.random.rand(17, 2) * 1000
    out = tmp_path / "sk2d.png"
    plot_skeleton_2d(P, out, title="2d")
    assert out.exists() and out.stat().st_size > 1000


def test_skeleton_consistent_with_mapping():
    # 可视化骨架与关节映射模块一致
    assert COCO17_SKELETON == MAP_SKELETON
