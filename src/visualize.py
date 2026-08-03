"""骨架可视化: 2D 叠加图 / 3D 双视角 / 投影一致性图"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.data.joint_mapping import COCO17_SKELETON


def _normalize_axes(ax, X3d):
    X = np.asarray(X3d, dtype=np.float64)
    lim = float(np.nanmax(np.abs(X))) * 1.2
    if lim < 1e-6:
        lim = 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


def plot_skeleton_3d(X3d, out_path, title="", elev=20, azim=60, figsize=(12, 5)):
    """X3d:(17,3) 或 (N,17,3)(取第0帧)。输出两视角图。NaN 关节跳过。"""
    X = np.asarray(X3d, dtype=np.float64)
    if X.ndim == 3:
        X = X[0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize)
    for i, (e, a) in enumerate([(elev, azim), (elev, azim + 90)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for (j, k) in COCO17_SKELETON:
            if np.isnan(X[j]).any() or np.isnan(X[k]).any():
                continue
            ax.plot([X[j, 0], X[k, 0]], [X[j, 1], X[k, 1]],
                    [X[j, 2], X[k, 2]], "o-", lw=2, ms=4)
        _normalize_axes(ax, X)
        ax.view_init(elev=e, azim=a)
        ax.set_title(f"{title} [{e},{a}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_skeleton_2d(pose2d, out_path, img=None, title=""):
    """pose2d:(17,2) 像素。若有 img 则叠加原图。y 轴翻转 (图像坐标)。"""
    P = np.asarray(pose2d, dtype=np.float64)
    if P.ndim == 3:
        P = P[0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    if img is not None:
        ax.imshow(img)
    for (j, k) in COCO17_SKELETON:
        if np.isnan(P[j]).any() or np.isnan(P[k]).any():
            continue
        ax.plot([P[j, 0], P[k, 0]], [P[j, 1], P[k, 1]], "o-", lw=2, ms=4, color="lime")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_projection_check(cam3d, pose2d, K_norm, out_path, img_w=1000, img_h=1000):
    """投影一致性: 3D 投影点 vs 2D GT 叠加对比 (前 17 个关节)"""
    from src.data.camera import project_to_pixel
    cam3d = np.asarray(cam3d, dtype=np.float64)
    pose2d = np.asarray(pose2d, dtype=np.float64)
    if cam3d.ndim == 3:
        cam3d = cam3d[0]
    if pose2d.ndim == 3:
        pose2d = pose2d[0]
    proj = project_to_pixel(cam3d[:17], K_norm, img_w, img_h)
    gt = pose2d[:17]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt[:, 0], gt[:, 1], "o", ms=6, label="2D GT")
    ax.plot(proj[:, 0], proj[:, 1], "x", ms=6, label="3D proj")
    for j in range(17):
        ax.plot([gt[j, 0], proj[j, 0]], [gt[j, 1], proj[j, 1]], "k-", lw=0.5)
    ax.legend()
    ax.invert_yaxis()
    ax.set_title("Projection Consistency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
