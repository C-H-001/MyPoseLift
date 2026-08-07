"""生成 5 个 3D 增强示例 (蹲/跪/抱头/举手/弯腰) 可视化"""
import sys
sys.path.insert(0, "/home/user/ch/MyPoseLift")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from src.data.augment_3d import apply_template, augment_3d
from src.data.dataset import TemporalPoseDataset

SKEL = [(0,7),(7,8),(8,9),(9,10),(8,11),(11,12),(12,13),(8,14),(14,15),
        (15,16),(0,1),(1,2),(2,3),(0,4),(4,5),(5,6)]

# 取一个站立样本的 3D (root 相对, 米) -> mm
ds = TemporalPoseDataset("/home/user/ch/MyPoseLift/data/cache/h36m_train_h17.npz",
                         rf=27, stride_aug=(1,), stride=100, root_idx=0)
x, y, m = ds[0]
orig = y.numpy() * 1000.0  # 米 -> mm, (17,3) root 相对

# 5 个增强: 模板 + 随机
names = ["squat", "kneel", "hands_on_head", "raise_arms", "bend_waist"]
variants = [apply_template(orig, n) for n in names]

def mp(x, y, z):
    return x, z, -y  # matplotlib: x右, z深度, -y上

def plot_skel(ax, X):
    segments = []
    for (i, j) in SKEL:
        segments.append([mp(*X[i]), mp(*X[j])])
    lc = Line3DCollection(segments, colors="cyan", linewidths=2.5)
    ax.add_collection3d(lc)
    pts = np.array([mp(*p) for p in X])
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=15, color="red")
    zmax = np.abs(X).max() * 1.1
    ax.set_xlim(-zmax, zmax); ax.set_ylim(-zmax, zmax); ax.set_zlim(-zmax, zmax)
    ax.set_axis_off()

# 可视化: 2x3 grid (原始 + 5 增强)
fig = plt.figure(figsize=(15, 9))
titles = ["Original (Walking)", "Squat", "Kneel", "Hands-on-head", "Raise arms", "Bend waist"]
all_X = [orig] + variants
for k, X in enumerate(all_X):
    ax = fig.add_subplot(2, 3, k+1, projection="3d")
    plot_skel(ax, X)
    ax.set_title(titles[k], color="black", fontsize=11)
    ax.view_init(elev=15, azim=60)
fig.tight_layout()
out = "/home/user/ch/MyPoseLift/outputs/demo/augment_demo.png"
fig.savefig(out, dpi=100, facecolor="white")
print("已保存:", out)

# 打印增强后的关键差异 (膝/髋角度变化)
for name, X in zip(names, variants):
    # 膝角度: 大腿(LHip->LKnee) vs 小腿(LKnee->LAnkle) 夹角
    def angle(a, b, c):
        v1 = a - b; v2 = c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))
    knee_l = angle(X[4], X[5], X[6])
    hip_l = angle(X[7], X[4], X[5]) if False else None
    print(f"{name}: LKnee 角={knee_l:.0f}° (原始 {angle(orig[4], orig[5], orig[6]):.0f}°)")
