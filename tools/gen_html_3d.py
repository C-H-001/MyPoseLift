"""生成 GT vs Pred 3D 可旋转对比 HTML (Plotly)
每个样本: 左右两个场景 (GT 绿色 / Pred 红色), 可鼠标旋转/缩放
"""
import sys
sys.path.insert(0, "/home/user/ch/MyPoseLift")
import numpy as np
import torch
import plotly.graph_objects as go
from pathlib import Path

from src.data.dataset import TemporalPoseDataset
from src.model.tcn import TemporalConvNet
from src.data.joint_mapping import COCO17_SKELETON, build_coco17_supervision_mask

RF = 27
device = "cuda"

# 加载模型
model = TemporalConvNet(34, 17, RF, causal=True, num_layers=5, channels=1024).to(device)
ck = torch.load("/home/user/ch/MyPoseLift/outputs/ckpt/epoch_149.pth", map_location=device)
model.load_state_dict(ck["model"]); model.eval()
print(f"模型: epoch {ck['epoch']}")

# 验证集样本
ds = TemporalPoseDataset("/home/user/ch/MyPoseLift/data/cache/h36m_valid.npz", rf=RF,
                         stride_aug=(1,), stride=100)
print("验证样本数:", len(ds))

# 取 3 个样本
import random
random.seed(0)
idxs = random.sample(range(len(ds)), min(3, len(ds)))

JOINT_NAMES = ["Hip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","Spine",
               "Thorax","Neck","Head","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist"]

def add_skeleton(fig, X, name, color):
    """X: (17,3) mm root 相对, 添加骨架线条 + 关节点"""
    # 线条
    for (i, j) in COCO17_SKELETON:
        if np.isnan(X[i]).any() or np.isnan(X[j]).any():
            continue
        fig.add_trace(go.Scatter3d(
            x=[X[i,0], X[j,0]], y=[X[i,1], X[j,1]], z=[X[i,2], X[j,2]],
            mode="lines", line=dict(color=color, width=5),
            showlegend=False, hoverinfo="skip"))
    # 关节点
    fig.add_trace(go.Scatter3d(
        x=X[:,0], y=X[:,1], z=X[:,2], mode="markers+text",
        marker=dict(size=4, color=color),
        text=[JOINT_NAMES[i] for i in range(17)],
        textposition="top center", textfont=dict(size=8),
        name=name))

# 生成 HTML
html_parts = []
for k, idx in enumerate(idxs):
    x, y = ds[idx]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).cpu().numpy()[0]  # (17,3) mm
    gt = y.numpy()  # (17,3) mm

    fig = go.Figure()
    # 两个场景: GT 左, Pred 右 (用 domain 分行)
    for side, X, nm, col in [("left", gt, "GT", "lime"), ("right", pred, "Pred", "red")]:
        pass
    # 简化: 一个图里 GT + Pred 叠加 (不同颜色), 可旋转
    add_skeleton(fig, gt, "GT (绿)", "lime")
    add_skeleton(fig, pred, "Pred (红)", "red")
    fig.update_layout(
        title=f"Sample {k}: GT vs Pred (mm, root-relative) - 拖拽旋转/滚轮缩放",
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data"),
        width=900, height=700,
        legend=dict(x=0.8, y=0.95))
    html = fig.to_html(full_html=True, include_plotlyjs="cdn" if k > 0 else True)
    if k > 0:
        # 只保留 <div> 部分, 不重复引入 plotlyjs
        start = html.find("<div>")
        html = html[start:]
    html_parts.append(html)

# 组装
out = Path("/home/user/ch/MyPoseLift/outputs/demo/gt_pred_3d.html")
if len(html_parts) == 1:
    out.write_text(html_parts[0])
else:
    head = html_parts[0][:html_parts[0].find("<div>")]
    body = "".join(html_parts)
    out.write_text(head + body)
print("已生成:", out)
