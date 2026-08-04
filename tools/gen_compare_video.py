import sys
sys.path.insert(0, "/home/user/ch/MyPoseLift")
import numpy as np
import torch
import cv2
from pathlib import Path

from src.data.dataset import TemporalPoseDataset
from src.model.tcn import TemporalConvNet
from src.data.joint_mapping import COCO17_SKELETON

RF = 27
device = "cuda"
model = TemporalConvNet(34, 17, RF, causal=True, num_layers=5, channels=1024).to(device)
ck = torch.load("/home/user/ch/MyPoseLift/outputs/ckpt/epoch_149.pth", map_location=device)
model.load_state_dict(ck["model"]); model.eval()
v = ck.get("val", 0)
print("模型: epoch %d, val %.4f (=%.1fmm)" % (ck["epoch"], float(v), float(v)*1000))

ds = TemporalPoseDataset("/home/user/ch/MyPoseLift/data/cache/h36m_valid.npz", rf=RF,
                         stride_aug=(1,), stride=1)
start = None
for i in range(len(ds)):
    s, a, c, _, _, _ = ds.samples[i]
    if s == "S9" and "Discussion 1" in a and c == "60457274":
        start = i
        break
print("起始样本:", start)
N_FRAMES = 300
preds_3d, gts_3d, inputs_2d = [], [], []
with torch.no_grad():
    for t in range(N_FRAMES):
        idx = start + t
        if idx >= len(ds):
            break
        x, y = ds[idx]
        p = model(x.unsqueeze(0).to(device)).cpu().numpy()[0]
        preds_3d.append(p)
        gts_3d.append(y.numpy())
        inputs_2d.append(x.numpy())
preds_3d = np.array(preds_3d)
gts_3d = np.array(gts_3d)
inputs_2d = np.array(inputs_2d)
T = len(preds_3d)
print("视频帧数:", T)

def draw_3d(img, X, color, offset, scale, name=None):
    h, w = img.shape[:2]
    cx, cy = offset
    pts = {}
    for j in range(17):
        if np.isnan(X[j]).any():
            continue
        xr = X[j,0]*np.cos(0.6) + X[j,2]*np.sin(0.6)
        zr = -X[j,0]*np.sin(0.6) + X[j,2]*np.cos(0.6)
        px = int(cx + xr*scale)
        py = int(cy - X[j,1]*scale)
        pts[j] = (px, py)
    for (i, j) in COCO17_SKELETON:
        if i in pts and j in pts:
            cv2.line(img, pts[i], pts[j], color, 2)
    for j, (px, py) in pts.items():
        cv2.circle(img, (px, py), 3, (0,0,255), -1)
    if name:
        cv2.putText(img, name, (cx-60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

def draw_2d(img, xf):
    h, w = img.shape[:2]
    P = xf.reshape(17, 2)
    pts = {}
    for j in range(17):
        px = int((P[j,0] + 1) / 2 * w)
        py = int((P[j,1] + 1) / 2 * h)
        pts[j] = (px, py)
    for (i, j) in COCO17_SKELETON:
        if i in pts and j in pts:
            cv2.line(img, pts[i], pts[j], (0,255,0), 2)
    for j in pts.values():
        cv2.circle(img, j, 3, (0,0,255), -1)

W, H = 1500, 500
out_path = Path("/home/user/ch/MyPoseLift/outputs/demo/gt_pred_video.mp4")
vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (W, H))
scale = 150.0
for t in range(T):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    draw_2d(frame[:, :500], inputs_2d[t, -1])
    cv2.putText(frame, "2D Input", (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    draw_3d(frame[:, 500:1000], gts_3d[t], (0,255,0), offset=(250, 250), scale=scale, name="GT 3D")
    draw_3d(frame[:, 1000:], preds_3d[t], (0,0,255), offset=(250, 250), scale=scale, name="Pred 3D")
    cv2.line(frame, (500,0), (500,H), (128,128,128), 1)
    cv2.line(frame, (1000,0), (1000,H), (128,128,128), 1)
    cv2.putText(frame, "frame %d" % t, (10, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    vw.write(frame)
    if t % 50 == 0:
        print("  %d/%d" % (t, T), flush=True)
vw.release()
print("视频已生成:", out_path)
