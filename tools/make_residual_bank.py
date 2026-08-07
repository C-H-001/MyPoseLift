"""制作 detector residual bank: RTMW 检测 2D (H36M17) vs h5 GT part 残差
输出: data/cache/residual_bank.npy (N, 17, 2) 归一化 [-1,1] 坐标残差
"""
import sys
sys.path.insert(0, "/home/user/ch/MyPoseLift")
import numpy as np
import cv2, h5py
from pathlib import Path
from mmpose.apis import inference_topdown, init_model
from mmengine.config import Config

# RTMW 检测器
cfg = Config.fromfile("/tmp/mmpose_repo/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py")
det = init_model(cfg, "/home/user/ch/MyPoseLift/weights/rtmw-m_cocktail14.pth", device="cuda")

# H36M 图像 + GT
IMG_ROOT = Path("/mnt/disk2/ch/H36M/images")
with h5py.File("/mnt/disk2/ch/H36M/h36m/annot/train.h5", "r") as f:
    part = f["part"][:]
with open("/mnt/disk2/ch/H36M/h36m/annot/train_images.txt") as ff:
    imgs = [l.strip() for l in ff if l.strip()]

def parse_name(name):
    base = name.replace(".jpg", "")
    left, right = base.split(".")
    subj = left.split("_")[0]
    action = left[len(subj)+1:].replace("_", " ")
    cam = right.split("_")[0]
    return subj, action, cam

# 取 S1/S5/S7 的部分样本 (每 50 帧抽 1, 约 1200 帧)
samples = []
for i, name in enumerate(imgs):
    subj, action, cam = parse_name(name)
    if subj not in ("S1", "S5", "S7"):
        continue
    if i % 50 != 0:
        continue
    d = IMG_ROOT / subj / "Images" / f"{action}.{cam}"
    if not d.is_dir():
        continue
    frames = sorted(d.glob("frame_*.jpg"))
    if not frames:
        continue
    samples.append((i, frames[0]))
print(f"待处理样本: {len(samples)}")

def extract_h36m17(kpt133):
    out = np.zeros((17, 2), dtype=np.float32)
    out[1] = kpt133[12]; out[2] = kpt133[14]; out[3] = kpt133[16]
    out[4] = kpt133[11]; out[5] = kpt133[13]; out[6] = kpt133[15]
    out[11] = kpt133[6]; out[12] = kpt133[8]; out[13] = kpt133[10]
    out[14] = kpt133[5]; out[15] = kpt133[7]; out[16] = kpt133[9]
    hip_c = (kpt133[11] + kpt133[12]) / 2
    sh_c = (kpt133[5] + kpt133[6]) / 2
    out[0] = hip_c; out[8] = sh_c
    out[7] = hip_c + (sh_c - hip_c) * (1/3)
    face = kpt133[23:91]
    top = face[np.argmin(face[:, 1])]
    out[10] = top
    out[9] = sh_c + (top - sh_c) * 0.35
    return out

def norm2d(points):
    p = points.copy()
    p[..., 0] = 2.0 * p[..., 0] / 1000.0 - 1.0
    p[..., 1] = 2.0 * p[..., 1] / 1002.0 - 1.0
    return p

residuals = []
for k, (idx, frame_path) in enumerate(samples):
    img = cv2.imread(str(frame_path))
    h, w = img.shape[:2]
    res = inference_topdown(det, img, bboxes=[[0, 0, w-1, h-1]])
    kpt133 = np.asarray(res[0].pred_instances.keypoints)[0]
    det17 = extract_h36m17(kpt133)
    gt17 = part[idx]
    r = norm2d(det17) - norm2d(gt17)
    residuals.append(r)
    if k % 200 == 0:
        print(f"  {k}/{len(samples)}", flush=True)

bank = np.stack(residuals).astype(np.float32)
np.save("/home/user/ch/MyPoseLift/data/cache/residual_bank.npy", bank)
print("residual bank:", bank.shape, "| mean|r|:", np.abs(bank).mean())
