"""完整视频推理: RTMW -> H36M17 -> TCN -> 3D (无帧数限制)"""
import sys
sys.path.insert(0, "/home/user/ch/MyPoseLift")
import numpy as np
import torch
import cv2
from pathlib import Path

from src.model.tcn import TemporalConvNet
from src.data.dataset import _fill_missing_with_root

RF = 27
device = "cuda"

from mmpose.apis import inference_topdown, init_model
from mmengine.config import Config
rtmw_cfg = Config.fromfile("/tmp/mmpose_repo/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py")
det = init_model(rtmw_cfg, "/home/user/ch/MyPoseLift/weights/rtmw-m_cocktail14.pth", device=device)

model = TemporalConvNet(34, 17, RF, causal=True, num_layers=5, channels=1024).to(device)
ck = torch.load("/home/user/ch/MyPoseLift/outputs/ckpt/best.pth", map_location=device)
model.load_state_dict(ck["model"]); model.eval()
print("模型 OK")


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


def process(video_path, out_dir, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    kpts_seq, frames = [], []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        if max_frames and n > max_frames:
            break
        h, w = frame.shape[:2]
        res = inference_topdown(det, frame, bboxes=[[0, 0, w-1, h-1]])
        kpt133 = np.asarray(res[0].pred_instances.keypoints)[0]
        kpts_seq.append(extract_h36m17(kpt133))
        frames.append(frame)
        if n % 60 == 0:
            print(f"  帧 {n}", flush=True)
    cap.release()
    print(f"共 {len(kpts_seq)} 帧, fps={fps:.1f}")

    preds = []
    with torch.no_grad():
        for t in range(len(kpts_seq)):
            w0 = max(0, t - RF + 1)
            win = np.array(kpts_seq[w0:t+1])
            if len(win) < RF:
                win = np.concatenate([np.repeat(win[:1], RF-len(win), axis=0), win], axis=0)
            p2d, _ = _fill_missing_with_root(win.astype(np.float64))
            normed2d = p2d / np.array([1000.0, 1000.0]) * 2.0 - 1.0
            x = torch.from_numpy(normed2d.reshape(1, RF, 34)).float().to(device)
            preds.append(model(x).cpu().numpy()[0] * 1000.0)
    preds = np.array(preds)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(out_dir / f"frame_{i:03d}.jpg"), f)
    np.savez(out_dir / "pred.npz", kpts2d=np.array(kpts_seq), preds3d=preds, fps=fps)
    print(f"保存: {out_dir} ({len(frames)} 帧)")
    return out_dir, fps


if __name__ == "__main__":
    video = sys.argv[1]
    tag = Path(video).stem
    out = f"/home/user/ch/MyPoseLift/outputs/demo/rtmw_{tag}"
    process(video, out)
