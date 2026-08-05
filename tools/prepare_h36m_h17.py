"""H36M 17 点原序缓存 (不映射 COCO): part(2D) + S(3D) 直接用 H36M 17 顺序
关节顺序 (VideoPose3D): 0 Hip, 1 RHip, 2 RKnee, 3 RAnkle, 4 LHip, 5 LKnee,
6 LAnkle, 7 Spine, 8 Thorax, 9 Neck, 10 Head, 11 RShoulder, 12 RElbow,
13 RWrist, 14 LShoulder, 15 LElbow, 16 LWrist
监督: 全 17 点 (H36M 17 关节 3D 全有 GT!)
"""
import sys
from pathlib import Path
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import CACHE_DIR, H36M_DIR

ANNOT_DIR = H36M_DIR / "h36m" / "annot"


def parse_imgname(name):
    base = Path(name).stem
    left, right = base.split(".")
    subj = left.split("_")[0]
    action = left[len(subj)+1:].replace("_", " ")
    cam_str, frame_str = right.split("_")
    return subj, action, int(cam_str), int(frame_str)


def build(h5_path, imglist_path, out_path, label):
    with h5py.File(h5_path, "r") as f:
        S = f["S"][:]      # (N,17,3) 相机系 mm
        part = f["part"][:]  # (N,17,2) 像素
    with open(imglist_path) as ff:
        imgs = [l.strip() for l in ff if l.strip()]
    groups = {}
    for i, name in enumerate(imgs):
        subj, action, cam, frame = parse_imgname(name)
        groups.setdefault((subj, action, cam), []).append(i)
    out = {}
    for (subj, action, cam), idxs in groups.items():
        idxs = np.array(idxs)
        order = np.argsort(idxs)
        idxs = idxs[order]
        s3d = S[idxs].astype(np.float32)
        p2d = part[idxs].astype(np.float32)
        frames = np.array([parse_imgname(imgs[i])[3] for i in idxs], dtype=np.int32)
        out.setdefault(subj, {}).setdefault(action, {})[str(cam)] = {
            "pose2d_coco17": p2d,   # H36M 17 原序 (命名沿用, 实际 H36M 17)
            "cam3d_coco17": s3d,
            "world3d_coco17": s3d,
            "frame_id": frames.astype(str),
        }
    np.savez(out_path, data=out, supervision_mask=np.ones(17, dtype=bool))
    total = sum(len(c["frame_id"])
                for s in out.values() for acts in s.values() for c in acts.values())
    print(f"[{label}] 缓存: {out_path}, 总帧: {total}, subjects: {list(out.keys())}")


if __name__ == "__main__":
    build(ANNOT_DIR / "train.h5", ANNOT_DIR / "train_images.txt",
          CACHE_DIR / "h36m_train_h17.npz", "TRAIN-H17")
    build(ANNOT_DIR / "valid.h5", ANNOT_DIR / "valid_images.txt",
          CACHE_DIR / "h36m_valid_h17.npz", "VALID-H17")
