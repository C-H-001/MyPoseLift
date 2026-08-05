"""端到端数据: 图像 + 2D part + 3D S 对齐
输出: data/cache/e2e_h36m.npz (每样本: img_path, part_2d(17,2), s_3d(17,3))
3D: root(Hip) 相对, 米制 (与 lifting 一致)
"""
import sys, json
sys.path.insert(0, "/home/user/ch/MyPoseLift")
import h5py
import numpy as np
from pathlib import Path

H5 = Path("/mnt/disk2/ch/H36M/h36m/annot/train.h5")
IMG_LIST = Path("/mnt/disk2/ch/H36M/h36m/annot/train_images.txt")
IMG_ROOT = Path("/mnt/disk2/ch/H36M/images")
OUT = Path("/home/user/ch/MyPoseLift/data/cache/e2e_h36m.npz")


def parse_name(name):
    base = name.replace(".jpg", "")
    left, right = base.split(".")
    subj = left.split("_")[0]
    action = left[len(subj)+1:].replace("_", " ")
    cam_str, _ = right.split("_")
    return subj, action, cam_str


def build():
    with h5py.File(H5, "r") as f:
        part = f["part"][:]
        S = f["S"][:]
    with open(IMG_LIST) as ff:
        imgs = [l.strip() for l in ff if l.strip()]

    groups = {}
    for i, name in enumerate(imgs):
        subj, action, cam = parse_name(name)
        if subj not in ("S1", "S5", "S7"):
            continue
        groups.setdefault((subj, action, cam), []).append(i)

    samples = []  # (img_path, part_2d, s_3d)
    n_skip = 0
    for (subj, action, cam), idxs in groups.items():
        d = IMG_ROOT / subj / "Images" / f"{action}.{cam}"
        if not d.is_dir():
            n_skip += len(idxs)
            continue
        frames = sorted(d.glob("frame_*.jpg"))
        for k, i in enumerate(idxs):
            if k >= len(frames):
                break
            s3d = S[i].astype(np.float32)          # (17,3) 相机系 mm
            s3d_rel = s3d - s3d[0:1, :]            # root=Hip 相对 mm
            s3d_m = s3d_rel / 1000.0               # -> 米
            samples.append({
                "img": str(frames[k]),
                "part": part[i].astype(np.float32),  # (17,2) 像素
                "s3d": s3d_m,                        # (17,3) 米 root 相对
            })
    # 保存为 numpy 结构
    img_paths = np.array([s["img"] for s in samples], dtype=object)
    parts = np.stack([s["part"] for s in samples])
    s3ds = np.stack([s["s3d"] for s in samples])
    np.savez(OUT, img_paths=img_paths, parts=parts, s3ds=s3ds)
    print(f"样本: {len(samples)}, 跳过: {n_skip}")
    print(f"part 范围: [{parts.min():.0f},{parts.max():.0f}], s3d 范围: [{s3ds.min():.3f},{s3ds.max():.3f}]")


if __name__ == "__main__":
    build()
