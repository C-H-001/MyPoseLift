"""H36M 官方预处理标注 -> 训练缓存 npz (与 T3WB 格式一致)。

数据源: /mnt/disk2/ch/H36M/h36m/annot/
  train.h5 / valid.h5: S(3D 17关节, 相机系 mm), part(2D 17关节, 原图像素)
  train_images.txt / valid_images.txt: 图像名 (S{subject}_{action}.{camera}_{frame}.jpg)

关节: H36M 17 (VideoPose3D 顺序) -> COCO17 (12 监督关节)
划分: train = S1,S5,S6,S7,S8 (312k); valid = S9,S11 (110k) -- 标准协议!
"""
import sys
from pathlib import Path
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.joint_mapping import H36M_17_TO_COCO17, build_coco17_supervision_mask
from configs.config import CACHE_DIR, H36M_DIR

ANNOT_DIR = H36M_DIR / "h36m" / "annot"
OUT_TRAIN = CACHE_DIR / "h36m_train.npz"
OUT_VALID = CACHE_DIR / "h36m_valid.npz"


def parse_imgname(name):
    """S5_Directions_1.54138969_000001.jpg -> (S5, 'Directions 1', 54138969, 1)
    格式: {subject}_{action}.{camera}_{frame}  (action 内可能含下划线)
    """
    base = Path(name).stem           # S5_Directions_1.54138969_000001
    left, right = base.split(".")    # ['S5_Directions_1', '54138969_000001']
    subj = left.split("_")[0]        # S5
    action = left[len(subj) + 1:].replace("_", " ")  # 'Directions 1'
    cam_str, frame_str = right.split("_")  # ['54138969', '000001']
    return subj, action, int(cam_str), int(frame_str)


def build(h5_path, imglist_path, out_path, label):
    with h5py.File(h5_path, "r") as f:
        S = f["S"][:]      # (N,17,3) 相机系 mm
        part = f["part"][:]  # (N,17,2) 原图像素
    with open(imglist_path) as ff:
        imgs = [l.strip() for l in ff if l.strip()]
    assert len(imgs) == len(S), f"图像数 {len(imgs)} != 标注数 {len(S)}"

    # 分组: (subject, action, camera) -> 帧索引
    groups = {}
    for i, name in enumerate(imgs):
        subj, action, cam, frame = parse_imgname(name)
        key = (subj, action, cam)
        groups.setdefault(key, []).append(i)

    print(f"[{label}] 序列数: {len(groups)}")
    out = {}
    for (subj, action, cam), idxs in groups.items():
        idxs = np.array(idxs)
        # 按图像顺序排序
        order = np.argsort(idxs)
        idxs = idxs[order]
        s3d = S[idxs]    # (T,17,3)
        p2d = part[idxs] # (T,17,2)
        # H36M17 -> COCO17 (12 监督关节), 缺失 NaN
        c3d = np.full((len(idxs), 17, 3), np.nan, dtype=np.float32)
        p2d_coco = np.full((len(idxs), 17, 2), np.nan, dtype=np.float32)
        for h36m_i, coco_i in H36M_17_TO_COCO17.items():
            if coco_i is not None:
                c3d[:, coco_i, :] = s3d[:, h36m_i, :]
                p2d_coco[:, coco_i, :] = p2d[:, h36m_i, :]
        frames = np.array([parse_imgname(imgs[i])[3] for i in idxs], dtype=np.int32)
        # 三层结构: {subject: {action: {camera: item}}}
        out.setdefault(subj, {}).setdefault(action, {})[str(cam)] = {
            "pose2d_coco17": p2d_coco,
            "cam3d_coco17": c3d,
            "world3d_coco17": c3d,  # 相机系即使用值
            "frame_id": frames.astype(str),
        }
        if len(out) <= 2:
            print(f"  示例: {subj}/{action}/cam{cam}: T={len(idxs)}")
    np.savez(out_path, data=out, supervision_mask=build_coco17_supervision_mask())
    total = sum(len(c["frame_id"])
                for s in out.values()
                for acts in s.values()
                for c in acts.values())
    print(f"[{label}] 缓存: {out_path}, 总帧: {total}, subjects: {list(out.keys())}")


if __name__ == "__main__":
    build(ANNOT_DIR / "train.h5", ANNOT_DIR / "train_images.txt", OUT_TRAIN, "TRAIN")
    build(ANNOT_DIR / "valid.h5", ANNOT_DIR / "valid_images.txt", OUT_VALID, "VALID")
