"""H36M 2D 数据: h5 part + 图像 -> COCO json (H36M 17 点)
帧匹配: 按 (subj, action, cam) 组内顺序对应 (images.txt 顺序 == h36m.zip 帧顺序)
"""
import sys, json
sys.path.insert(0, "/home/user/ch/MyPoseLift")
import h5py
import numpy as np
from pathlib import Path

H5_TRAIN = Path("/mnt/disk2/ch/H36M/h36m/annot/train.h5")
IMG_LIST_TRAIN = Path("/mnt/disk2/ch/H36M/h36m/annot/train_images.txt")
IMG_ROOT = Path("/mnt/disk2/ch/H36M/images")
OUT_DIR = Path("/home/user/ch/MyPoseLift/data/cache/h36m_2d")

JOINT_NAMES = ["Hip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","Spine",
               "Thorax","Neck","Head","RShoulder","RElbow","RWrist",
               "LShoulder","LElbow","LWrist"]
SKELETON = [[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],
            [15,16],[0,1],[1,2],[2,3],[0,4],[4,5],[5,6]]


def parse_name(name):
    """S1_Directions_1.54138969_000001.jpg -> (S1, 'Directions 1', '54138969', 1)"""
    base = name.replace(".jpg", "")
    left, right = base.split(".")
    subj = left.split("_")[0]
    action = left[len(subj)+1:].replace("_", " ")
    cam_str, _ = right.split("_")
    return subj, action, cam_str


def build():
    with h5py.File(H5_TRAIN, "r") as f:
        part = f["part"][:]
        center = f["center"][:]
        scale = f["scale"][:]
    with open(IMG_LIST_TRAIN) as ff:
        imgs = [l.strip() for l in ff if l.strip()]

    # 分组: (subj, action, cam) -> [(img_idx, part_idx)]
    groups = {}
    for i, name in enumerate(imgs):
        subj, action, cam = parse_name(name)
        if subj not in ("S1", "S5", "S7"):
            continue
        groups.setdefault((subj, action, cam), []).append(i)

    images, annotations = [], []
    n_total, n_skip = 0, 0
    for (subj, action, cam), idxs in groups.items():
        # 该组对应的实际帧目录
        d = IMG_ROOT / subj / "Images" / f"{action}.{cam}"
        if not d.is_dir():
            n_skip += len(idxs)
            continue
        frames = sorted(d.glob("frame_*.jpg"))
        if len(frames) < len(idxs):
            # 目录帧数少于标注数, 按顺序匹配可用部分
            pass
        for k, i in enumerate(idxs):
            if k >= len(frames):
                break
            img_rel = f"{subj}/Images/{action}.{cam}/{frames[k].name}"
            images.append({
                "id": len(images), "file_name": img_rel, "width": 1000, "height": 1002,
            })
            kp = part[i].astype(float)
            kps_flat = np.zeros((17, 3))
            kps_flat[:, :2] = kp
            kps_flat[:, 2] = 1.0
            w = 200 * scale[i]; h = 200 * scale[i]
            x0 = center[i][0] - w/2; y0 = center[i][1] - h/2
            annotations.append({
                "id": len(annotations), "image_id": len(images)-1,
                "category_id": 1, "num_keypoints": 17,
                "keypoints": kps_flat.reshape(-1).tolist(),
                "bbox": [x0, y0, w, h], "area": w*h, "iscrowd": 0,
            })
            n_total += 1

    ann = {
        "info": {"description": "H36M 2D keypoints 17 joints", "year": 2026},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person",
                        "keypoints": JOINT_NAMES, "skeleton": SKELETON}],
    }
    out = OUT_DIR / "annotations" / "h36m_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(ann, f)
    print(f"图像 {len(images)}, 标注 {len(annotations)}, 跳过 {n_skip}")


if __name__ == "__main__":
    build()
