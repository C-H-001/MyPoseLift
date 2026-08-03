"""3DPW -> 训练 npz 缓存。

输出: data/cache/pw3d_train.npz
  data: {sequence: {camera: {pose2d_coco17, cam3d_coco17, frame_id}}}
  (与 t3wb_train.npz 格式一致, 可复用 TemporalPoseDataset)

3DPW 结构 (pkl, python2 pickle):
  jointPositions: (n_people, T, 24*3)  SMPL 24 关节, 世界系, 米
  cam_poses: (T, 4, 4) 世界->相机 外参
  cam_intrinsics: (3,3)
  poses2d: (n_people, T, 3, 18)  2D 关节 (含置信度)

SMPL 24 -> COCO 17 映射 (12 监督关节, 与 T3WB 一致):
  SMPL: 1 l_hip, 2 r_hip, 3 l_knee, 4 r_knee, 5 l_ankle, 6 r_ankle,
        14 l_shoulder, 15 r_shoulder, 16 l_elbow, 17 r_elbow,
        18 l_wrist, 19 r_wrist
  COCO: 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle,
        5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist
"""
import sys
import pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.joint_mapping import build_coco17_supervision_mask
from configs.config import CACHE_DIR, PW3D_DIR

SEQ_ROOT = PW3D_DIR / "sequenceFiles" / "sequenceFiles"
OUT_PATH = CACHE_DIR / "pw3d_train.npz"

# SMPL24 idx -> COCO17 idx
# 3DPW jointPositions 实际顺序 (跨帧投票+骨骼验证确定):
#   0 pelvis, 1 l_hip, 2 r_hip, [3?], 4 l_knee, 5 r_knee, [6?],
#   7 l_ankle, 8 r_ankle, [9-15 躯干/胸/颈], 16 l_shoulder, 17 r_shoulder,
#   18 l_elbow, 19 r_elbow, 20 l_wrist, 21 r_wrist, [22-23 手]
SMPL_TO_COCO17 = {
    1: 11, 2: 12,                          # l_hip, r_hip
    4: 13, 5: 14,                          # l_knee, r_knee
    7: 15, 8: 16,                          # l_ankle, r_ankle
    16: 5, 17: 6,                          # l_shoulder, r_shoulder
    18: 7, 19: 8,                          # l_elbow, r_elbow
    20: 9, 21: 10,                         # l_wrist, r_wrist
}
MM_PER_M = 1000.0


def load_sequence(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def seq_to_coco(data):
    """pkl -> (T, 17, 3) 相机系 3D + (T, 17, 2) 2D 投影 + frame_id"""
    jp = np.array(data["jointPositions"])     # (nP, T, 24*3)
    cam_poses = np.array(data["cam_poses"])   # (T, 4, 4)
    K = np.array(data["cam_intrinsics"])      # (3,3)
    T_frames = jp.shape[1]

    # 取第一个人 (3DPW 多人, 取第一个)
    # 注意: jointPositions 与 cam_poses 平移列单位都是【米】, 保持米计算, 最后转 mm
    jp0 = jp[0].reshape(T_frames, 24, 3)  # 米

    # 世界 -> 相机: X_cam = cam_pose @ [X_world; 1]
    cam3d = np.zeros((T_frames, 24, 3))
    valid = np.zeros(T_frames, dtype=bool)
    for t in range(T_frames):
        P = cam_poses[t]  # (4,4)
        Xw = np.concatenate([jp0[t], np.ones((24, 1))], axis=1)  # (24,4)
        Xc = (P @ Xw.T).T[:, :3]  # (24,3) 米
        cam3d[t] = Xc * MM_PER_M  # 米 -> mm
        valid[t] = bool(np.isfinite(Xc).all())

    # 映射到 COCO17 (12 监督关节), 缺失关节 NaN
    out3d = np.full((T_frames, 17, 3), np.nan)
    for smpl_i, coco_i in SMPL_TO_COCO17.items():
        out3d[:, coco_i, :] = cam3d[:, smpl_i, :]

    # 2D 投影 (针孔, 无畸变)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = np.where(cam3d[:, :, 2:3] == 0, 1e-6, cam3d[:, :, 2:3])
    x = fx * cam3d[:, :, 0:1] / z + cx
    y = fy * cam3d[:, :, 1:2] / z + cy
    proj = np.concatenate([x, y], axis=2).astype(np.float32)  # (T,24,2)
    out2d = np.full((T_frames, 17, 2), np.nan)
    for smpl_i, coco_i in SMPL_TO_COCO17.items():
        out2d[:, coco_i, :] = proj[:, smpl_i, :]

    frame_id = np.array(data["img_frame_ids"]).astype(str)
    return out2d, out3d, frame_id, valid


def build_cache():
    out = {}
    for split in ["train"]:
        split_dir = SEQ_ROOT / split
        for pkl_path in sorted(split_dir.glob("*.pkl")):
            seq_name = pkl_path.stem
            try:
                data = load_sequence(pkl_path)
            except Exception as e:
                print(f"  跳过 {seq_name}: {e}")
                continue
            p2d, c3d, fids, valid = seq_to_coco(data)
            out[seq_name] = {
                "cam0": {
                    "pose2d_coco17": p2d,
                    "cam3d_coco17": c3d,
                    "world3d_coco17": c3d,  # 3DPW 无独立世界系缓存, 占位
                    "frame_id": fids,
                }
            }
            n_valid = int(valid.sum())
            print(f"  {seq_name}: T={len(fids)}, 有效帧={n_valid}/{len(fids)}")
    np.savez(OUT_PATH,
             data=out,
             supervision_mask=build_coco17_supervision_mask())
    print(f"3DPW 缓存已保存: {OUT_PATH}")
    return out


if __name__ == "__main__":
    build_cache()
