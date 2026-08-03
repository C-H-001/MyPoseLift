"""T3WB 测试集缓存 (S8): task1/task2 -> t3wb_test1.npz / t3wb_test2.npz

已知数据 bug (T3WB task2): pose_2d 大量为 0 (缺失标注)。
修复: 对 (x,y)==(0,0) 的点, 用 camera_3d 投影生成 2D (3D 标注完整且投影精确)。
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.t3wb import get_camera_params
from src.data.joint_mapping import build_coco17_supervision_mask
from src.data.camera import project_to_pixel
from tools.prepare_t3wb import extract_body_coco17
from configs.config import CACHE_DIR, T3WB_DIR, T3WB_IMG_W, T3WB_IMG_H

TASKS = {"t3wb_test1.npz": T3WB_DIR / "task1_test_3d.npz",
         "t3wb_test2.npz": T3WB_DIR / "task2_test_3d.npz"}


def fix_zero_2d(pose2d, cam3d, K):
    """(x,y)==(0,0) 的 2D 点用 camera_3d 投影替换。
    返回 (fixed_pose2d, n_fixed)。"""
    pose2d = np.asarray(pose2d, dtype=np.float32).copy()
    cam3d = np.asarray(cam3d, dtype=np.float32)
    is_zero = (pose2d == 0).all(axis=-1)  # (N,133)
    n_fixed = int(is_zero.sum())
    if n_fixed > 0:
        proj = project_to_pixel(cam3d, K, T3WB_IMG_W, T3WB_IMG_H)
        pose2d[is_zero] = proj[is_zero]
    return pose2d, n_fixed


def build():
    for out_name, src in TASKS.items():
        d = np.load(src, allow_pickle=True)
        data = d["data"].item()
        out = {}
        total_fixed = 0
        for subj, actions in data.items():
            out[subj] = {}
            for act, adata in actions.items():
                cam_keys = [k for k in adata if str(k).isdigit()]
                out[subj][act] = {}
                for ck in cam_keys:
                    K, R, T, D = get_camera_params(subj, ck)
                    cam3d = np.asarray(adata[ck]["camera_3d"], dtype=np.float32)
                    pose2d_raw = np.asarray(adata[ck]["pose_2d"], dtype=np.float32)
                    pose2d_fix, nf = fix_zero_2d(pose2d_raw, cam3d, K)
                    total_fixed += nf
                    world3d = np.asarray(adata["global_3d"], dtype=np.float32)
                    out[subj][act][ck] = {
                        "pose2d_coco17": extract_body_coco17(pose2d_fix),
                        "cam3d_coco17": extract_body_coco17(cam3d),
                        "world3d_coco17": extract_body_coco17(world3d),
                        "frame_id": np.asarray(adata["frame_id"]),
                    }
        np.savez(CACHE_DIR / out_name,
                 data=out,
                 supervision_mask=build_coco17_supervision_mask())
        print(f"已生成 {CACHE_DIR / out_name}: 修复 2D=0 点数: {total_fixed}")


if __name__ == "__main__":
    build()
