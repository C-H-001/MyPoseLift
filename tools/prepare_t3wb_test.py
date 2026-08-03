"""T3WB 测试集缓存 (S8): task1/task2 -> t3wb_test1.npz / t3wb_test2.npz
格式与 t3wb_train.npz 一致, 可直接复用 TemporalPoseDataset。
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.t3wb import get_camera_params
from src.data.joint_mapping import build_coco17_supervision_mask
from tools.prepare_t3wb import extract_body_coco17
from configs.config import CACHE_DIR, T3WB_DIR

TASKS = {"t3wb_test1.npz": T3WB_DIR / "task1_test_3d.npz",
         "t3wb_test2.npz": T3WB_DIR / "task2_test_3d.npz"}


def build():
    for out_name, src in TASKS.items():
        d = np.load(src, allow_pickle=True)
        data = d["data"].item()
        out = {}
        for subj, actions in data.items():
            out[subj] = {}
            for act, adata in actions.items():
                cam_keys = [k for k in adata if str(k).isdigit()]
                out[subj][act] = {}
                for ck in cam_keys:
                    K, R, T, D = get_camera_params(subj, ck)
                    cam3d = np.asarray(adata[ck]["camera_3d"], dtype=np.float32)
                    pose2d = np.asarray(adata[ck]["pose_2d"], dtype=np.float32)
                    world3d = np.asarray(adata["global_3d"], dtype=np.float32)
                    out[subj][act][ck] = {
                        "pose2d_coco17": extract_body_coco17(pose2d),
                        "cam3d_coco17": extract_body_coco17(cam3d),
                        "world3d_coco17": extract_body_coco17(world3d),
                        "frame_id": np.asarray(adata["frame_id"]),
                    }
        np.savez(CACHE_DIR / out_name,
                 data=out,
                 supervision_mask=build_coco17_supervision_mask())
        print(f"已生成 {CACHE_DIR / out_name}: "
              f"{sum(len(a) for ac in out.values() for a in ac.values())} samples")


if __name__ == "__main__":
    build()
