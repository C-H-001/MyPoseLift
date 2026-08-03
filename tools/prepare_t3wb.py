"""T3WB -> 训练 npz 缓存。

输出: data/cache/t3wb_train.npz
  data: {subject: {action: {camera_id: {
      pose2d_coco17: (N,17,2),   # 像素, COCO17 顺序, 缺失关节 NaN
      cam3d_coco17:  (N,17,3),   # 相机系 mm, COCO17 顺序, 缺失关节 NaN
      world3d_coco17:(N,17,3),   # 世界系 mm, COCO17 顺序, 缺失关节 NaN
      frame_id: (N,)}}}}
  supervision_mask: (17,) bool
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.t3wb import load_t3wb_train, get_camera_params, T3WB_IMG_W, T3WB_IMG_H
from src.data.joint_mapping import T3WB_BODY_TO_COCO17, build_coco17_supervision_mask
from src.data.camera import project_to_pixel
from configs.config import CACHE_DIR

OUT_PATH = CACHE_DIR / "t3wb_train.npz"


def extract_body_coco17(xyz133):
    """(N,133,D) -> (N,17,D), 按 T3WB_BODY_TO_COCO17 映射, 缺失关节置 NaN。
    D=2 (2D) 或 3 (3D)。
    """
    xyz = np.asarray(xyz133, dtype=np.float32)
    N, D = xyz.shape[0], xyz.shape[2]
    out = np.full((N, 17, D), np.nan, dtype=np.float32)
    for t3wb_i, coco_i in T3WB_BODY_TO_COCO17.items():
        out[:, coco_i, :] = xyz[:, t3wb_i, :]
    return out


def check_projection_consistency(cam3d, pose2d, K_norm, thresh=5.0):
    """camera_3d -> 像素 与 pose_2d 对比 (无畸变模型)。
    返回 (ok, median_err_px)。仅比较前 17 个 body 关节 (监督集)。
    """
    cam3d = np.asarray(cam3d, dtype=np.float32)
    pose2d = np.asarray(pose2d, dtype=np.float32)
    proj = project_to_pixel(cam3d, K_norm, T3WB_IMG_W, T3WB_IMG_H)
    err = np.linalg.norm(proj - pose2d, axis=-1)  # (N,133)
    med = float(np.nanmedian(err[:, :17]))
    return med < thresh, med


def build_cache(verbose=True):
    td = load_t3wb_train()
    out = {}
    total_actions = 0
    n_fail = 0
    for subj, actions in td.items():
        out[subj] = {}
        for act, data in actions.items():
            cam_keys = [k for k in data if k.isdigit()]
            out[subj][act] = {}
            for ck in cam_keys:
                K, R, T, D = get_camera_params(subj, ck)
                cam3d = np.asarray(data[ck]["camera_3d"], dtype=np.float32)
                pose2d = np.asarray(data[ck]["pose_2d"], dtype=np.float32)
                world3d = np.asarray(data["global_3d"], dtype=np.float32)
                out[subj][act][ck] = {
                    "pose2d_coco17": extract_body_coco17(pose2d),
                    "cam3d_coco17": extract_body_coco17(cam3d),
                    "world3d_coco17": extract_body_coco17(world3d),
                    "frame_id": np.asarray(data["frame_id"]),
                }
                if verbose:
                    ok, med = check_projection_consistency(cam3d, pose2d, K)
                    n = len(cam3d)
                    flag = "OK" if ok else "FAIL"
                    if not ok:
                        n_fail += 1
                    print(f"  [{subj}/{act}/{ck}] N={n} 投影中位误差={med:.2f}px {flag}")
            total_actions += 1
    np.savez(OUT_PATH,
             data=out,
             supervision_mask=build_coco17_supervision_mask())
    print(f"缓存已保存: {OUT_PATH}")
    print(f"subjects: {list(out.keys())}, 总 actions: {total_actions}, 投影FAIL数: {n_fail}")
    return out


if __name__ == "__main__":
    build_cache()
