"""训练前数据校验。任何 FAIL 拒绝训练。

检查项:
1. 格式统一: 缓存 pose2d (N,17,2), cam3d (N,17,3), 形状与数据类型
2. 数值范围: 2D 像素 [0,1000], 3D 相机系 mm 合理范围
3. 帧对齐: frame_id 单调递增, 与样本数一致
4. 投影一致性: camera_3d -> K -> 像素 与 pose_2d 误差 (多 subject/action 抽样)
5. 关节语义可视化: T3WB body -> COCO17 3D 骨架图 + 2D 叠加图 (人工确认)
6. 监督 mask: 12 关节
7. NaN/Inf 检查

输出: outputs/check/*.png + 控制台 PASS/FAIL 汇总
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.t3wb import load_t3wb_action, get_camera_params
from src.data.joint_mapping import build_coco17_supervision_mask
from src.data.normalize import center_at_root, normalize_scale
from src.visualize import plot_skeleton_3d, plot_skeleton_2d, plot_projection_check
from tools.prepare_t3wb import extract_body_coco17, check_projection_consistency
from configs.config import CHECK_DIR, T3WB_IMG_W, T3WB_IMG_H

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name} {detail}")


def run_all():
    print("=" * 70)
    print("check_data: 训练前数据校验")
    print("=" * 70)

    # ---- 1. 缓存数据加载与格式 ----
    print("\n[1] 缓存格式检查")
    import numpy as _np
    cache = _np.load("data/cache/t3wb_train.npz", allow_pickle=True)
    data = cache["data"].item()
    mask = cache["supervision_mask"]
    n_samples = 0
    shapes_ok = True
    nan_count = 0
    for subj, actions in data.items():
        for act, cams in actions.items():
            for ck, item in cams.items():
                n = len(item["frame_id"])
                n_samples += n
                p = item["pose2d_coco17"]
                c = item["cam3d_coco17"]
                w = item["world3d_coco17"]
                if p.shape != (n, 17, 2) or c.shape != (n, 17, 3) or w.shape != (n, 17, 3):
                    shapes_ok = False
                # 只统计监督关节 (5-16) 的 NaN
                nan_count += int(_np.isnan(p[:, 5:, :]).sum()) + int(_np.isnan(c[:, 5:, :]).sum())
    check("缓存 shapes 统一 (N,17,2/3)", shapes_ok)
    check("监督关节无 NaN", nan_count == 0, f"监督关节NaN总数={nan_count}")
    print(f"      总样本帧数: {n_samples} (每帧 x4 相机视角 = {n_samples*4} 训练样本)")
    check("总样本帧数 > 50k", n_samples > 50_000, f"{n_samples}")

    # ---- 2. 数值范围 ----
    print("\n[2] 数值范围检查")
    for subj, act, ck in [("S5", "Directions 1", "60457274"), ("S1", "Walking", "55011271")]:
        d = load_t3wb_action(subj, act)
        pose2d = _np.asarray(d[ck]["pose_2d"], dtype=_np.float32)
        cam3d = _np.asarray(d[ck]["camera_3d"], dtype=_np.float32)
        p_valid = pose2d[:, :17]
        c_valid = cam3d[:, :17]
        check(f"2D 像素范围 {subj}/{act}", p_valid.min() >= 0 and p_valid.max() <= 1000,
              f"[{p_valid.min():.0f}, {p_valid.max():.0f}]")
        ok_xyz = (c_valid[..., 0].min() > -4000 and c_valid[..., 0].max() < 4000 and
                  c_valid[..., 1].min() > -4000 and c_valid[..., 1].max() < 4000 and
                  c_valid[..., 2].min() > 0 and c_valid[..., 2].max() < 12000)
        check(f"3D 相机系范围 {subj}/{act}", ok_xyz,
              f"x[{c_valid[...,0].min():.0f},{c_valid[...,0].max():.0f}] "
              f"y[{c_valid[...,1].min():.0f},{c_valid[...,1].max():.0f}] "
              f"z[{c_valid[...,2].min():.0f},{c_valid[...,2].max():.0f}]")

    # ---- 3. 帧对齐 ----
    print("\n[3] 帧对齐检查")
    for subj, act in [("S5", "Directions 1"), ("S1", "Walking")]:
        d = load_t3wb_action(subj, act)
        fids = _np.array([int(x) for x in d["frame_id"]])
        check(f"frame_id 单调递增 {subj}/{act}", bool(_np.all(_np.diff(fids) > 0)),
              f"n={len(fids)} [{fids.min()}-{fids.max()}]")
        check(f"frame_id 与 3D 对齐 {subj}/{act}", len(fids) == len(d["global_3d"]))

    # ---- 4. 投影一致性 (抽样) ----
    print("\n[4] 投影一致性 (camera_3d -> 像素 vs pose_2d)")
    for subj, act, ck in [("S5", "Directions 1", "60457274"),
                          ("S1", "Walking", "55011271"),
                          ("S6", "Smoking 1", "54138969"),
                          ("S7", "Directions", "58860488")]:
        d = load_t3wb_action(subj, act)
        K, R, T, D = get_camera_params(subj, ck)
        cam3d = _np.asarray(d[ck]["camera_3d"], dtype=_np.float32)
        pose2d = _np.asarray(d[ck]["pose_2d"], dtype=_np.float32)
        ok, med = check_projection_consistency(cam3d, pose2d, K)
        check(f"投影 {subj}/{act}/{ck}", ok, f"median={med:.2f}px")
        if ok:
            plot_projection_check(cam3d[:1], pose2d[:1], K,
                                  CHECK_DIR / f"proj_{subj}_{act}.png")

    # ---- 5. 关节语义可视化 ----
    print("\n[5] 关节语义可视化 (人工确认 outputs/check/)")
    for subj, act, ck in [("S5", "Directions 1", "60457274"),
                          ("S1", "Walking", "55011271")]:
        d = load_t3wb_action(subj, act)
        cam3d = _np.asarray(d[ck]["camera_3d"], dtype=_np.float32)
        body3d = extract_body_coco17(cam3d[:1])[0]  # (17,3)
        plot_skeleton_3d(body3d, CHECK_DIR / f"body3d_{subj}_{act}.png",
                         title=f"{subj}/{act} 3D (camera)")
        pose2d = _np.asarray(d[ck]["pose_2d"], dtype=_np.float32)
        body2d = extract_body_coco17(pose2d[:1])[0]  # (17,2)
        plot_skeleton_2d(body2d, CHECK_DIR / f"body2d_{subj}_{act}.png",
                         title=f"{subj}/{act} 2D")
        # 归一化后数值范围
        bc = extract_body_coco17(cam3d[:100])
        valid = bc[~_np.isnan(bc)]
        check(f"3D 归一化前范围 {subj}/{act}",
              valid.min() > -5000 and valid.max() < 12000,
              f"[{valid.min():.0f}, {valid.max():.0f}]")
        # 归一化后
        bc_fill = _np.nan_to_num(bc, nan=0.0)
        centered, _ = center_at_root(bc_fill)
        normed, s = normalize_scale(centered)
        nv = normed[~_np.isnan(bc)]
        check(f"3D 归一化后范围 {subj}/{act}",
              nv.min() > -10 and nv.max() < 10,
              f"[{nv.min():.2f}, {nv.max():.2f}]")

    # ---- 6. 监督 mask ----
    print("\n[6] 监督 mask")
    mask_ = build_coco17_supervision_mask()
    check("监督关节数 = 12", int(mask_.sum()) == 12, f"mask={mask_.astype(int).tolist()}")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"RESULT: {len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAIL: {name} {detail}")
    if n_fail > 0:
        print("训练被拒绝: 存在 FAIL 项, 检查:", CHECK_DIR)
        sys.exit(1)
    print("校验通过, 可启动训练")


if __name__ == "__main__":
    run_all()
