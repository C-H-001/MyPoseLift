"""T3WB 预处理缓存测试: 投影一致性 + COCO17 提取"""
import numpy as np
from src.data.t3wb import load_t3wb_action, get_camera_params
from tools.prepare_t3wb import extract_body_coco17, check_projection_consistency


def test_extract_body_coco17_3d():
    """T3WB body(133->17) 提取为 COCO17(17,3), 未监督关节为 NaN"""
    data = load_t3wb_action("S5", "Directions 1")
    cam_key = "60457274"
    cam3d = data[cam_key]["camera_3d"]  # (N,133,3)
    body = extract_body_coco17(cam3d)   # (N,17,3)
    assert body.shape == (cam3d.shape[0], 17, 3)
    # 12 个监督关节有值, 5 个缺失 (nose/eyes/ears)
    nan_cols = np.isnan(body).all(axis=(0, 2))
    assert nan_cols.sum() == 5
    assert body.shape[0] > 100


def test_extract_body_coco17_2d():
    data = load_t3wb_action("S5", "Directions 1")
    pose2d = data["60457274"]["pose_2d"]  # (N,133,2)
    body = extract_body_coco17(pose2d)
    assert body.shape == (pose2d.shape[0], 17, 2)
    # 监督关节 2D 值在合理像素范围 (0-1000)
    valid = body[:, 5:, :]
    assert np.nanmin(valid) >= 0 and np.nanmax(valid) <= 1000


def test_projection_consistency_small_error():
    """camera_3d 投影回像素 与 pose_2d 误差应 < 5px (相机参数正确性验证)"""
    data = load_t3wb_action("S5", "Directions 1")
    cam_key = "60457274"
    K, R, T, D = get_camera_params("S5", cam_key)
    cam3d = data[cam_key]["camera_3d"][:50]
    pose2d = data[cam_key]["pose_2d"][:50]
    ok, err = check_projection_consistency(cam3d, pose2d, K)
    assert ok, f"投影误差过大: {err:.2f}px"
    assert err < 5.0


def test_projection_consistency_wrong_K_fails():
    """错误的 K (如 cx 偏移) 应导致投影误差大 -> 校验能发现问题"""
    data = load_t3wb_action("S5", "Directions 1")
    cam_key = "60457274"
    K, R, T, D = get_camera_params("S5", cam_key)
    K_bad = K.copy()
    K_bad[0, 0, 2] += 0.1  # 偏移 cx
    cam3d = data[cam_key]["camera_3d"][:50]
    pose2d = data[cam_key]["pose_2d"][:50]
    ok, err = check_projection_consistency(cam3d, pose2d, K_bad)
    assert not ok or err > 5.0
