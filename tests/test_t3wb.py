"""T3WB 解析器测试"""
import numpy as np
from src.data.t3wb import (
    load_t3wb_meta, load_t3wb_train, load_t3wb_action,
    list_train_actions, get_camera_params,
    H3WB_TRAIN, T3WB_IMG_W, T3WB_IMG_H,
)


def test_meta_structure():
    meta = load_t3wb_meta()
    assert "S1" in meta and "S5" in meta and "S6" in meta and "S7" in meta
    # 每个 subject 有 4 个相机
    assert len(meta["S1"]) == 4
    cam = meta["S1"]["60457274"]
    assert set(cam.keys()) >= {"K", "R", "T", "Distortion"}


def test_meta_contains_mapping():
    meta = load_t3wb_meta()
    assert "h3wb_vs_h36m" in meta
    assert len(meta["h3wb_vs_h36m"]) == 13


def test_action_data_shape():
    data = load_t3wb_action("S5", "Directions 1")
    cam_keys = [k for k in data if k.isdigit()]
    assert len(cam_keys) == 4
    g3d = data["global_3d"]
    assert g3d.shape[1] == 133 and g3d.shape[2] == 3
    assert len(data["frame_id"]) == len(g3d)
    # 每个相机数据与 global_3d 对齐
    for ck in cam_keys:
        assert data[ck]["camera_3d"].shape[0] == len(g3d)
        assert data[ck]["pose_2d"].shape[0] == len(g3d)
        assert data[ck]["camera_3d"].shape[1:] == (133, 3)
        assert data[ck]["pose_2d"].shape[1:] == (133, 2)


def test_frame_id_aligns_with_3d():
    data = load_t3wb_action("S5", "Directions 1")
    fids = np.array([int(x) for x in data["frame_id"]])
    assert np.all(np.diff(fids) > 0)  # 单调递增


def test_get_camera_params():
    K, R, T, D = get_camera_params("S1", "60457274")
    assert K.shape == (1, 3, 3)
    assert R.shape == (1, 3, 3)
    assert T.shape == (1, 1, 3)
    assert D.shape == (1, 5)
    # K 为归一化内参
    assert K[0, 0, 0] < 10


def test_list_train_actions():
    acts = list_train_actions("S5")
    assert "Directions 1" in acts
    assert len(acts) >= 20
