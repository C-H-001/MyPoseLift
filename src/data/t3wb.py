"""T3WB (H36M 全身重标注) 解析器。

文件: /mnt/disk2/ch/T3WB/h3wb_train.npz
结构:
  metadata: {subject: {camera_id: {K,R,T,Distortion}}}
            + 附加键: body/face/left_hand/right_hand/left_foot/right_foot 索引,
                      h3wb_vs_h36m 映射, flip_idx 等
  train_data: {subject: {action: {global_3d(N,133,3), frame_id(N,),
                                  camera_id: {camera_3d(N,133,3), pose_2d(N,133,2),
                                              sample_id(N,)}}}}
133 点 = 17 body + 68 face + 42 hands + 6 feet

关键约定:
- global_3d: 世界坐标系 3D (mm)
- camera_3d: 相机坐标系 3D (mm)
- pose_2d: 像素坐标 (图像 1000x1000)
- frame_id: 与 h36m.zip 图像帧名 frame_XXXX.jpg 直接对应
"""
from pathlib import Path
import numpy as np

T3WB_ROOT = Path("/mnt/disk2/ch/T3WB")
H3WB_TRAIN = T3WB_ROOT / "h3wb_train.npz"
TASK1_TEST = T3WB_ROOT / "task1_test_3d.npz"
TASK2_TEST = T3WB_ROOT / "task2_test_3d.npz"

T3WB_IMG_W, T3WB_IMG_H = 1000, 1000  # H36M 图像分辨率

# 全部 T3WB 相机 ID (H36M 4 相机)
CAMERA_IDS = ["60457274", "55011271", "54138969", "58860488"]

_cache = {}


def _load_npz(path):
    if path not in _cache:
        _cache[path] = np.load(path, allow_pickle=True)
    return _cache[path]


def load_t3wb_meta():
    """返回 metadata dict (含 h3wb_vs_h36m 等)"""
    d = _load_npz(H3WB_TRAIN)
    return d["metadata"].item()


def load_t3wb_train():
    """返回 train_data dict {subject: {action: {...}}}"""
    d = _load_npz(H3WB_TRAIN)
    return d["train_data"].item()


def load_t3wb_action(subject, action):
    """返回单个 action 的完整数据 dict"""
    td = load_t3wb_train()
    return td[subject][action]


def list_train_actions(subject=None):
    """列出训练 action。subject=None 返回 {subject: [actions]}"""
    td = load_t3wb_train()
    if subject is None:
        return {s: list(v.keys()) for s, v in td.items()}
    return list(td[subject].keys())


def get_camera_params(subject, camera_id):
    """返回 (K_norm, R, T, Distortion) 均 float32"""
    meta = load_t3wb_meta()
    cam = meta[subject][camera_id]
    return (np.asarray(cam["K"], dtype=np.float32),
            np.asarray(cam["R"], dtype=np.float32),
            np.asarray(cam["T"], dtype=np.float32),
            np.asarray(cam["Distortion"], dtype=np.float32))
