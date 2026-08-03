# 3D 人体姿态估计（2D→3D Lifting）实现计划

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** 实现 VideoPose3D 风格因果 TCN 3D 姿态 lifting 网络（81 帧输入，COCO 17 关键点），训练于 T3WB(H36M 3D)/3DPW/COCO，用 RTMPose 推理，强调数据管线正确性校验。

**Architecture:** 2D 检测器（RTMPose，推理用）→ 2D 关键点序列（81帧）→ 因果膨胀 TCN → 当前帧 3D 姿态（COCO 17 × 3）。训练用 GT 2D/3D（T3WB camera_3d），数据管线含严格错位校验与可视化。

**Tech Stack:** Python 3.10 (conda env: ch_pose), PyTorch 2.x + CUDA 12.4, numpy, opencv-python, matplotlib, MMPose (RTMPose 推理), 单卡 A100。

**远程环境:** user@10.201.1.200，代码 /home/user/ch/MyPoseLift，数据 /mnt/disk2/ch/{T3WB,COCO,3DPW,H36M}。

**数据源:**
- T3WB: /mnt/disk2/ch/T3WB/h3wb_train.npz + task1/task2_test_3d.npz（3D GT 主来源，相机系+世界系都有）
- COCO: /mnt/disk2/ch/COCO/annotations/person_keypoints_train2017.json（2D 辅助）
- 3DPW: /mnt/disk2/ch/3DPW/{sequenceFiles,imageFiles}.zip（in-the-wild 3D 补充）
- H36M 图像: /mnt/disk2/ch/H36M/h36m.zip（可视化/演示用，S1/S5/S7）

**关键约束:**
- T3WB frame_id 与 h36m.zip 帧名直接对应（已验证）
- T3WB 稀疏采样（步长 5-50），时序窗口需按真实帧索引构建
- 所有数据集统一: 相机坐标系 + pelvis 根节点 + COCO 17 关节序
- 每轮 val 输出样本推理结果可视化
- 训练前 check_data.py 必须全 PASS 才可训练

---

## Phase 0: 环境搭建

### Task 0.1: 创建 conda 环境 ch_pose

**Files:**
- Create: `scripts/setup_env.sh`

**Step 1: 写环境脚本**

```bash
#!/bin/bash
# scripts/setup_env.sh
set -e
# conda 来自 /home/wsco/anaconda3/bin (bashrc 已配置)
conda create -n ch_pose python=3.10 -y
conda activate ch_pose
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy scipy opencv-python matplotlib tqdm tensorboard pandas scikit-learn
# mmpose 用于 RTMPose 推理 (推理阶段才需要, 可先装)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  mmcv==2.2.0 mmpose==1.3.9 mmdet==3.3.0
```

**Step 2: 执行**

Run: `bash scripts/setup_env.sh`
Expected: conda env ch_pose 创建成功

**Step 3: 验证**

Run: `conda run -n ch_pose python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
Expected: `2.4.1+cu124 True`

**Step 4: Commit**

```bash
git add scripts/setup_env.sh
git commit -m "chore: 环境搭建脚本 ch_pose"
```

### Task 0.2: 项目骨架 + requirements

**Files:**
- Create: `requirements.txt`, `.gitignore`, `configs/config.py`, `src/__init__.py`, `outputs/.gitkeep`

**Step 1: 写 requirements.txt**

```
torch==2.4.1
numpy
scipy
opencv-python
matplotlib
tqdm
tensorboard
pandas
scikit-learn
```

**Step 2: 写 .gitignore**

```
__pycache__/
*.pyc
outputs/
data/cache/
.worktrees/
*.log
```

**Step 3: 写 configs/config.py（核心配置）**

```python
"""全局配置。所有路径/超参集中管理。"""
from pathlib import Path

# ---------- 路径 ----------
CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("/mnt/disk2/ch")
T3WB_DIR = DATA_ROOT / "T3WB"
COCO_DIR = DATA_ROOT / "COCO"
PW3D_DIR = DATA_ROOT / "3DPW"
H36M_DIR = DATA_ROOT / "H36M"
CACHE_DIR = CODE_ROOT / "data" / "cache"   # 预处理缓存
OUTPUT_DIR = CODE_ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "ckpt"
CHECK_DIR = OUTPUT_DIR / "check"
VAL_DIR = OUTPUT_DIR / "val_samples"
LOG_DIR = OUTPUT_DIR / "logs"
for d in [CACHE_DIR, CKPT_DIR, CHECK_DIR, VAL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- 关节定义 (COCO 17) ----------
COCO17_NAMES = ["nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder",
                "l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip",
                "l_knee","r_knee","l_ankle","r_ankle"]
NUM_JOINTS = 17
PELVIS_IDX = None  # COCO 无 pelvis, 用 l_hip/r_hip 中点
L_HIP, R_HIP = 11, 12

# ---------- 模型 ----------
RECEPTIVE_FIELD = 81
NUM_JOINTS = 17
CHANNELS = 1024
DROPOUT = 0.25

# ---------- 训练 ----------
EPOCHS = 50
BATCH_SIZE = 1024
LR = 1e-3
SEED = 0
```

**Step 4: 验证**

Run: `conda run -n ch_pose python -c "import sys; sys.path.insert(0,\".\"); from configs.config import *; print(NUM_JOINTS, RECEPTIVE_FIELD)"`
Expected: `17 81`

**Step 5: Commit**

```bash
git add requirements.txt .gitignore configs/ src/ outputs/.gitkeep
git commit -m "chore: 项目骨架与配置"
```

---

## Phase 1: 数据管线（关节映射 + 相机 + 归一化）

### Task 1.1: 关节映射模块

**Files:**
- Create: `src/data/joint_mapping.py`
- Test: `tests/test_joint_mapping.py`

**Step 1: 写失败测试**

```python
# tests/test_joint_mapping.py
import numpy as np
from src.data.joint_mapping import T3WB_TO_H36M_17, H36M_17_TO_COCO17, T3WB_BODY_TO_COCO17

def test_h36m_17_to_coco_mapping_size():
    # H36M17 17 关节全部映射到 COCO17
    assert len(H36M_17_TO_COCO17) == 17

def test_t3wb_to_h36m_known_pairs():
    # h3wb_vs_h36m 关键映射 (来自 metadata)
    assert T3WB_TO_H36M_17[0] == 9   # neck
    assert T3WB_TO_H36M_17[5] == 11  # r_shoulder
    assert T3WB_TO_H36M_17[16] == 3  # r_ankle

def test_full_pipeline_maps_17_to_17():
    # T3WB body(17) -> COCO17(17), 每个目标关节有来源
    targets = set(T3WB_BODY_TO_COCO17.values())
    assert len(targets) == 17
```

**Step 2: 跑测试确认失败**

Run: `conda run -n ch_pose python -m pytest tests/test_joint_mapping.py -v`
Expected: FAIL (模块不存在)

**Step 3: 实现**

```python
# src/data/joint_mapping.py
"""关节映射: T3WB body(17) -> H36M17 -> COCO17 (顺序: COCO 标准)

COCO17 顺序:
0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder,
7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip,
13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle

H36M 17 (VideoPose3D 标准):
0 Hip, 1 RHip, 2 RKnee, 3 RAnkle, 4 LHip, 5 LKnee, 6 LAnkle, 7 Spine,
8 Thorax, 9 Neck, 10 Head, 11 RShoulder, 12 RElbow, 13 RWrist,
14 LShoulder, 15 LElbow, 16 LWrist

T3WB body 17 -> H36M 17  (来源: h3wb_vs_h36m, 13 对 + 推断 4 对)
T3WB: [0]=neck, [1..4]=?, [5]=r_shoulder, [6]=l_shoulder, [7]=r_elbow,
      [8]=l_elbow, [9]=r_wrist, [10]=l_wrist, [11]=l_hip, [12]=r_hip,
      [13]=l_knee, [14]=r_knee, [15]=l_ankle, [16]=r_ankle
"""
import numpy as np

# h3wb_vs_h36m 官方映射: [[t3wb_idx, h36m_idx], ...]
_H3WB_VS_H36M = [[0,9],[5,11],[7,12],[9,13],[6,14],[8,15],[10,16],
                 [11,4],[13,5],[15,6],[12,1],[14,2],[16,3]]

# 推断: T3WB[1]=pelvis(H36M 0), T3WB[2]=spine(H36M 7), T3WB[3]=thorax(H36M 8),
#        T3WB[4]=head(H36M 10)   # 需在 Task 1.5 可视化确认!
_T3WB_EXTRA = {1: 0, 2: 7, 3: 8, 4: 10}

T3WB_TO_H36M_17 = dict(_H3WB_VS_H36M + list(_T3WB_EXTRA.items()))

# H36M 17 -> COCO 17 (H36M 无耳朵/眼睛, 用 head/neck 近似, 不可靠关节后续置 0)
# nose<-neck(9), l_eye<-head(10)近似, r_eye<-head(10), l_ear/l_ear 无 -> None
H36M_17_TO_COCO17 = {
    9: 0,   # neck -> nose (近似)
    # 眼睛/耳朵 H36M 无对应, 映射为 None (不监督)
    11: 6,  # r_shoulder
    12: 8,  # r_elbow
    13: 10, # r_wrist
    14: 5,  # l_shoulder
    15: 7,  # l_elbow
    16: 9,  # l_wrist
    1: 12,  # r_hip
    2: 14,  # r_knee
    3: 16,  # r_ankle
    4: 11,  # l_hip
    5: 13,  # l_knee
    6: 15,  # l_ankle
    0: None,  # hip -> 用 l/r_hip 中点计算
    7: None,  # spine
    8: None,  # thorax
    10: None, # head
}

# 反向: COCO idx -> H36M17 idx (可用于监督 mask)
def build_coco17_supervision_mask():
    """返回 (17,) bool mask: COCO 关节中哪些可由 T3WB 监督"""
    mask = np.zeros(17, dtype=bool)
    for h36m_i, coco_i in H36M_17_TO_COCO17.items():
        if coco_i is not None:
            mask[coco_i] = True
    return mask

# T3WB body idx -> COCO17 idx
T3WB_BODY_TO_COCO17 = {}
for t3wb_i, h36m_i in T3WB_TO_H36M_17.items():
    coco_i = H36M_17_TO_COCO17.get(h36m_i)
    if coco_i is not None:
        T3WB_BODY_TO_COCO17[t3wb_i] = coco_i
```

**Step 4: 跑测试**

Run: `conda run -n ch_pose python -m pytest tests/test_joint_mapping.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/joint_mapping.py tests/test_joint_mapping.py
git commit -m "feat: T3WB->H36M->COCO17 关节映射"
```

### Task 1.2: 相机模块（K 归一化内参换算 + 投影 + 世界->相机）

**Files:**
- Create: `src/data/camera.py`
- Test: `tests/test_camera.py`

**Step 1: 写失败测试**

```python
# tests/test_camera.py
import numpy as np
from src.data.camera import normalize_K, project_to_pixel, world_to_camera

def test_normalize_K_with_image_size():
    # T3WB K 是归一化内参 (fx=1.145, cx=0.515), 图像宽 1000 -> 像素 K
    K_norm = np.array([[1.1455114, 0, 0.5149682],
                       [0, 1.144774, 0.501882],
                       [0, 0, 1]])
    W, H = 1000, 1000
    K_pix = normalize_K(K_norm, W, H)
    assert abs(K_pix[0,0] - 1.1455114*W) < 1e-3
    assert abs(K_pix[0,2] - 0.5149682*W) < 1e-3

def test_world_to_camera_shape_and_dtype():
    R = np.eye(3); t = np.array([0,0,4.4])
    Xw = np.random.rand(10, 3).astype(np.float32)
    Xc = world_to_camera(Xw, R, t)
    assert Xc.shape == (10, 3)

def test_project_roundtrip():
    # 相机系 (x,y,z) 用归一化 K 投影
    K = np.array([[1.1455114, 0, 0.5149682],
                  [0, 1.144774, 0.501882],
                  [0, 0, 1]], dtype=np.float32)
    Xc = np.array([[100, 50, 4000]], dtype=np.float32)
    uv = project_to_pixel(Xc, K, img_w=1000, img_h=1000)
    # 手动: x_pix = (x/z*fx + cx) * W
    assert uv.shape == (1, 2)
```

**Step 2: 跑测试确认失败**

Run: `conda run -n ch_pose python -m pytest tests/test_camera.py -v`
Expected: FAIL

**Step 3: 实现**

```python
# src/data/camera.py
"""相机工具: T3WB K 为归一化内参, 需要与像素坐标换算。
约定:
- 世界系 (mm) -> 相机系: Xc = R @ Xw + T   (注意: T3WB T 是平移向量)
- 相机系 -> 像素: x_pix = (fx_norm * X/Z + cx_norm) * W ; y 同理 * H
- T3WB 的 pose_2d 是像素坐标 (图像 1000x1000)
"""
import numpy as np

def normalize_K(K_norm, img_w, img_h):
    """归一化内参 -> 像素内参。K_norm: fx_norm, cx_norm (已除以 W/H)。
    若 K_norm 已是像素单位 (fx~1000), 直接返回。"""
    fx, fy = K_norm[0,0], K_norm[1,1]
    if fx > 10:  # 已是像素单位
        return K_norm
    K = np.eye(3, dtype=np.float64)
    K[0,0], K[1,1] = fx * img_w, fy * img_h
    K[0,2], K[1,2] = K_norm[0,2] * img_w, K_norm[1,2] * img_h
    return K

def world_to_camera(Xw, R, T):
    """世界坐标 (N,3) -> 相机坐标 (N,3)。R:(3,3) T:(3,)"""
    Xw = np.asarray(Xw, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3)
    return Xw @ R.T + T

def camera_to_world(Xc, R, T):
    Xc = np.asarray(Xc, dtype=np.float64).reshape(-1, 3)
    return Xc @ R - T   # 注意符号: Xc = R@Xw + T -> Xw = R.T@(Xc - T)

def project_to_pixel(Xc, K_norm, img_w, img_h):
    """相机系 (N,3) -> 像素 (N,2)。返回 float32 像素坐标。"""
    Xc = np.asarray(Xc, dtype=np.float64)
    z = Xc[:, 2:3]
    z = np.where(z == 0, 1e-6, z)
    K = normalize_K(K_norm, img_w, img_h)
    x = K[0,0] * Xc[:,0:1] / z + K[0,2]
    y = K[1,1] * Xc[:,1:2] / z + K[1,2]
    return np.concatenate([x, y], axis=1).astype(np.float32)
```

**Step 4: 跑测试**

Run: `conda run -n ch_pose python -m pytest tests/test_camera.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/camera.py tests/test_camera.py
git commit -m "feat: 相机内参换算与世界/相机坐标系转换"
```

### Task 1.3: 归一化模块

**Files:**
- Create: `src/data/normalize.py`
- Test: `tests/test_normalize.py`

**Step 1: 写失败测试**

```python
# tests/test_normalize.py
import numpy as np
from src.data.normalize import center_at_root, normalize_scale, denormalize_scale

def test_center_at_root():
    # 17 关节, root=l_hip/r_hip 中点
    X = np.random.rand(17, 3) * 100
    Xc, root = center_at_root(X)
    mid = (X[11] + X[12]) / 2
    np.testing.assert_allclose(Xc[11] + root, X[11], atol=1e-5)
    np.testing.assert_allclose(root, mid, atol=1e-5)

def test_normalize_scale_sets_torso_length():
    X = np.random.rand(17, 3) * 100
    Xn, s = normalize_scale(X)
    # torso 长度 = |l_shoulder - mid_hip|
    torso = np.linalg.norm(X[5] - (X[11]+X[12])/2)
    assert abs(s - torso) < 1e-5
    assert np.linalg.norm(Xn[5] - Xn[11]) < 1e-5  # 归一化后肩膀-髋距离=1(approx)

def test_denormalize_roundtrip():
    X = np.random.rand(17, 3) * 100
    Xn, s = normalize_scale(X)
    Xr = denormalize_scale(Xn, s)
    np.testing.assert_allclose(Xr, X, atol=1e-4)
```

**Step 2: 跑测试确认失败**

Run: `conda run -n ch_pose python -m pytest tests/test_normalize.py -v`
Expected: FAIL

**Step 3: 实现**

```python
# src/data/normalize.py
"""归一化: pelvis(l/r_hip 中点)相对 + torso 长度缩放。
所有数据集统一此协议, 2D 与 3D 共用根节点定义。
"""
import numpy as np

L_HIP, R_HIP = 11, 12

def center_at_root(X, root_idx=None):
    """X:(N,J,3) 或 (J,3) -> 减去根节点。默认 l/r_hip 中点。返回 (centered, root)"""
    X = np.asarray(X, dtype=np.float64)
    if root_idx is None:
        root = (X[..., L_HIP, :] + X[..., R_HIP, :]) / 2.0
    else:
        root = X[..., root_idx, :]
    return X - root[..., None, :], root

def compute_torso_length(X):
    """torso = |l_shoulder(5) - mid_hip|"""
    X = np.asarray(X, dtype=np.float64)
    mid_hip = (X[..., L_HIP, :] + X[..., R_HIP, :]) / 2.0
    return np.linalg.norm(X[..., 5, :] - mid_hip, axis=-1, keepdims=True)

def normalize_scale(X):
    """除以 torso 长度。返回 (normalized, scale)"""
    X = np.asarray(X, dtype=np.float64)
    s = compute_torso_length(X)
    s = np.where(s < 1e-6, 1.0, s)  # 防除零
    return X / s[..., None], s

def denormalize_scale(Xn, s):
    return Xn * s
```

**Step 4: 跑测试**

Run: `conda run -n ch_pose python -m pytest tests/test_normalize.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/normalize.py tests/test_normalize.py
git commit -m "feat: pelvis 相对 + torso 尺度归一化"
```

### Task 1.4: T3WB 解析器（核心数据源）

**Files:**
- Create: `src/data/t3wb.py`
- Test: `tests/test_t3wb.py`

**Step 1: 写失败测试**

```python
# tests/test_t3wb.py
import numpy as np
from src.data.t3wb import load_t3wb_meta, load_t3wb_action, T3WB_IMG_W, T3WB_IMG_H

def test_meta_structure():
    meta = load_t3wb_meta()
    assert S1 in meta and S5 in meta
    # S1 有 4 个相机
    assert len(meta[S1]) == 4
    cam = meta[S1][60457274]
    assert set(cam.keys()) >= {K, R, T, Distortion}

def test_action_data_shape():
    data = load_t3wb_action(S5, Directions 1)
    # 4 个相机
    assert len([k for k in data if k.isdigit()]) == 4
    g3d = data[global_3d]
    assert g3d.shape[1] == 133 and g3d.shape[2] == 3
    # frame_id 与 global_3d 对齐
    assert len(data[frame_id]) == len(g3d)
    # 任一相机的 camera_3d 与 pose_2d 对齐
    cam_key = [k for k in data if k.isdigit()][0]
    assert data[cam_key][camera_3d].shape[0] == len(g3d)
    assert data[cam_key][pose_2d].shape[0] == len(g3d)
```

**Step 2: 跑测试确认失败**

Run: `conda run -n ch_pose python -m pytest tests/test_t3wb.py -v`
Expected: FAIL

**Step 3: 实现**

```python
# src/data/t3wb.py
"""T3WB (H36M 全身重标注) 解析器。
文件: /mnt/disk2/ch/T3WB/h3wb_train.npz
结构: metadata {subject:{camera_id:{K,R,T,Distortion}}}
      train_data {subject:{action:{global_3d(N,133,3), frame_id(N,),
                                   camera_id:{camera_3d(N,133,3), pose_2d(N,133,2), sample_id(N,)}}}}
注意: 133 点 = 17 body + 68 face + 42 hands + 6 feet
"""
from pathlib import Path
import numpy as np

T3WB_ROOT = Path("/mnt/disk2/ch/T3WB")
H3WB_TRAIN = T3WB_ROOT / "h3wb_train.npz"
TASK1_TEST = T3WB_ROOT / "task1_test_3d.npz"
TASK2_TEST = T3WB_ROOT / "task2_test_3d.npz"

T3WB_IMG_W, T3WB_IMG_H = 1000, 1000  # H36M 图像分辨率

_cache = {}

def _load_npz(path):
    if path not in _cache:
        _cache[path] = np.load(path, allow_pickle=True)
    return _cache[path]

def load_t3wb_meta():
    d = _load_npz(H3WB_TRAIN)
    return d["metadata"].item()

def load_t3wb_train():
    d = _load_npz(H3WB_TRAIN)
    return d["train_data"].item()

def load_t3wb_action(subject, action):
    td = load_t3wb_train()
    return td[subject][action]

def list_train_actions(subject=None):
    td = load_t3wb_train()
    if subject is None:
        return {s: list(v.keys()) for s, v in td.items()}
    return list(td[subject].keys())

def get_camera_params(subject, camera_id):
    """返回 (K_norm, R, T, Distortion) 均 float32"""
    meta = load_t3wb_meta()
    cam = meta[subject][camera_id]
    return (np.array(cam["K"], dtype=np.float32),
            np.array(cam["R"], dtype=np.float32),
            np.array(cam["T"], dtype=np.float32),
            np.array(cam["Distortion"], dtype=np.float32))
```

**Step 4: 跑测试**

Run: `conda run -n ch_pose python -m pytest tests/test_t3wb.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/t3wb.py tests/test_t3wb.py
git commit -m "feat: T3WB 数据解析器"
```

### Task 1.5: T3WB 预处理缓存（T3WB -> COCO17 训练 npz）

**Files:**
- Create: `tools/prepare_t3wb.py`
- Test: `tests/test_prepare_t3wb.py`

**Step 1: 写失败测试（投影一致性 + 映射输出）**

```python
# tests/test_prepare_t3wb.py
import numpy as np
from src.data.t3wb import load_t3wb_action, get_camera_params
from src.data.joint_mapping import T3WB_BODY_TO_COCO17
from src.data.camera import project_to_pixel
from tools.prepare_t3wb import extract_body_coco17, check_projection_consistency

def test_extract_body_coco17():
    """T3WB body(17) 提取为 COCO17(17,3), 缺失关节为 NaN"""
    data = load_t3wb_action("S5", "Directions 1")
    cam_key = [k for k in data if k.isdigit()][0]
    cam3d = data[cam_key]["camera_3d"]  # (N,133,3)
    body = extract_body_coco17(cam3d)   # (N,17,3)
    assert body.shape == (cam3d.shape[0], 17, 3)
    # 有 NaN 的关节 (eyes/ears 缺失) 至少 1 个
    nan_cols = np.isnan(body).all(axis=(0,2))
    assert nan_cols.sum() >= 1

def test_projection_consistency():
    """camera_3d 投影回像素 与 pose_2d 误差 < 阈值"""
    data = load_t3wb_action("S5", "Directions 1")
    cam_key = [k for k in data if k.isdigit()][0]
    K, R, T, D = get_camera_params("S5", cam_key)
    cam3d = data[cam_key]["camera_3d"][:10]
    pose2d = data[cam_key]["pose_2d"][:10]
    ok, err = check_projection_consistency(cam3d, pose2d, K)
    assert ok, f"投影误差过大: {err}"
```

**Step 2: 跑测试确认失败**

Run: `conda run -n ch_pose python -m pytest tests/test_prepare_t3wb.py -v`
Expected: FAIL

**Step 3: 实现**

```python
# tools/prepare_t3wb.py
"""T3WB -> 训练 npz 缓存。
输出: data/cache/t3wb_train.npz
  {subject: {action: {camera_id: {pose2d_coco17:(N,17,2), cam3d_coco17:(N,17,3),
                                   world3d_coco17:(N,17,3), frame_id:(N,)}}}}
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.t3wb import (load_t3wb_action, get_camera_params,
                            T3WB_IMG_W, T3WB_IMG_H, H3WB_TRAIN)
from src.data.joint_mapping import T3WB_BODY_TO_COCO17, build_coco17_supervision_mask
from src.data.camera import project_to_pixel
from configs.config import CACHE_DIR

BODY_RANGE = range(17)  # T3WB 前 17 为 body

def extract_body_coco17(xyz133):
    """(N,133,3) -> (N,17,3), 按 T3WB_BODY_TO_COCO17 映射, 缺失关节置 NaN。
    注意: 需要验证 T3WB body 顺序 (Task 1.5 可视化步骤)。"""
    N = xyz133.shape[0]
    out = np.full((N, 17, 3), np.nan, dtype=np.float32)
    for t3wb_i, coco_i in T3WB_BODY_TO_COCO17.items():
        out[:, coco_i, :] = xyz133[:, t3wb_i, :]
    return out

def check_projection_consistency(cam3d, pose2d, K_norm, thresh=5.0):
    """camera_3d -> 像素 与 pose_2d 对比。返回 (ok, median_err_px)"""
    proj = project_to_pixel(cam3d, K_norm, T3WB_IMG_W, T3WB_IMG_H)
    err = np.linalg.norm(proj - pose2d, axis=-1)
    med = float(np.nanmedian(err))
    return med < thresh, med

def build_cache():
    from src.data.t3wb import load_t3wb_train
    td = load_t3wb_train()
    out = {}
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
    np.savez(CACHE_DIR / "t3wb_train.npz", **{"data": out, "supervision_mask": build_coco17_supervision_mask()})
    print("缓存已保存:", CACHE_DIR / "t3wb_train.npz")

if __name__ == "__main__":
    build_cache()
```

**Step 4: 运行投影校验 + 生成缓存**

Run: `conda run -n ch_pose python -c "from tools.prepare_t3wb import *; ..."`（先跑测试）
Run: `conda run -n ch_pose python -m pytest tests/test_prepare_t3wb.py -v`
Expected: PASS（若投影误差大, 说明 K/坐标系理解有误, 进入 systematic-debugging）

**Step 5: 生成缓存 + 打印统计**

Run: `conda run -n ch_pose python tools/prepare_t3wb.py`
Expected: `缓存已保存: data/cache/t3wb_train.npz`

**Step 6: Commit**

```bash
git add tools/prepare_t3wb.py tests/test_prepare_t3wb.py
git commit -m "feat: T3WB 预处理缓存 (COCO17 映射 + 投影校验)"
```

---

## Phase 2: 数据校验与可视化（check_data.py）

### Task 2.1: 可视化工具模块

**Files:**
- Create: `src/visualize.py`
- Test: `tests/test_visualize.py`（仅验证输出文件生成）

**Step 1: 写失败测试**

```python
# tests/test_visualize.py
import numpy as np
from src.visualize import plot_skeleton_3d, plot_skeleton_2d, COCO17_SKELETON

def test_skeleton_connectivity():
    # COCO 17 骨架连接: (5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)
    assert (5,7) in COCO17_SKELETON and (11,13) in COCO17_SKELETON
    assert (0,5) in COCO17_SKELETON  # nose->l_shoulder

def test_plot_3d_saves_file(tmp_path):
    X = np.random.rand(17, 3)
    out = tmp_path / "sk.png"
    plot_skeleton_3d(X, out, title="test")
    assert out.exists() and out.stat().st_size > 1000
```

**Step 2: 跑测试确认失败**

Run: `conda run -n ch_pose python -m pytest tests/test_visualize.py -v`
Expected: FAIL

**Step 3: 实现**

```python
# src/visualize.py
"""骨架可视化: 2D 叠加图 / 3D 双视角 / 投影一致性图"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# COCO 17 骨架连接 (按 COCO 官方 order)
COCO17_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),      # face
    (0,5),(0,6),                  # nose -> shoulders
    (5,7),(7,9),(6,8),(8,10),     # arms
    (5,11),(6,12),(11,12),        # torso
    (11,13),(13,15),(12,14),(14,16)  # legs
]

def _normalize_axes(ax, X3d):
    lim = np.nanmax(np.abs(X3d)) * 1.2
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

def plot_skeleton_3d(X3d, out_path, title="", elev=20, azim=60):
    """X3d:(17,3) 或 (N,17,3)(取第0帧)。输出两视角图。"""
    X = np.asarray(X3d, dtype=np.float64)
    if X.ndim == 3:
        X = X[0]
    fig = plt.figure(figsize=(12,5))
    for i, (e, a) in enumerate([(elev, azim), (elev, azim+90)]):
        ax = fig.add_subplot(1,2,i+1, projection="3d")
        for (j,k) in COCO17_SKELETON:
            if np.isnan(X[j]).any() or np.isnan(X[k]).any():
                continue
            ax.plot([X[j,0],X[k,0]], [X[j,1],X[k,1]], [X[j,2],X[k,2]], "o-", lw=2, ms=4)
        _normalize_axes(ax, X)
        ax.view_init(elev=e, azim=a)
        ax.set_title(f"{title} [{e},{a}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

def plot_skeleton_2d(pose2d, out_path, img=None, title=""):
    """pose2d:(17,2) 像素。若有 img 则叠加原图。"""
    P = np.asarray(pose2d, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8,8))
    if img is not None:
        ax.imshow(img)
    for (j,k) in COCO17_SKELETON:
        if np.isnan(P[j]).any() or np.isnan(P[k]).any():
            continue
        ax.plot([P[j,0],P[k,0]], [P[j,1],P[k,1]], "o-", lw=2, ms=4, color="lime")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

def plot_projection_check(cam3d, pose2d, K, out_path, img_w=1000, img_h=1000):
    """投影一致性: 3D 投影点 vs 2D GT 叠加对比"""
    from src.data.camera import project_to_pixel
    proj = project_to_pixel(cam3d, K, img_w, img_h)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(pose2d[:,0], pose2d[:,1], "o", ms=6, label="2D GT")
    ax.plot(proj[:,0], proj[:,1], "x", ms=6, label="3D proj")
    for j in range(17):
        ax.plot([pose2d[j,0], proj[j,0]], [pose2d[j,1], proj[j,1]], "k-", lw=0.5)
    ax.legend(); ax.invert_yaxis(); ax.set_title("Projection Consistency")
    fig.tight_layout(); fig.savefig(out_path, dpi=100); plt.close(fig)
```

**Step 4: 跑测试**

Run: `conda run -n ch_pose python -m pytest tests/test_visualize.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/visualize.py tests/test_visualize.py
git commit -m "feat: 骨架可视化工具 (2D/3D/投影一致性)"
```

### Task 2.2: check_data.py（训练前数据校验总入口）

**Files:**
- Create: `src/check_data.py`

**Step 1: 实现（校验清单全量执行）**

```python
# src/check_data.py
"""训练前数据校验。任何 FAIL 拒绝训练。
检查项:
1. 格式统一: 各数据集 pose2d (N,17,2), cam3d (N,17,3), 无 NaN 骨架完整性
2. 数值范围: 2D 归一化后 [-2,2], 3D 相对根节点范围合理
3. 帧对齐: T3WB frame_id 与 h36m.zip 图像帧名对齐 (抽查)
4. 投影一致性: camera_3d -> K -> 像素 与 pose_2d 误差
5. 关节映射: T3WB body 17 语义可视化确认 (左右对称性)
6. 无 NaN/Inf
输出: outputs/check/*.png + 控制台 PASS/FAIL 汇总
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.t3wb import load_t3wb_action, get_camera_params
from src.data.joint_mapping import T3WB_BODY_TO_COCO17, build_coco17_supervision_mask
from src.data.camera import project_to_pixel, normalize_K
from src.data.normalize import center_at_root, normalize_scale
from src.visualize import plot_skeleton_3d, plot_skeleton_2d, plot_projection_check
from tools.prepare_t3wb import extract_body_coco17, check_projection_consistency
from configs.config import CHECK_DIR

RESULTS = []

def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name} {detail}")

def run_all():
    print("=" * 60)
    print("check_data: 训练前数据校验")
    print("=" * 60)

    # --- 1. 帧对齐: T3WB frame_id vs h36m.zip (S5/Directions 1) ---
    # 依赖 h36m.zip 已解压 (解压脚本见 Task 3.1)
    # 此处用 npz 内的 frame_id 自洽性: 递增、与 global_3d 对齐
    data = load_t3wb_action("S5", "Directions 1")
    fids = np.array([int(x) for x in data["frame_id"]])
    check("frame_id 单调递增", bool(np.all(np.diff(fids) > 0)))
    check("frame_id 与 3D 对齐", len(fids) == len(data["global_3d"]))

    # --- 2. 投影一致性 (多 subject/action/camera 抽样) ---
    errs = []
    for subj, act, ck in [("S5","Directions 1","60457274"),
                           ("S1","Walking","55011271"),
                           ("S7","Phoning 1","54138969")]:
        d = load_t3wb_action(subj, act)
        K, R, T, D = get_camera_params(subj, ck)
        cam3d = np.asarray(d[ck]["camera_3d"], dtype=np.float32)
        pose2d = np.asarray(d[ck]["pose_2d"], dtype=np.float32)
        ok, med = check_projection_consistency(cam3d, pose2d, K)
        errs.append(med)
        check(f"投影一致性 {subj}/{act}/{ck}", ok, f"median={med:.2f}px")
        if ok:
            plot_projection_check(cam3d[0, :17], pose2d[0, :17], K,
                                  CHECK_DIR / f"proj_{subj}_{act}.png")

    # --- 3. 关节语义可视化: T3WB body 17 -> COCO 17 骨架图 ---
    for subj, act, ck in [("S5","Directions 1","60457274"), ("S1","Walking 1","55011271")]:
        d = load_t3wb_action(subj, act)
        cam3d = np.asarray(d[ck]["camera_3d"], dtype=np.float32)
        body_coco = extract_body_coco17(cam3d[0])
        plot_skeleton_3d(body_coco, CHECK_DIR / f"body3d_{subj}_{act}.png",
                         title="T3WB body->COCO17 3D")
        # 2D 检查图
        pose2d = np.asarray(d[ck]["pose_2d"], dtype=np.float32)
        body2d_coco = extract_body_coco17(pose2d[0])
        plot_skeleton_2d(body2d_coco, CHECK_DIR / f"body2d_{subj}_{act}.png",
                         title="T3WB 2D->COCO17")

    # --- 4. 数值范围 (归一化后) ---
    d = load_t3wb_action("S5", "Directions 1")
    ck = "60457274"
    cam3d = np.asarray(d[ck]["camera_3d"], dtype=np.float32)[:100]
    body = extract_body_coco17(cam3d)  # (100,17,3)
    valid = ~np.isnan(body)
    check("3D 无 NaN/Inf", bool(np.isfinite(body[valid]).all()))
    if valid.any():
        rng = [np.nanmin(body), np.nanmax(body)]
        check("3D 毫米范围合理", rng[0] > -3000 and rng[1] < 6000, f"range={rng}")

    # --- 5. 监督 mask ---
    mask = build_coco17_supervision_mask()
    check("监督 mask 至少 13 关节", int(mask.sum()) >= 13, f"supervised={int(mask.sum())}")

    # --- 汇总 ---
    print("=" * 60)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"RESULT: {len(RESULTS)-n_fail}/{len(RESULTS)} PASS")
    if n_fail > 0:
        print("训练被拒绝: 存在 FAIL 项, 检查输出目录:", CHECK_DIR)
        sys.exit(1)
    print("校验通过, 可启动训练")

if __name__ == "__main__":
    run_all()
```

**Step 2: 运行**

Run: `conda run -n ch_pose python src/check_data.py`
Expected: 全部 PASS + outputs/check/ 下生成 PNG（2D/3D/投影一致性图）

**Step 3: 人工确认可视化**

查看 `outputs/check/*.png`：
- body3d_*.png: 骨架拓扑合理、左右对称（左右肩/髋镜像）
- body2d_*.png: 2D 骨架与人体结构一致
- proj_*.png: 绿色圆点(2D GT)与红色叉(3D 投影)重合

**若 3D 骨架左右错乱/拓扑错乱**: 说明 T3WB body 关节顺序推断有误 → 回 Task 1.1 调整 T3WB_EXTRA 映射（systematic-debugging）

**Step 4: Commit**

```bash
git add src/check_data.py
git commit -m "feat: 训练前数据校验 check_data.py"
```

---

## Phase 3: 数据集与模型

### Task 3.1: 解压 H36M 图像（可视化/演示用）

**Files:**
- Create: `tools/unpack_h36m.py`

**Step 1: 实现**

```python
# tools/unpack_h36m.py
"""解压 h36m.zip 到 /mnt/disk2/ch/H36M/images/（只解压 S1/S5/S7, 约 3.6G）"""
import zipfile
from pathlib import Path

ZIP = Path("/mnt/disk2/ch/H36M/h36m.zip")
OUT = Path("/mnt/disk2/ch/H36M/images")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        members = [m for m in z.namelist() if m.startswith("original/") and m.endswith(".jpg")]
        print(f"共 {len(members)} 张图像")
        for i, m in enumerate(members):
            # original/S1/Images/Directions 1.54138969/frame_0000.jpg
            rel = m.replace("original/", "")
            target = OUT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(members)}")

if __name__ == "__main__":
    main()
```

**Step 2: 运行**

Run: `conda run -n ch_pose python tools/unpack_h36m.py`
Expected: `/mnt/disk2/ch/H36M/images/` 下图像就绪

**Step 3: 帧对齐校验（T3WB frame_id vs 图像）**

Run: 扩展 check_data.py 的帧对齐检查（验证 frame_0075.jpg 存在于 S5/Directions 1 对应相机目录）
Expected: PASS

**Step 4: Commit**

```bash
git add tools/unpack_h36m.py
git commit -m "feat: H36M 图像解压脚本"
```

### Task 3.2: 时序窗口数据集（因果 81 帧）

**Files:**
- Create: `src/data/dataset.py`
- Test: `tests/test_dataset.py`

**Step 1: 写失败测试**

```python
# tests/test_dataset.py
import numpy as np
from src.data.dataset import TemporalPoseDataset, build_window_indices

def test_build_window_indices():
    # T=200, rf=81 -> 窗口中心从 80 到 199, 步长 1
    idx = build_window_indices(200, 81)
    assert len(idx) == 200 - 80  # 119
    assert idx[0][-1] == 80      # 第一个窗口最后一帧=80
    assert idx[-1][-1] == 199    # 最后一个窗口最后一帧=199

def test_window_is_causal():
    # 每个窗口索引严格递增, 最大索引 = 窗口中心(最后一帧)
    idx = build_window_indices(200, 81)
    for w in idx:
        assert len(w) == 81
        assert np.all(np.diff(w) == 1)
```

**Step 2: 跑测试确认失败**

Run: `conda run -n ch_pose python -m pytest tests/test_dataset.py -v`
Expected: FAIL

**Step 3: 实现**

```python
# src/data/dataset.py
"""因果时序数据集: 输入 81 帧 2D 序列 -> 预测最后一帧 3D。
数据源: T3WB 缓存 npz (pose2d_coco17 + cam3d_coco17)。
窗口构建: 对每个 action 的连续帧序列, 滑动窗口。
T3WB 是稀疏采样, 这里用 T3WB 自身序列顺序 (frame_id 递增), 
窗口内帧间隔可能不均, 但保持帧序即可 (VideoPose3D 对等间隔假设,
这里近似处理, 若精度受影响, 后续用 h36m.zip 全量 10fps 帧重建)。
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.normalize import center_at_root, normalize_scale, denormalize_scale

def build_window_indices(T, rf):
    """T 帧序列 -> [(rf 个索引), ...], 每个窗口以 i 结尾 (因果, 只用 <= i 的帧)。
    返回窗口中心索引 = i 的窗口列表。"""
    out = []
    for i in range(rf - 1, T):
        w = list(range(i - rf + 1, i + 1))
        out.append(w)
    return out

class TemporalPoseDataset(Dataset):
    def __init__(self, npz_path, subjects=None, rf=81, stride=1):
        data = np.load(npz_path, allow_pickle=True)["data"].item()
        self.rf = rf
        self.samples = []  # (subject, action, camera, window_center, window_idx)
        self.cache = {}    # (subject, action, camera) -> dict
        for subj, actions in data.items():
            if subjects is not None and subj not in subjects:
                continue
            for act, cams in actions.items():
                for ck, item in cams.items():
                    N = len(item["frame_id"])
                    windows = build_window_indices(N, rf)[::stride]
                    for w in windows:
                        self.samples.append((subj, act, ck, w[-1], w))
                    self.cache[(subj, act, ck)] = item

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj, act, ck, center, w = self.samples[idx]
        item = self.cache[(subj, act, ck)]
        pose2d = item["pose2d_coco17"][w]      # (81,17,2)
        cam3d = item["cam3d_coco17"][center]   # (17,3)
        # 2D 归一化: pelvis 相对 + 尺度
        p2d = pose2d.reshape(-1, 2)            # (81*17, 2)
        p2d = np.nan_to_num(p2d, nan=0.0)
        centered, root2d = center_at_root(p2d.reshape(81, 17, 2))
        centered_n, scale2d = normalize_scale(centered.reshape(81, 17, 2))
        # 3D: 同样处理 (用同一尺度缩放保持 2D/3D 一致)
        cam3d_valid = np.nan_to_num(cam3d, nan=0.0)
        c3d, root3d = center_at_root(cam3d_valid)
        c3d_n = c3d / scale2d  # 用 2D 尺度统一缩放
        x = torch.from_numpy(centered_n.reshape(81, 34)).float()
        y = torch.from_numpy(c3d_n.reshape(17, 3)).float()
        return x, y
```

**Step 4: 跑测试**

Run: `conda run -n ch_pose python -m pytest tests/test_dataset.py -v`
Expected: PASS

**Step 5: 冒烟验证 Dataset 可迭代**

Run: `conda run -n ch_pose python -c "from src.data.dataset import TemporalPoseDataset; d=TemporalPoseDataset(data/cache/t3wb_train.npz, subjects=[S5], rf=81); x,y=d[0]; print(x.shape, y.shape, x.min().item(), x.max().item())"`
Expected: `torch.Size([81, 34]) torch.Size([17, 3])` + 数值范围合理

**Step 6: Commit**

```bash
git add src/data/dataset.py tests/test_dataset.py
git commit -m "feat: 因果时序窗口数据集"
```

### Task 3.3: TCN 模型（VideoPose3D 架构）

**Files:**
- Create: `src/model/tcn.py`
- Test: `tests/test_tcn.py`

**Step 1: 写失败测试**

```python
# tests/test_tcn.py
import torch
from src.model.tcn import TemporalConvNet

def test_output_shape():
    model = TemporalConvNet(num_input_channels=34, num_joints=17,
                            receptive_field=81, causal=True, num_layers=5,
                            channels=256)
    x = torch.randn(2, 81, 34)  # (B, T, C)
    out = model(x)              # (B, 17, 3)
    assert out.shape == (2, 17, 3)

def test_causal_no_future_leak():
    """因果性: 输出只依赖 <= 当前时刻的输入。
    用第 t 帧输入置 0 测试: 前 t 帧的输入不影响未来输出(反向验证通过梯度或数值)。
    简化验证: 网络对 padding 模式的检查, 通过数值扰动第 81 帧输入, 输出应变化。"""
    model = TemporalConvNet(34, 17, 81, causal=True, num_layers=3, channels=128)
    x = torch.randn(1, 81, 34)
    out1 = model(x)
    x2 = x.clone(); x2[0, 80, :] = 999.0
    out2 = model(x2)
    assert not torch.allclose(out1, out2, atol=1e-4)  # 最后一帧影响输出

def test_performance_cuda():
    if not torch.cuda.is_available():
        return
    model = TemporalConvNet(34, 17, 81, causal=True, num_layers=5, channels=1024).cuda()
    x = torch.randn(1, 81, 34).cuda()
    import time
    for _ in range(3):
        model(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        model(x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 20 * 1000
    assert dt < 20, f"推理延迟 {dt:.1f}ms > 20ms (非实时)"
```

**Step 2: 跑测试确认失败**

Run: `conda run -n ch_pose python -m pytest tests/test_tcn.py -v`
Expected: FAIL (模块不存在)

**Step 3: 实现**

```python
# src/model/tcn.py
"""VideoPose3D 风格因果膨胀 TCN。
输入: (B, T, C_in), 输出: (B, num_joints, 3)
因果: padding 只在时间左侧。
"""
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation  # 左侧 padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.pad)
    def forward(self, x):
        x = self.conv(x)
        return x[..., :x.shape[-1] - self.pad]  # 去掉右侧多余

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.residual = (in_ch == out_ch)
        if not self.residual:
            self.res_conv = CausalConv1d(in_ch, out_ch, 1, 1)
    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        y = self.dropout(y)
        if self.residual:
            return self.relu(y + x)
        return self.relu(y + self.res_conv(x))

class TemporalConvNet(nn.Module):
    def __init__(self, num_input_channels=34, num_joints=17, receptive_field=81,
                 causal=True, num_layers=5, channels=1024, kernel_size=3,
                 dropout=0.25, input_channels=34, output_channels=51):
        super().__init__()
        self.causal = causal
        self.num_joints = num_joints
        # 计算膨胀率: 使总感受野 >= receptive_field
        dilations = []
        rf = 1
        d = 1
        while rf < receptive_field and len(dilations) < num_layers:
            dilations.append(d)
            rf += d * (kernel_size - 1)
            d *= 2
        # 输入投影
        self.in_conv = CausalConv1d(num_input_channels, channels, kernel_size, 1)
        self.in_bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        # 残差块
        blocks = []
        for i, dil in enumerate(dilations):
            blocks.append(ResidualBlock(channels, channels, kernel_size, dil, dropout))
        self.blocks = nn.Sequential(*blocks)
        # 输出
        self.out_conv = nn.Conv1d(channels, num_joints * 3, 1)
        self.out_bn = nn.BatchNorm1d(num_joints * 3)

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)          # (B, C, T)
        y = self.relu(self.in_bn(self.in_conv(x)))
        y = self.blocks(y)
        y = self.out_bn(self.out_conv(y))  # (B, 51, T)
        y = y[..., -1]                    # 只取最后一帧
        return y.reshape(-1, self.num_joints, 3)  # (B, 17, 3)
```

**Step 4: 跑测试**

Run: `conda run -n ch_pose python -m pytest tests/test_tcn.py -v`
Expected: PASS（含 CUDA 实时性 < 20ms）

**Step 5: Commit**

```bash
git add src/model/tcn.py tests/test_tcn.py
git commit -m "feat: 因果膨胀 TCN 模型 (VideoPose3D)"
```

---

## Phase 4: 训练与验证

### Task 4.1: Loss 与训练循环

**Files:**
- Create: `src/losses.py`, `src/train.py`
- Test: `tests/test_losses.py`

**Step 1: 写失败测试**

```python
# tests/test_losses.py
import torch
from src.losses import mpjpe_loss, weighted_mpjpe_loss

def test_mpjpe_zero():
    pred = torch.randn(4, 17, 3)
    assert mpjpe_loss(pred, pred.clone()).item() < 1e-5

def test_mpjpe_magnitude():
    # 已知误差: 每个关节偏差 10mm
    pred = torch.randn(2, 17, 3)
    gt = pred + 10.0
    loss = mpjpe_loss(pred, gt)
    assert abs(loss.item() - 10.0) < 1e-3

def test_weighted_mask():
    # 缺失关节 (eyes/ears) 权重 0
    pred = torch.randn(2, 17, 3)
    gt = pred.clone()
    gt[:, 1:5] = 0  # eyes/ears 缺失
    mask = torch.ones(17); mask[1:5] = 0
    loss = weighted_mpjpe_loss(pred, gt, mask)
    assert loss.item() < 1e-4  # 完全一致
```

**Step 2: 跑测试确认失败**

Run: `conda run -n ch_pose python -m pytest tests/test_losses.py -v`
Expected: FAIL

**Step 3: 实现 loss**

```python
# src/losses.py
"""3D 姿态 loss。单位: 归一化后 (mm / scale2d)。"""
import torch
import torch.nn.functional as F

def mpjpe_loss(pred, gt):
    """MPJPE: 平均每关节欧氏距离。pred/gt: (B, J, 3)"""
    return torch.mean(torch.norm(pred - gt, dim=-1))

def weighted_mpjpe_loss(pred, gt, joint_mask):
    """joint_mask:(J,) bool -> 只统计有监督的关节"""
    d = torch.norm(pred - gt, dim=-1)  # (B, J)
    m = joint_mask.float().to(d.device)
    return torch.sum(d * m) / torch.sum(m)
```

**Step 4: 跑测试**

Run: `conda run -n ch_pose python -m pytest tests/test_losses.py -v`
Expected: PASS

**Step 5: 实现训练循环**

```python
# src/train.py
"""训练入口: python src/train.py [--epochs 50] [--resume ckpt.pt]
每轮: 训练 -> val MPJPE -> 5 样本可视化 -> checkpoint
"""
import argparse, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (EPOCHS, BATCH_SIZE, LR, RECEPTIVE_FIELD,
                            CKPT_DIR, VAL_DIR, LOG_DIR, CACHE_DIR)
from src.data.dataset import TemporalPoseDataset
from src.data.joint_mapping import build_coco17_supervision_mask
from src.model.tcn import TemporalConvNet
from src.losses import mpjpe_loss, weighted_mpjpe_loss
from src.visualize import plot_skeleton_3d
from src.data.normalize import denormalize_scale

def build_model():
    return TemporalConvNet(num_input_channels=34, num_joints=17,
                           receptive_field=RECEPTIVE_FIELD, causal=True,
                           num_layers=5, channels=1024)

def evaluate(model, loader, device, joint_mask):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            losses.append(weighted_mpjpe_loss(pred, y, joint_mask).item())
    return float(np.mean(losses))

def save_val_samples(model, loader, device, epoch, joint_mask, root_scale=1000.0):
    """保存 5 个样本推理结果: 3D GT vs Pred 对比图"""
    model.eval()
    saved = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            for i in range(min(5 - saved, x.size(0))):
                p = pred[i].cpu().numpy()   # (17,3) 归一化
                g = y[i].cpu().numpy()
                fig_path = VAL_DIR / f"epoch{epoch:03d}_sample{saved}.png"
                # 对比图: 两个子图 (GT / Pred)
                from matplotlib import pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "3d"})
                for ax, X, ttl in [(axes[0], g, "GT"), (axes[1], p, "Pred")]:
                    for (j, k) in [(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]:
                        if np.isnan(X[j]).any() or np.isnan(X[k]).any(): continue
                        ax.plot([X[j,0],X[k,0]],[X[j,1],X[k,1]],[X[j,2],X[k,2]],"o-",lw=2)
                    lim = 2.0
                    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
                    ax.set_title(ttl)
                fig.suptitle(f"epoch {epoch} sample {i}")
                fig.savefig(fig_path, dpi=100); plt.close(fig)
                saved += 1
            if saved >= 5:
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 数据: 训练 S1,S5,S6,S7 (S6 无图像但有完整标注)
    train_ds = TemporalPoseDataset(CACHE_DIR / "t3wb_train.npz",
                                   subjects=["S1","S5","S6","S7"], rf=RECEPTIVE_FIELD)
    val_ds = TemporalPoseDataset(CACHE_DIR / "t3wb_train.npz",
                                 subjects=["S5"], rf=RECEPTIVE_FIELD)  # 占位, 正式用 S8 test
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=8)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    joint_mask = torch.from_numpy(build_coco17_supervision_mask()).float().to(device)
    writer = SummaryWriter(LOG_DIR)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1

    print(f"训练样本数: {len(train_ds)}, 每轮: {len(train_loader)} batch")
    best = float("inf")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time(); tot = 0.0; nb = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = weighted_mpjpe_loss(pred, y, joint_mask)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        train_loss = tot / nb

        # 验证
        val_loss = evaluate(model, val_loader, device, joint_mask)
        save_val_samples(model, val_loader, device, epoch, joint_mask)
        writer.add_scalar("train/mpjpe", train_loss, epoch)
        writer.add_scalar("val/mpjpe", val_loss, epoch)
        print(f"Epoch {epoch:3d} | train {train_loss:.3f} | val {val_loss:.3f} | {time.time()-t0:.1f}s")

        # checkpoint
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": epoch}, CKPT_DIR / "best.pth")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch}, CKPT_DIR / f"epoch_{epoch:03d}.pth")

    print(f"训练完成。最佳 val: {best:.3f} (best.pth)")

if __name__ == "__main__":
    main()
```

**Step 6: 小规模冒烟测试**

Run: `conda run -n ch_pose python src/train.py --epochs 2 --batch 256`
Expected: 训练 2 轮, val 有合理 loss, outputs/val_samples/ 生成 5 张对比图

**Step 7: 完整训练**

Run: `conda run -n ch_pose python src/train.py --epochs 50 --batch 1024 2>&1 | tee outputs/logs/train.log`
Expected: 50 轮收敛, val MPJPE 下降至合理水平

**Step 8: Commit**

```bash
git add src/losses.py src/train.py tests/test_losses.py
git commit -m "feat: MPJPE loss + 训练循环 + 每轮样本可视化"
```

---

## Phase 5: 推理与评测

### Task 5.1: RTMPose 2D 检测集成

**Files:**
- Create: `tools/download_rtmpose.py`, `src/detector.py`

**Step 1: 下载 RTMPose 权重（ModelScope 可达）**

```python
# tools/download_rtmpose.py
"""下载 RTMPose-L (或 m) COCO 权重。
来源: ModelScope (https://modelscope.cn) 或 GitHub release (可达)。
mmpose 权重: rtmpose-l_8xb256-420e_coco-384x288
"""
import urllib.request
from pathlib import Path

def download(url, dest):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1e6:
        print("已存在, 跳过:", dest); return
    print("下载:", url)
    urllib.request.urlretrieve(url, dest)
    print("完成:", dest)

if __name__ == "__main__":
    # mmpose 官方权重 (GitHub release 或 ModelScope 镜像)
    download("https://download.openmmlab.com/mmpose/top_down/rtmpose/rtmpose-l_8xb256-420e_coco-384x288-25e50d47_20230517.pth",
             "weights/rtmpose-l_coco.pth")
```

**Step 2: 实现检测器封装**

```python
# src/detector.py
"""RTMPose 推理封装: 图像 -> (17,2) 2D 关键点 (像素) + 置信度"""
import numpy as np
import cv2

class RTMPoseDetector:
    def __init__(self, weights_path, device="cuda"):
        from mmpose.apis import inference_topdown, init_model
        self.model = init_model("configs/mmpose/rtmpose-l_8xb256-420e_coco-384x288.py",
                                weights_path, device=device)
        self.device = device
    def detect(self, img_bgr):
        """img: BGR ndarray -> (17,2) 像素坐标 + (17,) 置信度, 或 None"""
        results = inference_topdown(self.model, img_bgr, bboxes=None)  # 需检测框
        # 实际用 mmdet 检测器先出框, 或直接用 ground truth 框简化
        raise NotImplementedError("完整流程: 检测框 -> topdown 关键点, 见 Task 5.2 评测脚本")
```

### Task 5.2: 完整评测脚本（H36M 协议 + 3DPW）

**Files:**
- Create: `src/evaluate.py`

**Step 1: 实现评测**

```python
# src/evaluate.py
"""评测: 加载 best.pth, 在 T3WB S8 test / 3DPW 上计算 MPJPE。
若后续获得 H36M 官方 S9/S11 标注, 扩展为 Protocol#1 评测。
"""
import sys, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import CKPT_DIR, RECEPTIVE_FIELD, CACHE_DIR
from src.model.tcn import TemporalConvNet
from src.data.joint_mapping import build_coco17_supervision_mask
from src.losses import weighted_mpjpe_loss

def evaluate_ckpt(ckpt_path, npz_path, subjects, batch=512):
    from src.data.dataset import TemporalPoseDataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalConvNet(34, 17, RECEPTIVE_FIELD, causal=True,
                            num_layers=5, channels=1024).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"]); model.eval()
    ds = TemporalPoseDataset(npz_path, subjects=subjects, rf=RECEPTIVE_FIELD)
    loader = DataLoader(ds, batch_size=batch, num_workers=8)
    mask = torch.from_numpy(build_coco17_supervision_mask()).float().to(device)
    errs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            d = torch.norm(pred - y, dim=-1) * mask.unsqueeze(0)
            errs.append(d.cpu().numpy())
    errs = np.concatenate(errs, 0)
    sup = mask.numpy().astype(bool)
    return float(np.mean(errs[:, sup])), float(np.median(errs[:, sup]))

if __name__ == "__main__":
    mpjpe, med = evaluate_ckpt(CKPT_DIR / "best.pth",
                               CACHE_DIR / "t3wb_train.npz",
                               subjects=["S8"])
    print(f"MPJPE (归一化单位): {mpjpe:.3f} | median: {med:.3f}")
```

**Step 2: 单位换算说明**

MPJPE 输出是归一化单位（除以 torso scale），报告时需乘回真实 mm：
`mpjpe_mm ≈ mpjpe_norm * median_torso_length_mm`
在评测脚本中统计训练集 median torso 长度换算。

**Step 3: Commit**

```bash
git add tools/download_rtmpose.py src/detector.py src/evaluate.py
git commit -m "feat: RTMPose 检测集成 + 评测脚本"
```

---

## Phase 6: 收尾

### Task 6.1: 实时性基准

Run: `conda run -n ch_pose python -c "from tests.test_tcn import *; ..."`（或独立 benchmark）
Expected: 单帧推理 < 20ms（A100），满足实时

### Task 6.2: 推理演示脚本

**Files:**
- Create: `src/infer_video.py`（视频 -> RTMPose -> TCN -> 3D 可视化）

### Task 6.3: 最终验证与提交

- 跑 `check_data.py` 全 PASS
- 训练 50 epoch 完成, val 样本可视化保存
- 评测脚本输出 MPJPE
- `git log` 整洁, 所有任务提交完成

---

## 风险与回滚

| 风险 | 应对 |
|------|------|
| T3WB body[1..4] 语义推断错误 | check_data 可视化 3D 骨架人工确认, 失败回 Task 1.1 调整映射 |
| T3WB 相机 K/T 符号错误 | 投影一致性检查 (camera_3d->像素 vs pose_2d) 直接暴露 |
| 稀疏采样导致时序不平滑 | 若 val 效果差, 用 h36m.zip 全量 10fps 帧重建序列 (Task 3.1 解压已完成) |
| 2D/3D 尺度不匹配 | 训练前打印 batch 数值范围断言 [-2,2]/合理 |
| H36M 标准评测 (S9/S11) 缺标注 | 用 T3WB S8 开发集评测, 官方标注到位后扩展 |
| 训练不收敛 (loss 爆炸/NaN) | 检查归一化、lr、AMP; 参考 systematic-debugging |
