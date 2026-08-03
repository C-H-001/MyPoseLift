"""全局配置。所有路径/超参集中管理。

关节约定: COCO 17 keypoints, 顺序:
0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder,
7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip,
13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
"""
from pathlib import Path

# ---------- 路径 ----------
CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("/mnt/disk2/ch")
T3WB_DIR = DATA_ROOT / "T3WB"
COCO_DIR = DATA_ROOT / "COCO"
PW3D_DIR = DATA_ROOT / "3DPW"
H36M_DIR = DATA_ROOT / "H36M"
H36M_IMAGES_DIR = H36M_DIR / "images"
CACHE_DIR = CODE_ROOT / "data" / "cache"   # 预处理缓存
OUTPUT_DIR = CODE_ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "ckpt"
CHECK_DIR = OUTPUT_DIR / "check"
VAL_DIR = OUTPUT_DIR / "val_samples"
LOG_DIR = OUTPUT_DIR / "logs"
for d in [CACHE_DIR, CKPT_DIR, CHECK_DIR, VAL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- 关节定义 (COCO 17) ----------
COCO17_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder",
                "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
                "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
NUM_JOINTS = 17
# COCO 无 pelvis, 用 l_hip/r_hip 中点作为根节点
L_HIP, R_HIP = 11, 12
L_SHOULDER, R_SHOULDER = 5, 6

# ---------- 模型 ----------
RECEPTIVE_FIELD = 81      # 输入帧数 (因果)
CHANNELS = 1024           # TCN 通道数
NUM_LAYERS = 5            # 残差块数
KERNEL_SIZE = 3
DROPOUT = 0.25
CAUSAL = True

# ---------- 训练 ----------
EPOCHS = 50
BATCH_SIZE = 1024
LR = 1e-3
SEED = 0
NUM_WORKERS = 8

# ---------- T3WB ----------
T3WB_IMG_W, T3WB_IMG_H = 1000, 1000   # H36M 图像分辨率
