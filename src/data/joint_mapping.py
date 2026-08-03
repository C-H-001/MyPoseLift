"""关节映射: T3WB body(17) -> H36M17 -> COCO17

H36M 17 (VideoPose3D 标准):
0 Hip, 1 RHip, 2 RKnee, 3 RAnkle, 4 LHip, 5 LKnee, 6 LAnkle, 7 Spine,
8 Thorax, 9 Neck, 10 Head, 11 RShoulder, 12 RElbow, 13 RWrist,
14 LShoulder, 15 LElbow, 16 LWrist

COCO 17 顺序:
0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder,
7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip,
13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle

监督策略 (保守):
- COCO 12 个肢体关节 (5-16) 由 H36M 可靠映射 -> 监督
- COCO nose(0): H36M neck/head 与鼻尖语义不同, 映射会造成系统性偏差 -> 不监督
- COCO eyes(1,2)/ears(3,4): H36M 无对应 -> 不监督
- H36M pelvis/spine/thorax/head 为 H36M 内部关节, 不映射到 COCO
"""
import numpy as np

# h3wb_vs_h36m 官方映射: [[t3wb_idx, h36m_idx], ...] (来自 T3WB metadata)
_H3WB_VS_H36M = [[0, 9], [5, 11], [7, 12], [9, 13], [6, 14], [8, 15],
                 [10, 16], [11, 4], [13, 5], [15, 6], [12, 1], [14, 2], [16, 3]]

# 推断补全: T3WB[1]=pelvis(H36M 0), [2]=spine(H36M 7), [3]=thorax(H36M 8),
#            [4]=head(H36M 10)
# 注: 这 4 对只影响 H36M 内部关节 (0,7,8,10), 均不映射到 COCO 监督,
#     故即使推断有误也不影响 12 关节监督集 (Task 1.5 可视化会确认语义)
_T3WB_EXTRA = {1: 0, 2: 7, 3: 8, 4: 10}

T3WB_TO_H36M_17 = dict(_H3WB_VS_H36M + list(_T3WB_EXTRA.items()))

# H36M 17 -> COCO 17 (值 None 表示无可靠对应, 不监督)
H36M_17_TO_COCO17 = {
    0: None,   # Hip (pelvis) -> 根节点, 不映射
    1: 12,     # RHip -> r_hip
    2: 14,     # RKnee -> r_knee
    3: 16,     # RAnkle -> r_ankle
    4: 11,     # LHip -> l_hip
    5: 13,     # LKnee -> l_knee
    6: 15,     # LAnkle -> l_ankle
    7: None,   # Spine
    8: None,   # Thorax
    9: None,   # Neck (与 COCO nose 语义不同, 不近似)
    10: None,  # Head
    11: 6,     # RShoulder -> r_shoulder
    12: 8,     # RElbow -> r_elbow
    13: 10,    # RWrist -> r_wrist
    14: 5,     # LShoulder -> l_shoulder
    15: 7,     # LElbow -> l_elbow
    16: 9,     # LWrist -> l_wrist
}

# T3WB body idx -> COCO17 idx (有效监督映射, 12 个)
T3WB_BODY_TO_COCO17 = {}
for t3wb_i, h36m_i in T3WB_TO_H36M_17.items():
    coco_i = H36M_17_TO_COCO17.get(h36m_i)
    if coco_i is not None:
        T3WB_BODY_TO_COCO17[t3wb_i] = coco_i

# COCO17 骨架连接 (可视化用)
COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # face
    (0, 5), (0, 6),                         # nose -> shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 11), (6, 12), (11, 12),             # torso
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
]


def build_coco17_supervision_mask():
    """返回 (17,) bool: COCO 关节中哪些可由 T3WB 监督 (12 个肢体关节)"""
    mask = np.zeros(17, dtype=bool)
    for coco_i in T3WB_BODY_TO_COCO17.values():
        mask[coco_i] = True
    return mask
