"""3D 骨架运动学增强: 关节角度扰动 (骨骼长度不变, 生理范围约束)
在相机坐标系 (x 右, y 下, z 前) 下, 绕父关节旋转, 生成蹲/跪/抱头等新姿态
"""
import numpy as np

# H36M 17 关节: 父子关系 (0=Hip 根)
PARENT = {
    0: -1,  1: 0,  2: 1,  3: 2,  4: 0,  5: 4,  6: 5,
    7: 0,   8: 7,  9: 8, 10: 9,
    11: 8, 12: 11, 13: 12, 14: 8, 15: 14, 16: 15,
}
CHILDREN = {}
for j, p in PARENT.items():
    if p >= 0:
        CHILDREN.setdefault(p, []).append(j)

# 关节活动范围 (度): (绕X轴屈曲范围, 绕Z轴外展范围)
# 相机系 y 向下: X 轴屈曲 (前后), Z 轴外展 (左右)
JOINT_LIMITS = {
    1: (-120, 30, -40, 40),    # RHip (屈/伸, 外展/内收)
    4: (-120, 30, -40, 40),    # LHip
    2: (-150, 10, -15, 15),    # RKnee (屈曲为主)
    5: (-150, 10, -15, 15),    # LKnee
    11: (-100, 170, -150, 150), # RShoulder
    14: (-100, 170, -150, 150), # LShoulder
    12: (-150, 20, -20, 20),   # RElbow
    15: (-150, 20, -20, 20),   # LElbow
    7: (-40, 40, -30, 30),     # Spine
    9: (-30, 30, -30, 30),     # Neck
}

def rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def _subtree_nodes(joint_idx):
    """返回 joint_idx 的子节点列表 (不含自身, 含后代)"""
    nodes = []
    stack = list(CHILDREN.get(joint_idx, []))
    while stack:
        n = stack.pop()
        nodes.append(n)
        stack.extend(CHILDREN.get(n, []))
    return nodes


def _apply_rotation(X, pivot_idx, joint_idx, R):
    """将 joint_idx 的子树 (不含 joint 自身) 绕 pivot_idx 位置旋转 R。
    注意: 运动学弯曲 = 旋转"远端肢体"绕关节。如弯膝 = 踝绕膝。
    """
    if pivot_idx < 0:
        return
    pivot = X[pivot_idx]
    for n in _subtree_nodes(joint_idx):
        X[n] = pivot + R @ (X[n] - pivot)

def augment_3d(pose, rng=None, num_joints=None):
    """pose: (17,3) 相机系, root 相对 (Hip=0 附近)
    随机选 num_joints 个关节, 随机弯曲 (生理范围)
    返回增强后的 (17,3) (骨骼长度不变)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    X = pose.copy().astype(np.float64)
    joints = list(JOINT_LIMITS.keys())
    n = num_joints if num_joints else int(rng.integers(1, 4))
    chosen = rng.choice(joints, size=n, replace=False)
    for j in chosen:
        if j not in JOINT_LIMITS or not CHILDREN.get(j):
            continue
        fx_min, fx_max, fz_min, fz_max = JOINT_LIMITS[j]
        # 大角度概率: 生成明显姿态 (蹲/跪/抱头)
        if rng.random() < 0.5:
            ang_x = rng.uniform(fx_min, fx_max)
        else:
            # 偏向极端 (大弯曲)
            ang_x = rng.choice([rng.uniform(fx_min, fx_min*0.5),
                                rng.uniform(fx_max*0.5, fx_max)])
        ang_z = rng.uniform(fz_min, fz_max)
        R = rot_z(ang_z) @ rot_x(ang_x)
        # 弯曲关节 j = 旋转 j 的子树绕 j (运动学)
        _apply_rotation(X, j, j, R)
    return X

# 预置姿态模板: 生成特定语义姿态
def make_squat(pose, rng=None):
    """蹲下: 双膝大弯 + 髋屈"""
    X = pose.copy().astype(np.float64)
    for j, ax in [(1, -70), (4, -70), (2, -110), (5, -110)]:
        R = rot_x(ax)
        _apply_rotation(X, j, j, R)
    return X

def make_kneel(pose, rng=None):
    """跪姿: 膝 90°+"""
    X = pose.copy().astype(np.float64)
    for j, ax in [(1, -50), (4, -50), (2, -95), (5, -95)]:
        R = rot_x(ax)
        _apply_rotation(X, j, j, R)
    return X

def make_hands_on_head(pose, rng=None):
    """抱头: 肩抬 + 肘弯"""
    X = pose.copy().astype(np.float64)
    for j, ax in [(11, 150), (14, 150), (12, -120), (15, -120)]:
        R = rot_x(ax)
        _apply_rotation(X, j, j, R)
    return X

def make_raise_arms(pose, rng=None):
    """举手: 肩大抬"""
    X = pose.copy().astype(np.float64)
    for j, ax in [(11, -160), (14, -160)]:
        R = rot_x(ax)
        _apply_rotation(X, j, j, R)
    return X

def make_bend_waist(pose, rng=None):
    """弯腰: 脊柱前屈"""
    X = pose.copy().astype(np.float64)
    R = rot_x(60)
    _apply_rotation(X, 7, 7, R)
    return X

TEMPLATES = {
    "squat": make_squat,
    "kneel": make_kneel,
    "hands_on_head": make_hands_on_head,
    "raise_arms": make_raise_arms,
    "bend_waist": make_bend_waist,
}

def apply_template(pose, name):
    return TEMPLATES[name](pose)
