"""关节映射模块测试: T3WB body(17) -> H36M17 -> COCO17

关键事实:
- H36M 17 关节中, 只有 12 个能可靠映射到 COCO 17 (肩肘腕髋膝踝)
- COCO 的 eyes(1,2)/ears(3,4) H36M 无对应
- H36M neck(9)->COCO nose(0) 是错误语义近似, 不采用 (保守: 不监督 nose)
- pelvis/spine/thorax/head 是 H36M 内部关节, 不映射到 COCO
"""
import numpy as np
from src.data.joint_mapping import (
    T3WB_TO_H36M_17,
    H36M_17_TO_COCO17,
    T3WB_BODY_TO_COCO17,
    build_coco17_supervision_mask,
)


def test_h36m_17_to_coco_mapping_size():
    # H36M17 17 关节全部有映射条目 (含 None)
    assert len(H36M_17_TO_COCO17) == 17


def test_t3wb_to_h36m_known_pairs():
    # h3wb_vs_h36m 官方映射关键对 (来自 T3WB metadata)
    assert T3WB_TO_H36M_17[0] == 9   # neck
    assert T3WB_TO_H36M_17[5] == 11  # r_shoulder
    assert T3WB_TO_H36M_17[6] == 14  # l_shoulder
    assert T3WB_TO_H36M_17[11] == 4  # l_hip
    assert T3WB_TO_H36M_17[16] == 3  # r_ankle


def test_t3wb_to_h36m_full_17():
    # T3WB body 前 17 个关节全部有 H36M 映射
    assert len(T3WB_TO_H36M_17) == 17
    assert set(T3WB_TO_H36M_17.keys()) == set(range(17))
    assert set(T3WB_TO_H36M_17.values()) == set(range(17))


def test_t3wb_to_coco_effective_mapping():
    # T3WB 能监督的 COCO 关节 = 12 个 (肩肘腕髋膝踝)
    # 不监督: nose(0), eyes(1,2), ears(3,4) [H36M 无可靠对应]
    assert len(T3WB_BODY_TO_COCO17) == 12
    targets = set(T3WB_BODY_TO_COCO17.values())
    assert targets == {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}


def test_supervision_mask():
    mask = build_coco17_supervision_mask()
    assert mask.shape == (17,)
    assert mask.dtype == bool
    assert mask.sum() == 12
    # 不监督的关节
    assert not mask[0]  # nose
    assert not mask[1] and not mask[2]  # eyes
    assert not mask[3] and not mask[4]  # ears
    # 监督的关节
    assert mask[5] and mask[6] and mask[16]
