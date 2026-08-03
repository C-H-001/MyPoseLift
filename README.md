# MyPoseLift: 2D → 3D 人体姿态估计（VideoPose3D 风格因果 TCN）

基于 2D 检测 → 3D lifting 的因果 3D 姿态估计。参考 VideoPose3D（TCN + 残差），输入 81 帧 2D 关键点序列（COCO 17），输出当前帧 3D 姿态，满足 GPU 实时性（A100 单帧 2.2ms）。

## 架构

```
[视频帧] → [RTMPose 2D检测] → [17kp 2D序列 81帧] → [因果膨胀TCN] → [17kp 3D]
                                          ↑
                             训练用 GT 2D (T3WB), 推理用 RTMPose
```

- 训练: GT 2D/3D（先验证 lifting 网络本身）
- 推理: RTMPose（mmpose）检测 2D → TCN lifting → 3D

## 数据管线（重点校验区域）

| 数据集 | 来源 | 用途 |
|--------|------|------|
| T3WB (H36M 全身重标注) | /mnt/disk2/ch/T3WB/ | 主训练 (3D GT, 相机系/世界系) |
| 3DPW | /mnt/disk2/ch/3DPW/ | in-the-wild 补充 |
| COCO 2017 | /mnt/disk2/ch/COCO/ | 2D 辅助 |
| H36M 图像 | /mnt/disk2/ch/H36M/ | 可视化/演示 |

**管线正确性校验（已通过）:**
1. **投影一致性**: camera_3d → K → 像素 vs pose_2d，误差 0.01px（108 action × 4 相机全过）
2. **世界→相机一致性**: Xc = R@Xw + T_mm，误差 0.0mm（发现并修复 T3WB T 单位米→mm）
3. **帧对齐**: T3WB frame_id ↔ H36M 图像帧名 100% 匹配
4. **骨架语义数值验证**: 骨骼长度符合人体比例（torso ~450mm 等）
5. **T3WB task2 数据 bug 修复**: pose_2d 大量为 0（351770 点），用 camera_3d 投影修复

**关键单位/坐标系约定:**
- 2D: 像素 → pelvis 相对 → torso 缩放（[-2,2] 范围）
- 3D: 相机系 mm → pelvis 相对 → 同一 torso 缩放（2D/3D 单位一致）
- 缺失关节（nose/eyes/ears，H36M 无对应）填 root 值，不参与监督（12 关节监督）
- 3DPW: 米→mm、SMPL 关节语义投票确定、图像竖屏系（1080×1920）

## 模型

- 因果膨胀 TCN: 输入 (B,81,34) → 5 层残差块 (dil=1,2,4,8,16) → (B,17,3)
- 单帧推理: 2.2ms (A100)
- Loss: 加权 MPJPE（12 监督关节平均）

## 训练

```bash
# 单 T3WB
python src/train.py --epochs 200 --datasets t3wb
# T3WB + 3DPW 混合（步长增强, 时间尺度鲁棒）
python src/train.py --epochs 300 --datasets t3wb,pw3d
```

- 训练样本: T3WB 46956×3(步长增强) + 3DPW 42192
- 每轮输出: val MPJPE + 5 个样本 GT/Pred 对比图（outputs/val_samples/）

## 评测

```bash
python src/evaluate.py  # S8 test (task1/task2), 归一化 + mm 双输出
```

## 推理

```bash
python src/infer_video.py --source video.mp4 --ckpt outputs/ckpt/best.pth
```

## 目录结构

```
configs/        配置
src/data/       T3WB/3DPW 解析、关节映射、相机、归一化、数据集
src/model/      因果 TCN
src/visualize.py 骨架可视化
src/check_data.py 训练前数据校验 (20 项, FAIL 拒绝训练)
src/train.py    训练循环
src/evaluate.py S8 评测
tools/          数据预处理脚本
```

## 已知问题 / 后续

- H36M 标准协议 S9/S11 评测需官方 3D 标注（T3WB 无 S9/S11）
- 跨 subject 泛化（S8 test ~288mm）vs val（48mm）差距主要来自 T3WB 稀疏采样的帧间隔差异，已用步长增强 + 3DPW 缓解
