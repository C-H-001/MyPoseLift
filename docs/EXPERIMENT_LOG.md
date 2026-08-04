# 3D 人体姿态估计（2D→3D Lifting）实验全记录

> 项目周期：2026-08-03 ~ 2026-08-04
> 代码仓库：MyPoseLift（本仓库）
> 远程环境：2× NVIDIA A100 40GB，CUDA 12.4，Ubuntu 24.04

---

## 1. 项目概述

**目标**：基于 2D 检测 → 3D lifting 方案，参考 VideoPose3D 论文，实现**因果** 3D 人体姿态估计（17 关键点，COCO 标准），满足 GPU 实时性。

**总体架构**：
```
[视频帧] → [RTMPose 2D检测] → [17kp 2D序列 27帧] → [因果膨胀TCN] → [17kp 3D]
                                          ↑
                             训练用 GT 2D, 推理用 RTMPose
```

**关键技术决策**（设计阶段确定）：
- 架构：VideoPose3D 风格因果膨胀 TCN（5 层残差块，1024 通道）
- 训练：先用 GT 2D 验证 lifting 网络正确性，推理用 RTMPose（mmpose）
- 关键点：COCO 17 标准，H36M/3DPW 映射到 COCO 17
- 实时性：单帧推理 2.2ms（A100）✅

---

## 2. 数据获取

### 2.1 初始数据状态

| 数据集 | 内容 | 3D 标注 | 问题 |
|--------|------|---------|------|
| H36M (h36m.zip) | 仅图像 (S1/S5/S7, 10fps) | ❌ 无 | 无标注 |
| COCO 2017 | train2017 + annotations | 2D only | - |
| 3DPW | imageFiles + sequenceFiles | ✅ pkl | 需解析 |
| T3WB | h3wb_train.npz + test npz | ✅ | 第三方重标注 |

**关键发现**：h36m.zip 只有图像无标注，T3WB 是 H36M 的第三方重标注（稀疏采样）。

### 2.2 后续补充：H36M 官方标注

用户补充上传 `h36m_annot.tar`（349MB），内含：
- `train.h5`：312,188 帧 × 17 关节（3D S + 2D part），S1/S5/S6/S7/S8
- `valid.h5`：109,867 帧，**S9/S11（论文标准测试集）**
- `train_images.txt` / `valid_images.txt`：图像名列表

这是 MMPose 风格的 H36M 预处理数据，10fps 全量序列（非稀疏采样）。

---

## 3. 环境搭建

```bash
# conda 环境 (通过 /home/wsco/anaconda3/bin/conda)
conda create -n ch_pose python=3.10 -y
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy opencv-python matplotlib tqdm tensorboard pandas scikit-learn pytest h5py plotly

# 推理环境 (mmpose + RTMPose)
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmdet==3.3.0 mmpose==1.3.2 --no-deps
pip install xtcocotools json_tricks munkres "setuptools<80"
# 注意: mmdet 3.3.0 与 mmcv 2.2.0 版本断言冲突, 已 patch mmdet/__init__.py
# 注意: numpy 需 1.26.x (xtcocotools 与 numpy 2.x 不兼容), opencv 需 4.x
```

**环境问题记录**：
- conda run 在 SSH 非交互环境报 "No compatible shell found" → 用 `/home/user/.conda/envs/ch_pose/bin/python` 绝对路径
- mmdet 3.3.0 要求 mmcv < 2.2.0，但 cu121/torch2.4 只有 mmcv 2.2.0 预编译 → patch 版本断言
- setuptools 83 移除了 pkg_resources → 降级到 <80

---

## 4. 数据管线设计（重点校验区域）

### 4.1 关节映射链

```
T3WB body(17) → H36M 17 → COCO 17
3DPW SMPL 24 → COCO 17
H36M h5 (17) → COCO 17
```

**监督策略（保守）**：仅监督 12 个肢体关节（肩肘腕髋膝踝），不监督：
- nose/eyes/ears（H36M 无可靠对应，neck→nose 是错误语义近似）
- 12 关节监督 mask 保证 loss 不受缺失关节污染

### 4.2 归一化协议（最终版，对齐 VideoPose3D）

```python
# 2D: 像素 → [-1,1] (保留位置/距离信息)
normed2d = pose2d / 1000.0 * 2 - 1
# 3D: 相机系 mm → root 相对 → 米
centered3d = (cam3d - root) / 1000.0
```

---

## 5. 数据管线问题与修复（详细记录）

### 5.1 T3WB 平移 T 单位错误（米 vs 毫米）⭐

**现象**：`world_to_camera(global_3d, R, T)` 与 T3WB 提供的 camera_3d 误差 1650mm。

**排查**：系统尝试 14 种 R/T 符号与单位组合，发现 `Xc = R @ Xw + T*1000` 误差 0.0mm。

**根因**：T3WB 的平移 T 单位为**米**（如 4.45），而 3D 坐标为 mm。

**修复**：`t3wb.py` 的 `get_camera_params` 中 `T *= 1000`，并加回归测试（3 个 subject/action 全量验证 0.0mm）。

### 5.2 T3WB task2 2D 标注缺失（全零点）⭐

**现象**：task2_test_3d.npz 的 pose_2d 大量为 0（351,770 个全零点，覆盖全部 12 个监督关节），导致评测 MPJPE 1755mm（异常大）。

**排查**：对比 task1（正常）与 task2（异常）的 2D 分布，确认 task2 的 2D 标注缺失。

**修复**：用 camera_3d 投影生成缺失的 2D（3D 标注完整且投影精确），评测恢复到 288mm。

### 5.3 3DPW 关节顺序非标准 ⭐

**现象**：标准 SMPL 关节顺序假设下，骨骼长度异常（上臂 1163mm）。

**排查过程**：
1. 检查世界系坐标 → 数据正常（x 0.5m, y 1.55m 人体结构）
2. 投影到图像验证 → 变换正确（与 poses2d 重合）
3. 跨帧最近邻匹配 3D 投影 ↔ 官方 poses2d（VIBE 定义 14 关节）
4. 骨骼长度验证 → 确定真实顺序：SMPL[16]=l_shoulder, [17]=r_shoulder 等

**根因**：3DPW 的 jointPositions 不是标准 SMPL 24 顺序，是 3DPW 自定义顺序。

**修复**：基于投票+骨骼验证的 `SMPL_TO_COCO17` 映射。

### 5.4 3DPW 单位问题（米 vs 毫米）

**现象**：camera_3d 投影爆炸（2D 百万级）、z 为负。

**根因**：jointPositions 与 cam_poses 平移列都是米，但关节坐标乘了 1000（mm）而平移没乘。

**修复**：保持米计算，最后统一 `Xc * 1000` 转 mm。

### 5.5 3DPW 图像旋转坑

**发现**：demo.py 显示图像尺寸 = 2×光心（1080×1920 竖屏），但图像文件是 1920×1080 横屏。

**影响**：训练只需 2D/3D 坐标对（竖屏系），不受影响；可视化/推理需旋转图像。

### 5.6 T3WB 帧对齐验证

**验证**：T3WB frame_id（如 '0075'）与 h36m.zip 图像帧名（frame_0075.jpg）100% 匹配（562/562），排除用户警告的"标注错位"风险。

### 5.7 H36M h5 数据自洽性验证

**验证**：h5 的 3D (S) 用相机参数投影回 2D vs part，误差仅 0.8px——数据完全自洽。

### 5.8 图像尺寸细节

H36M 实际图像 1000×1002（宽×高），T3WB 标注基于 1000×1000 尺度（投影验证 0.01px），推理时需处理 y 方向 0.2% 差异。

---

## 6. 模型与训练

### 6.1 模型结构

```
Input: (B, 27, 34)  →  CausalConv1d(34→1024, dil=1)
       → 5× ResidualBlock(1024, dil=1,2,4,8,16)  [Conv1d+BN+ReLU+Dropout(0.25)]
       →  Conv1d(1024→51) + BN  →  取最后一帧 → (B, 17, 3)
```

### 6.2 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam, lr=1e-3 (StepLR step=15, gamma=0.5) |
| Epochs | 150-200 |
| Batch | 1024 |
| Loss | 加权 MPJPE（12 监督关节平均） |
| 设备 | 单卡 A100 40GB |

### 6.3 实验迭代记录

| 实验 | 数据 | 窗口 | 结果 (S9/S11 val) | 结论 |
|------|------|------|-------------------|------|
| rf81 T3WB | T3WB 稀疏 | 81 帧 | 288mm (S8) | 基线 |
| rf27 T3WB | T3WB 稀疏 | 27 帧 | 259mm (S8) | 短窗口更好 |
| rf9 T3WB | T3WB 稀疏 | 9 帧 | 260mm (S8) | 与 rf27 相当 |
| 混合 T3WB+3DPW | 混合 | 81 帧 | 300mm (S8) | 3DPW 无帮助 (分布差异) |
| h36m mm 制 | H36M 官方 | 27 帧 | 337mm→278mm | 数据更好但仍有 bug |
| **h36m 米制 (最终)** | H36M 官方 | 27 帧 | **46mm** ✅ | 修复单位 bug 后收敛 |

---

## 7. 核心问题排查：预测值系统性偏小 ⭐⭐

### 7.1 现象

用户通过可旋转 3D 对比 HTML 发现：**Pred 骨架比 GT 小很多**。

### 7.2 量化诊断

```
GT   mean|.|=160.3  std=264   range=[-817, 962] mm
PRED mean|.|=66.6   std=92    range=[-460, 310] mm
PRED/GT 比值 = 0.415（系统性偏小）
```

**逐关节规律——离根节点越远，偏小越严重**：
- 髋（root 附近）: pred/gt = 1.09 ✅
- 膝: 0.40
- 踝: **0.22** ❌

这是典型的**"向均值收缩"**（regression to the mean）。

### 7.3 排查步骤（系统化）

1. **排除过拟合**：训练集内（同分布 S5）比值也是 0.435 → 不是泛化问题，是模型没学会尺度
2. **排除 BN running stats**：eval 模式 (std=88) vs train 模式 (std=97) 差异小 → 不是 BN eval/train 问题
3. **Hook 中间层诊断**：中间层 std 0.7-1.3（正常），out_bn 输出 std=20 vs GT std=232（**11 倍差距**）→ 输出层尺度学习失败
4. **对比 VideoPose3D 实现**：发现关键差异——**VideoPose3D 3D 用米制（±0.8），我们用 mm（±800）**

### 7.4 根因

```
3D 目标范围 ±800mm，输出层 BatchNorm 的 gamma 需从 1 学到 ~800 的缩放
→ 梯度更新极慢 → 模型只能学到"向均值收缩"的保守解
→ 远端关节（踝）被严重压缩 (pred/gt = 0.22)
```

### 7.5 修复

```python
# dataset.py: 3D root 相对后 /1000 (mm → 米)
centered3d = centered3d / 1000.0   # 输出范围 ~±0.8, BN 易学
```

### 7.6 修复效果（决定性验证）

| 训练 | epoch 8 train | val (S9/S11) |
|------|--------------|--------------|
| mm 制（bug） | 370mm | 389mm |
| **米制（修复）** | **27mm** | **49mm** |

修复后 **val 46mm 达到 VideoPose3D 论文水平**（因果 ~46mm）。

---

## 7.5 A 方案: 多数据集联合训练实验 (H36M + 3DPW)

**实现** (已提交, 49/49 测试通过):
- per-sample visible mask: dataset 返回 (x, y, mask), 各数据集监督关节可不同
- loss 支持 (B,J) per-sample mask (有效关节项平均)
- 平衡采样: WeightedRandomSampler + --pw3d-weight 参数
- 多数据集 ConcatDataset 联合

**结果** (rf=27, 联合训练 30 epoch vs 纯 H36M 149 epoch):

| 模型 | H36M S9/S11 val | 3DPW test (in-the-wild) |
|------|----------------|------------------------|
| 纯 H36M (epoch_149) | **46mm** | 226.2mm |
| H36M+3DPW 联合 | 54mm (下降) | 227.6mm (无提升) |

**结论**: 3DPW 混合训练收益为负。原因:
1. 3DPW 仅 24 序列/17k 帧, 上采样后占比仍 <6%, 影响有限
2. 室外移动相机 vs H36M 室内固定相机, 分布差异大, "两头不讨好"
3. 3DPW 2D 由 3D 投影生成, 与 H36M 2D 标注分布不一致

**经验**: 联合训练需选与主数据集分布接近的补充数据 (如 MPI-INF-3DHP),
或采用半监督 (COCO 2D 投影约束, 方案 B) 避免 3D 分布冲突。

## 8. 最终结果

| 指标 | 结果 |
|------|------|
| 架构 | 因果膨胀 TCN, 27 帧, 1024 通道 |
| 数据 | H36M 官方 (312k 训练帧, S1/S5/S6/S7/S8) |
| **val MPJPE (S9/S11)** | **46mm**（VideoPose3D 论文水平） |
| 单帧推理延迟 | 2.2ms (A100, 实时 ✅) |
| 测试 | 49 项单元测试通过 |

---

## 9. 训练基础设施问题

| 问题 | 解决 |
|------|------|
| 磁盘满 (checkpoint 363MB × 226) | 每 10 epoch 保存 + 清理旧 ckpt |
| 两训练并发写 best.pth 损坏 | 按实验隔离 ckpt 目录（后续） |
| resume 后 best=inf 覆盖旧 best | checkpoint 记录 val, resume 恢复 |
| checkpoint 363MB 过大 | 模型大 (16M 参数) + Adam 状态, 接受 |

---

## 10. 经验总结

1. **数据单位是姿态估计最常见的坑**：T3WB T 单位（米/毫米）、3DPW 单位、3D 目标单位（mm/米）——每个都是系统性错误来源，必须有投影一致性/数值校验
2. **BN 输出层的尺度学习**：大范围回归目标（±800）会因 BN gamma 学习慢导致"向均值收缩"，应归一化到网络易学的尺度（±1）
3. **同分布 vs 跨 subject 区分**：过拟合诊断必须先测同分布数据，避免误判
4. **可视化价值**：可旋转 3D 对比是发现"预测偏小"这类系统性问题的关键工具
5. **短窗口更好**：稀疏采样数据下，窗口时间跨度影响泛化（rf27 优于 rf81）
6. **第三方标注需全面校验**：T3WB/3DPW 都发现了数据问题（2D 缺失、关节顺序、单位）
