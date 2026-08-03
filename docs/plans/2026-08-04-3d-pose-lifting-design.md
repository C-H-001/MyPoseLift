# 3D 人体姿态估计（2D→3D Lifting）设计文档

日期: 2026-08-04
状态: 已确认（brainstorming 完成）

## 1. 目标

- 基于 2D 检测 → 3D lifting 方案，参考 VideoPose3D，实现**因果** 3D 姿态估计
- GPU 条件下满足实时性要求
- 17 关键点（COCO 标准），H36M/3DPW 映射到 COCO 17

## 2. 总体架构

```
[视频帧] -> [RTMPose 2D检测器(推理)] -> [17kp 2D序列 81帧] -> [TCN Lifting网络] -> [17kp 3D姿态]
                                      ^                          ^
                           离线用GT 2D训练              因果卷积，仅用历史帧
```

- **阶段一 (2D 检测)**: MMPose RTMPose 预训练权重，离线提取 2D keypoint；推理时实时运行
- **阶段二 (3D Lifting)**: VideoPose3D TCN 架构（多层膨胀 1D 卷积 + 残差连接 + BN + ReLU）
  - 输入: (B, 81, 34) = 81 帧 x 17 关节 x 2D
  - 输出: (B, 1, 51) = 当前帧 17 关节 x 3D
  - 因果 padding: 只在时间左侧 pad，padding=(dilation*(kernel-1), 0)

## 3. 关键点定义

- 标准: COCO 17 keypoints
- H36M 32 关节 -> COCO 17: 固定映射表（写死，需校验）
- 3DPW SMPL 关节 -> COCO 17: SMPL joint regressor 或固定映射
- 所有数据集统一到 COCO 17 顺序

## 4. 数据集

| 数据集 | 状态 | 用途 |
|--------|------|------|
| Human3.6M | /mnt/disk2/ch/H36M/h36m.zip (3.6G) | 主训练集 (3D GT + 2D GT) |
| COCO 2017 | /mnt/disk2/ch/COCO/ | 2D 辅助 |
| 3DPW | /mnt/disk2/ch/3DPW/ (3 zip) | in-the-wild 3D 补充 |

- H36M 划分: S1,S5,S6,S7,S8 训练; S9,S11 测试（Protocol #1）
- 3DPW: 混合训练，提高泛化

## 5. 数据管线（重点校验区域）

### 5.1 坐标系转换（世界 -> 相机）
- H36M 3D GT 是世界坐标 (mm)，训练用**相机坐标系**，消除全局位移
- X_cam = R @ (X_world - t)，注意 t 的符号与单位（易错点）
- 每个 subject 每个相机有独立外参 [R|t]

### 5.2 归一化策略
- 2D: 像素坐标 -> 归一化(除以图像宽高) -> 减去根节点(pelvis)
- 3D: mm -> 减去根节点(pelvis)
- 2D/3D 根节点定义必须一致

### 5.3 数据单位校验点
- 2D 坐标: 像素 -> 归一化 -> 根节点相对
- 3D 坐标: mm -> 根节点相对
- 3D 投影回 2D 校验相机参数正确性

## 6. 模型架构

```
Input: (B, 34, 81) -> 1D Conv Block (dil=1, 34->1024)
       -> Residual Blocks (dil=2,4,8,16,32, 1024->1024)
       -> Output Conv (dil=1, 1024->51) -> (B, 51, 1) -> (B, 1, 51)
```

- 每个 Residual Block: 1D Conv + BN + ReLU + Dropout(0.25)
- 总感受野 ~127 > 81，确保覆盖全部输入
- 参数量 ~16M

## 7. 训练配置

| 参数 | 设置 |
|------|------|
| 优化器 | Adam, lr=1e-3 |
| Epochs | 50 |
| Batch Size | 1024 clips |
| 设备 | 单卡 A100 40GB |
| 混合精度 | AMP |

- Loss: MPJPE（3D 相机坐标系，pelvis 相对，L2），参考 VideoPose3D 原论文/MMPose
- 可选: 中间帧监督（multi-frame supervision）

## 8. 训练前数据检查 (check_data.py)

必须通过后才能启动训练:
1. 格式统一性: 所有数据集 2D (N,17,2)、3D (N,17,3)，COCO 17 顺序
2. 数值范围: 2D 归一化后 [-2,2]，3D 合理毫米范围
3. 映射关系抽查: H36M 32->17 映射，左右肩对称性
4. 无 NaN/Inf/全零帧
5. 可视化 (每个数据集 2-3 样本):
   - 2D 检查图: 原图 + keypoint + 骨架连线
   - 3D 检查图: 正视图/侧视图 3D 骨架
   - 投影一致性图: 3D 投影回 2D 与 2D GT 叠加

## 9. 每轮验证输出

每个 val epoch:
1. 计算 val MPJPE（全测试集）
2. 随机采 5 样本: 2D 输入 + 3D 推理结果 + GT 对比
3. 保存可视化（3D 骨架旋转视图 + 2D 投影叠加原图）
4. 写 checkpoint (best.pth + epoch_N.pth) + tensorboard/log

## 10. 实现顺序（概要）

1. 环境搭建: conda env ch_pose + PyTorch + CUDA
2. 数据解压/整理: H36M, COCO, 3DPW
3. 数据预处理脚本: 坐标系转换、归一化、关节映射、缓存 npz
4. check_data.py 可视化校验
5. 模型实现 (TCN)
6. 训练脚本 + 每轮验证可视化
7. RTMPose 推理管线
8. 完整评测 (H36M Protocol #1, 3DPW test)

---

## 11. 补充：T3WB 标注（H36M 3D 来源）

**位置**: /mnt/disk2/ch/T3WB/
**文件**: h3wb_train.npz (训练), task1_test_3d.npz / task2_test_3d.npz (测试), T3WB_v1.json (2D 全身), json.zip

### 11.1 数据结构
- metadata: {subject: {camera_id: {K,R,T,Distortion}}}, subjects = S1,S5,S6,S7,S8
- train_data: {subject: {action: {global_3d(N,133,3), frame_id(N,), camera_3d(N,133,3), pose_2d(N,133,2), sample_id(N,)}}}
- 133 点 = 17 body + 68 face + 42 hands + 6 feet
- metadata 含 h3wb_vs_h36m 映射表 (T3WB body -> H36M, 13对) 和 body/face/hand/foot 索引定义
- K 为归一化相机内参 (fx=1.145, cx=0.515), 需与像素坐标换算

### 11.2 已验证事实
- frame_id (str, 如 0075) 与 h36m.zip 图像帧名 frame_XXXX.jpg 直接对应
- S5/Directions 1: h36m.zip 2248 帧 (0075-4925, 步长5=10fps), T3WB 562 帧 (75-4925, 稀疏采样)
- T3WB 时序为 10fps 基础 + 稀疏采样, 帧间隔 5-50 不等
- 相机坐标: T3WB 已提供 camera_3d (相机系), global_3d (世界系, mm)

### 11.3 错位风险与校验（实现时必须执行）
1. frame_id 与图像文件对齐断言（对每个 subject/action 全量校验）
2. h3wb_vs_h36m 只含 13 对, T3WB body[1,2,3,4] 语义需确认（推测为 pelvis/spine/thorax/head）
3. 关节映射后必须做 3D 骨架可视化抽查（左右对称、拓扑合理）
4. 相机投影一致性检查: camera_3d -> K -> pose_2d 与存储的 pose_2d 对比
5. subject 划分: T3WB 训练 S1,S5,S6,S7 (S6 无图像但有完整标注), 测试 S8
6. H36M 标准协议 S9/S11 评测需要官方 3D 标注（T3WB 不含, 需另寻）

### 11.4 对训练的影响
- lifting 网络训练只需 2D/3D 坐标序列, 不需要图像 -> S6 也可用
- h36m.zip 图像主要用途: 可视化、2D 检测器验证、推理演示
- 时序窗口构建需处理 T3WB 稀疏采样（帧间隔不均）
