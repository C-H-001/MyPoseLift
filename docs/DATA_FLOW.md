# 数据流与处理过程全记录

> 本项目完整的数据流、处理流程、以及所有遇到的问题与解决方法。

---

## 1. 整体数据流架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据准备阶段                                    │
├─────────────────────────────────────────────────────────────────────┤
│ H36M 图像 (h36m.zip)          T3WB 重标注         3DPW              │
│ S1/S5/S7 (10fps, 60k帧)       h3wb_train.npz      sequenceFiles.pkl │
│      │                             │                   │            │
│      ▼                             ▼                   ▼            │
│ 解压 (tools/unpack_h36m.py)  解析 (src/data/t3wb.py)  解析           │
│      │                             │                   │            │
│      ▼                             ▼                   ▼            │
│  H36M 官方标注 (h36m_annot.tar)  →  COCO 17 映射 (joint_mapping)     │
│  train.h5 (312k帧, 17关节)        ← 投影校验 (0.01px)                │
│  valid.h5 (110k帧, S9/S11)                                         │
│      │                                                             │
│      ▼                                                             │
│  缓存 (tools/prepare_h36m_official.py)                              │
│  h36m_train.npz / h36m_valid.npz                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        训练阶段                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 3D Lifting (src/train.py + src/data/dataset.py)                     │
│   2D: H36M part 17关节 → [-1,1] (保留位置信息)                       │
│   3D: 相机系mm → root相对 → 米 (对齐 VideoPose3D)                    │
│   网络: 因果膨胀TCN, 27帧, 1024通道                                  │
│   Loss: 加权 MPJPE (per-sample mask)                                │
│      │                                                             │
│  RTMPose 2D检测器 (configs/rtmpose_h36m.py)                         │
│   H36M 17点 2D标注 → COCO json → RTMPose-L 训练                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        推理/可视化阶段                                │
├─────────────────────────────────────────────────────────────────────┤
│ RTMPose (H36M 17点 2D) → 归一化 → TCN → H36M 17点 3D               │
│      │                                                             │
│      ├─ HTML 可视化 (tools/gen_html_3d.py → demo/gt_pred_3d.html)   │
│      │   GT/Pred 3D 同图, Plotly 可旋转缩放                          │
│      └─ 视频可视化 (tools/gen_compare_video.py)                     │
│          左 2D | 中 GT | 右 Pred 同步播放                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据预处理流程

### 2.1 关节定义链

```
T3WB body(17) ──┐
3DPW SMPL(24) ──┼──→ COCO 17 ──→ 12 点监督 (H36M 无 eyes/ears/nose)
H36M h5(17) ────┘
H36M h5(17) ─────────→ H36M 17 原序 (tools/prepare_h36m_h17.py) → 全 17 点监督
```

### 2.2 归一化协议（对齐 VideoPose3D）

```python
# 2D: 像素 → [-1,1] (保留位置/距离信息)
normed2d = pose2d / 1000.0 * 2 - 1
# 3D: 相机系 mm → root相对 → 米
centered3d = (cam3d - root) / 1000.0
```

### 2.3 数据缓存格式（统一）

```
{subject: {action: {camera_id: {
    pose2d_coco17: (N,17,2),   # 像素 (或 H36M 原序)
    cam3d_coco17:  (N,17,3),   # 相机系 mm
    world3d_coco17:(N,17,3),
    frame_id: (N,)
}}}
```

---

## 3. 遇到的问题与解决方法（完整清单）

### 3.1 数据单位问题

| # | 问题 | 现象 | 根因 | 解决 |
|---|------|------|------|------|
| 1 | T3WB 平移 T 单位 | world→camera 误差 1650mm | T 是米, 3D 是 mm | `T *= 1000`, 回归测试 0.0mm |
| 2 | 3DPW 单位 | 投影爆炸(百万级) | jointPositions 米, 平移列也米, 但坐标乘了 1000 | 保持米计算, 最后统一转 mm |
| 3 | **3D 目标 mm/米** | **预测系统性偏小 (pred/gt=0.4)** | 3D 目标 ±800mm, BN gamma 学到 ~800 太慢, 输出向均值收缩 | **3D 转米制 (±0.8), val 从 337mm→46mm** |

### 3.2 标注/数据质量问题

| # | 问题 | 现象 | 解决 |
|---|------|------|------|
| 4 | T3WB task2 2D 缺失 | 351770 个全零点, 评测 1755mm | camera_3d 投影补全, 评测恢复 288mm |
| 5 | 3DPW 关节顺序非标准 | 骨骼长度异常(上臂 1163mm) | 跨帧投票+骨骼验证确定 SMPL 映射 |
| 6 | 3DPW 图像旋转 | 竖屏标定(1080×1920) vs 横屏文件 | 训练只需坐标对, 不受影响 |
| 7 | H36M 帧号格式 | images.txt 6位 vs 文件 4位 | 按序列内顺序匹配 |

### 3.3 归一化/坐标系问题

| # | 问题 | 现象 | 解决 |
|---|------|------|------|
| 8 | 2D/3D 尺度混淆 | 3D mm 除以 2D 像素尺度 | 2D 独立 [-1,1], 3D 独立米制 |
| 9 | 缺失关节填 0 | 归一化后大数值噪声 | 填 root 值 (减 root 后为 0) |
| 10 | 训练/推理 2D 不一致 | H36M 缺 5 点头部关节 | 换 H36M 17 点 2D 检测器 (全一致) |
| 11 | H36M 无法监督 5 点头部 | 12 点监督, 5 点无 GT 学不到 | 方案B (2D 重投影) 或 H36M 32 关节 |

### 3.4 训练/工程问题

| # | 问题 | 现象 | 解决 |
|---|------|------|------|
| 12 | conda run 无 shell | "No compatible shell found" | 用环境 python 绝对路径 |
| 13 | mmdet 版本冲突 | mmcv 2.2.0 与 mmdet 3.3.0 断言冲突 | patch mmdet/__init__.py |
| 14 | setuptools 移除 pkg_resources | import 失败 | 降级 setuptools<80 |
| 15 | numpy 2.x 不兼容 | xtcocotools C 扩展报错 | numpy 1.26 + opencv 4.x |
| 16 | 磁盘满 | checkpoint 363MB×226 吃满磁盘 | 每 10 epoch 保存 + 清理 |
| 17 | 并发写 best.pth | 文件损坏 | 隔离 ckpt 目录 |
| 18 | resume 覆盖 best | best=inf 初始化 | checkpoint 记录 val, resume 恢复 |
| 19 | DataLoader 崩溃 | "received 0 items of ancdata" | workers 8→4, persistent=False |
| 20 | RTMPose lr 过大 | acc 0.23→0.03 发散 | 预训练微调 lr 4e-3→1e-3 |

### 3.5 模型/训练问题

| # | 问题 | 现象 | 解决 |
|---|------|------|------|
| 21 | 窗口时间跨度 | 81帧@10fps=8.1s 时序失真 | rf 27 (2.7s), S8 泛化 288→259mm |
| 22 | 过拟合 (0.3x 数据) | train acc 升, val AP 降 | 早停在平台期 (epoch 10) |
| 23 | bbox 训练/推理不一致 | 检测误差 96px | 用与训练一致的紧致 bbox, 22.9px |
| 24 | 多数据集联合无增益 | 3DPW 混合 S9/S11 46→54mm | 3DPW 数据少+分布差异, 保留纯 H36M |

---

## 4. 关键实验结果

| 实验 | val (S9/S11) | 备注 |
|------|-------------|------|
| 12点监督 (COCO映射) | 46mm | 早期方案 |
| **17点监督 (H36M 原序)** | **44mm** | 当前最优 |
| RTMPose 2D 0.3x 数据 | AP 0.236 | 平台期 epoch 10 |
| RTMPose 2D 0.3x 紧框误差 | 22.9px | bbox 一致性修复后 |

---

## 5. 代码结构

```
configs/config.py            全局配置
configs/rtmpose_h36m.py      RTMPose H36M 17点训练配置
src/data/                   数据解析/映射/相机/归一化/数据集
src/model/tcn.py             因果膨胀 TCN
src/check_data.py            训练前数据校验
src/train.py                 3D lifting 训练 (多数据集+per-sample mask)
src/evaluate.py              S9/S11 评测
src/visualize.py             骨架可视化
tools/prepare_h36m_official.py  H36M 官方 h5 → 缓存
tools/prepare_h36m_h17.py      H36M 17点原序缓存 (全监督)
tools/prepare_h36m_2d.py       H36M 2D 检测数据 (COCO json)
tools/prepare_t3wb.py          T3WB → 缓存
tools/prepare_pw3d.py          3DPW → 缓存
tools/gen_html_3d.py          可旋转 3D 对比 HTML
tools/gen_compare_video.py    2D|GT|Pred 对比视频
demo/gt_pred_3d.html          可视化产物
```
