# H36M 17 点训练-推理完整链路

> 从数据准备到端到端推理的可复现流程。适用于 H36M 17 点（VideoPose3D 协议）姿态估计。

---

## 1. 链路总览

```
┌─────────────────────────────────────────────────────────────────────┐
│  训练阶段                                                            │
├─────────────────────────────────────────────────────────────────────┤
│ H36M 标注 (h36m_annot.tar: train.h5 + images.txt)                   │
│   ├─ S (3D, 17关节, 相机系mm)                                      │
│   └─ part (2D, 17关节, 原图像素)                                    │
│        │                                                            │
│        ▼                                                            │
│ 缓存: tools/prepare_h36m_h17.py → h36m_train_h17.npz                │
│       (H36M 17 原序, 全 17 点监督, root=Hip 相对米制)               │
│        │                                                            │
│        ▼                                                            │
│ 3D lifting: src/train.py --datasets h36m_h17 --rf 27                 │
│       TCN: 27帧因果, 1024通道                                       │
│       结果: best.pth (val 37mm @ S9/S11)                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  推理阶段                                                            │
├─────────────────────────────────────────────────────────────────────┤
│ 视频 → RTMW 133点 (公开权重, wholebody)                             │
│   tools/infer_video_rtmw.py                                         │
│        │                                                            │
│        ▼                                                            │
│ 133点 → H36M 17点 (提取/估算)                                       │
│   - 12点直接对应 (肩肘腕髋膝踝)                                     │
│   - Hip=(l+r_hip)/2, Thorax=肩中点, Spine=髋肩1/3                   │
│   - Head=face最高点, Neck=肩到头35%                                 │
│        │                                                            │
│        ▼                                                            │
│ 归一化 → TCN (best.pth, 27帧因果) → 3D (root相对mm)                 │
│        │                                                            │
│        ▼                                                            │
│ 可视化: tools/gen_html_sync.py → inference.html                     │
│   左:视频帧图 | 中:2D骨架 | 右:3D可旋转 (同步播放)                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据准备

### 2.1 数据源

| 文件 | 内容 | 位置 |
|------|------|------|
| train.h5 | 312,188 帧: S(17×3 3D) + part(17×2 2D) | /mnt/disk2/ch/H36M/h36m/annot/ |
| valid.h5 | 109,867 帧 (S9/S11) | 同上 |
| train_images.txt | 图像名列表 | 同上 |
| H36M 图像 | S1/S5/S7 (60k帧, 10fps) | /mnt/disk2/ch/H36M/images/ |

### 2.2 缓存生成

```bash
python tools/prepare_h36m_h17.py
# 输出: data/cache/h36m_train_h17.npz (312k帧, S1/S5/S6/S7/S8)
#       data/cache/h36m_valid_h17.npz (110k帧, S9/S11)
# 格式: {subject:{action:{camera:{pose2d, cam3d, world3d, frame_id}}}}
# 监督: 全 17 点 (supervision_mask = all True)
```

### 2.3 关键处理

- **关节顺序**: H36M 17 (VideoPose3D): 0 Hip, 1 RHip, ..., 11 RShoulder, ..., 16 LWrist
- **3D 归一化**: root(Hip) 相对 → 米 (对齐 VideoPose3D)
- **2D 归一化**: 像素 → [-1,1] (保留位置信息)
- **root_idx=0**: H36M 关节 0 (Hip) 作为根 (dataset.py 支持 root_idx 参数)

---

## 3. 训练

```bash
python src/train.py --epochs 100 --batch 1024 --lr 1e-3 \
    --datasets h36m_h17 --rf 27
```

### 3.1 配置要点

| 项 | 值 |
|----|-----|
| 模型 | 因果膨胀 TCN, 27 帧, 5 层, 1024 通道 |
| 输入 | (B, 27, 34) = 27帧 × 17关节 × 2D |
| 输出 | (B, 17, 3) = 17关节 × 3D (root相对, 米) |
| Loss | 加权 MPJPE (per-sample mask) |
| 监督 | 全 17 点 |

### 3.2 结果

```
best.pth: val MPJPE 37mm @ S9/S11 (标准协议)
对比: 12点监督(COCO映射) 46mm → 17点全监督 37mm
```

### 3.3 关键经验（单位 bug 排查）

| 问题 | 现象 | 修复 |
|------|------|------|
| 3D 用 mm | 预测系统性偏小 (pred/gt=0.4) | 转米制 (±0.8), val 337→46mm |
| root 索引错误 | H36M 17 顺序下用 COCO 11/12 | root_idx=0 (Hip), val 44→37mm |

---

## 4. 推理（真实视频）

### 4.1 RTMW 2D 检测 (133 点)

```bash
# 权重: weights/rtmw-m_cocktail14.pth (130MB, 公开)
# 133点 = 17 body + 6 feet + 68 face + 42 hands
python tools/infer_video_rtmw.py video.mp4
```

### 4.2 133 → H36M 17 点提取

```python
# 直接对应 (12点)
H36M[1]=133[12] RHip    H36M[2]=133[14] RKnee   H36M[3]=133[16] RAnkle
H36M[4]=133[11] LHip    H36M[5]=133[13] LKnee   H36M[6]=133[15] LAnkle
H36M[11]=133[6] RShoulder  H36M[12]=133[8] RElbow  H36M[13]=133[10] RWrist
H36M[14]=133[5] LShoulder  H36M[15]=133[7] LElbow  H36M[16]=133[9] LWrist

# 计算点 (5点)
H36M[0]  = (133[11]+133[12])/2           # Hip
H36M[8]  = (133[5]+133[6])/2             # Thorax
H36M[7]  = Hip + (Thorax-Hip)/3          # Spine
H36M[10] = face68点中 y 最小             # Head
H36M[9]  = Thorax + (Head-Thorax)*0.35   # Neck
```

### 4.3 TCN 推理

- 27 帧因果窗口, 每帧输出 root 相对 3D
- 归一化与训练一致: 2D [-1,1] → TCN → 3D ×1000 (米→mm)

### 4.4 可视化

```bash
python tools/gen_html_sync.py
# 输出: inference.html (左视频帧图 | 中2D | 右3D可旋转, 统一 Play/slider 同步)
```

**HTML 特性**:
- 2D 用图像坐标 (y 向下, x 在底部)
- 3D 初始正面朝向屏幕, 可拖拽旋转/缩放
- **camera 保持**: 数据更新时保存并恢复视角 (不重置)
- plotly.min.js 本地引用 (避免 CDN 不可达)

---

## 5. 文件清单

```
configs/config.py              全局配置 (RECEPTIVE_FIELD=27)
src/data/dataset.py           数据集 (root_idx, per-sample mask)
src/model/tcn.py              因果 TCN
src/train.py                  训练 (h36m_h17 支持)
src/model/e2e.py              端到端模型 (可选)
tools/prepare_h36m_h17.py     H36M 17 点缓存
tools/infer_video_rtmw.py     视频推理 (RTMW→H36M17→TCN)
tools/gen_html_sync.py        同步可视化 HTML
outputs/ckpt/best.pth         H17 lifting 权重 (37mm)
weights/rtmw-m_cocktail14.pth RTMW 133点 2D 检测权重
```

---

## 6. 复现命令

```bash
# 1. 数据缓存
python tools/prepare_h36m_h17.py

# 2. 训练 lifting (约 2h, A100)
python src/train.py --epochs 100 --datasets h36m_h17 --rf 27

# 3. 视频推理
python tools/infer_video_rtmw.py videos/sample.mp4

# 4. 可视化
python tools/gen_html_sync.py  # 输出到 outputs/demo/rtmw_<tag>/
```
