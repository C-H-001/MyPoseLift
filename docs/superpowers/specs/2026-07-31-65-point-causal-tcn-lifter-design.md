# 65-Point Causal TCN Pose Lifter Design

## Goal

Replace the current 133-point experimental graph lifter with a strictly causal
temporal 2D-to-3D lifter over a reduced 65-point whole-body layout. The new
baseline must use historical frames as first-class signal, support local GPU
training, and keep runtime deployment simple.

## Problem Statement

The 133-point model underperforms on the official H3WB release. A linear
flattened 2D-to-3D diagnostic baseline reached about 91 mm MPJPE on fold0,
while the custom graph model remained far worse. The current dense face target
also contributes many points that are not needed for the intended application.

The next version should remove dense face supervision and use a literature-backed
strictly causal temporal architecture instead of a self-designed graph model.

## Keypoint Layout

The project will remap the original COCO-WholeBody/H3WB 133 points to a compact
65-point layout:

```text
0-16   body, copied from original 0-16
17-22  foot, copied from original 17-22
23-43  left hand, copied from original 91-111
44-64  right hand, copied from original 112-132
```

Dense face points `23-90` are removed. The retained head landmarks are the
COCO body points already present in `0-16`: nose `0`, left eye `1`, and right
eye `2`.

The pelvis/root remains the midpoint of left hip `11` and right hip `12`.
Targets remain root-relative camera-coordinate 3D poses.

## Model

The main model will be a strictly causal temporal convolutional network over
flattened skeleton vectors, not a single-frame lifter.

Input shape:

```text
(batch, time, 65, 3)
```

The model flattens joints and channels per frame to:

```text
(batch, time, 195)
```

Then it applies residual causal temporal convolution blocks with left padding
only. Each output may depend only on frames `<= t`. The final prediction is the
current frame:

```text
(batch, 65, 3)
```

The default long-history config uses `window=81`. A faster comparison config
uses `window=27`.

## Literature Rationale

The architecture follows the VideoPose3D family of temporal convolutional 2D
skeleton lifting models rather than a custom graph-only design. Temporal
convolutions are widely used for Human3.6M 2D-to-3D lifting because they provide
fixed latency, efficient GPU/CPU inference, and an explicit causal mode.

Single-frame Martinez-style residual MLPs remain useful as diagnostics, but are
not the primary architecture because this project requires historical motion
information.

Transformer, diffusion, and multi-hypothesis architectures are out of scope for
this iteration because they are heavier and harder to guarantee as strictly
online baselines.

## Data Flow

1. Download official H3WB `h3wb_train.npz`.
2. Prepare caches with remapped 65-point arrays:
   - `inputs_2d`: `(N, 65, 3)`
   - `targets_3d`: `(N, 65, 3)`
   - `target_masks`: `(N, 65)`
   - `sequence_ids`, `frame_ids`, `metas`
3. `H3WBDataset(window=81)` returns strictly causal history windows within one
   sequence, left-padding only at sequence starts.
4. Training uses `data.train_cache` and validates on `data.val_cache`.
5. Evaluation reports whole/body/foot/left-hand/right-hand/head3 metrics.
6. Visualization plots 65-point GT and prediction.

## Metrics

Report:

```text
MPJPE_whole
MPJPE_body
MPJPE_feet
MPJPE_head3
MPJPE_left_hand
MPJPE_right_hand
MPJPE_hands_wrist_aligned
```

Dense face metrics are removed.

## Constraints

- Strictly online: no future frame may be used during training or inference.
- No diffusion, large transformer, or non-causal centered windows.
- Do not depend on MMPose/MMEngine.
- Support GPU and CPU with `device: auto`.
- Default training epochs are 10 for fast local iteration.
- Large datasets, caches, checkpoints, and reports remain untracked.

## Success Criteria

- Official H3WB release NPZ can be prepared into 65-point train/validation
  caches.
- `window=81` and `window=27` configs train and evaluate.
- Tests prove remapping, causal windows, model output shape, no future leakage,
  metrics, and streaming inference behavior.
- A 10-epoch local GPU training run produces a checkpoint, MPJPE report, and
  GT-vs-pred 3D plot.
- The result is compared against the prior 133-point graph model and the linear
  diagnostic baseline.
