# 65-Point Causal TCN Pose Lifter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strictly causal temporal 65-point whole-body pose lifter using official H3WB 2D-to-3D release data and local GPU training.

**Architecture:** The project remaps original 133-point COCO-WholeBody/H3WB samples to a compact 65-point layout that keeps body, feet, and hands while dropping dense face points. The main model is a VideoPose3D-style residual causal TCN over flattened 2D skeleton histories, predicting only the current frame. Existing training, evaluation, and streaming interfaces remain but operate on the selected layout.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, PyYAML, pytest, matplotlib for local diagnostics.

## Global Constraints

- Strictly online: no future frame may be used during training or inference.
- Use 65-point layout: body original `0-16`, foot original `17-22`, left hand original `91-111`, right hand original `112-132`.
- Dense face points original `23-90` are removed.
- Retained head landmarks are body nose `0`, left eye `1`, and right eye `2`.
- Pelvis/root remains midpoint of left hip `11` and right hip `12`.
- Targets are current-frame root-relative camera-coordinate 3D with shape `(65, 3)`.
- Input histories have shape `(window, 65, 3)`.
- Default local training epochs are `10`.
- Default long-history model uses `window=81`; fast comparison config uses `window=27`.
- Do not depend on MMPose/MMEngine.
- Do not use diffusion, large transformer, multi-hypothesis sampling, or non-causal centered windows.
- Large datasets, caches, checkpoints, and reports are never committed.

---

## File Structure

- `mypose/data/keypoints65.py`: new 65-point layout, original-to-compact remap, parts, edges, and shape validation.
- `mypose/data/keypoints133.py`: keep existing compatibility constants for older tests and documentation only where needed.
- `mypose/data/h3wb.py`: remap official H3WB release NPZ and legacy JSON samples to 65-point caches.
- `mypose/data/coco_wholebody.py`: remap COCO-WholeBody annotations to 65-point caches.
- `mypose/data/transforms.py` and `mypose/data/validation.py`: use layout-aware constants and root indices.
- `mypose/utils/metrics.py`: support 65-point parts and head3 metric.
- `mypose/models/losses.py`: remove dense face losses and use 65-point edges/parts.
- `mypose/models/causal_tcn_lifter.py`: new flattened residual causal TCN lifter.
- `mypose/models/hrgcn_lifter.py`: keep old model for comparison but do not use as default.
- `mypose/engine/train.py`, `evaluate.py`, `infer_stream.py`: instantiate configured model type and report 65-point metrics.
- `configs/h3wb_tcn_t81.yaml`: default long-history causal TCN config.
- `configs/h3wb_tcn_t27.yaml`: fast comparison causal TCN config.
- `tools/plot_prediction.py`: create GT-vs-pred 3D plots for a checkpoint/sample.
- `README.md`, `docs/datasets.md`: update setup, layout, training, evaluation, and plotting commands.

---

### Task 1: Add 65-Point Layout and Dataset Remapping

**Files:**
- Create: `mypose/data/keypoints65.py`
- Modify: `mypose/data/h3wb.py`
- Modify: `mypose/data/coco_wholebody.py`
- Modify: `mypose/data/transforms.py`
- Modify: `mypose/data/validation.py`
- Test: `tests/test_keypoints65.py`
- Test: `tests/test_h3wb.py`
- Test: `tests/test_coco_wholebody.py`

**Interfaces:**
- Produces: `NUM_KEYPOINTS = 65`, `ORIGINAL_TO_65`, `remap_133_to_65(points)`, `remap_mask_133_to_65(mask)`, `get_part_indices(part)`, `COCO65_EDGES`.
- Consumes: existing H3WB release NPZ parser and COCO parser.

- [ ] **Step 1: Write layout tests**

Create `tests/test_keypoints65.py`:

```python
import numpy as np

from mypose.data.keypoints65 import (
    NUM_KEYPOINTS,
    ORIGINAL_TO_65,
    get_part_indices,
    remap_133_to_65,
)


def test_65_layout_keeps_body_feet_and_hands():
    assert NUM_KEYPOINTS == 65
    assert ORIGINAL_TO_65[:23] == list(range(23))
    assert ORIGINAL_TO_65[23:44] == list(range(91, 112))
    assert ORIGINAL_TO_65[44:65] == list(range(112, 133))


def test_65_parts_include_head3_and_no_dense_face():
    assert get_part_indices("body") == list(range(17))
    assert get_part_indices("foot") == list(range(17, 23))
    assert get_part_indices("head3") == [0, 1, 2]
    assert get_part_indices("left_hand") == list(range(23, 44))
    assert get_part_indices("right_hand") == list(range(44, 65))


def test_remap_133_to_65_preserves_expected_indices():
    points = np.arange(133 * 3, dtype=np.float32).reshape(133, 3)
    compact = remap_133_to_65(points)
    assert compact.shape == (65, 3)
    np.testing.assert_array_equal(compact[0], points[0])
    np.testing.assert_array_equal(compact[22], points[22])
    np.testing.assert_array_equal(compact[23], points[91])
    np.testing.assert_array_equal(compact[64], points[132])
```

- [ ] **Step 2: Verify layout tests fail**

Run: `python -m pytest tests/test_keypoints65.py -q`

Expected: FAIL because `mypose.data.keypoints65` does not exist.

- [ ] **Step 3: Implement `keypoints65.py`**

Implement:

```python
NUM_KEYPOINTS = 65
LEFT_HIP = 11
RIGHT_HIP = 12
ORIGINAL_TO_65 = list(range(23)) + list(range(91, 112)) + list(range(112, 133))
```

Parts:

```python
body: 0-16
foot: 17-22
head3: [0, 1, 2]
left_hand: 23-43
right_hand: 44-64
```

Edges:

```python
BODY_EDGES = original body edges unchanged
FOOT_EDGES = original foot edges unchanged
LEFT_HAND_EDGES = original left hand edges shifted from original 91-based to 23-based
RIGHT_HAND_EDGES = original right hand edges shifted from original 112-based to 44-based
COCO65_EDGES = BODY_EDGES + FOOT_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES
```

`remap_133_to_65(points)` accepts `(133, C)` or `(N, 133, C)` and returns compact points. `remap_mask_133_to_65(mask)` accepts `(133,)`, `(133, 1)`, or `(N, 133)` and returns compact masks.

- [ ] **Step 4: Update H3WB cache writing**

In `mypose/data/h3wb.py`, apply `remap_133_to_65` to:

```python
norm_2d
rel_3d
```

and apply `remap_mask_133_to_65` to masks before `validate_sample(sample)`.

For release NPZ samples, convert `pose_2d[index]` and `camera_3d[index]` from original 133 to compact 65 after normalization/rooting.

- [ ] **Step 5: Update COCO cache writing**

In `mypose/data/coco_wholebody.py`, remap parsed `(133, 3)` COCO-WholeBody arrays to `(65, 3)` before writing cache.

- [ ] **Step 6: Update validation and transforms**

Use `mypose.data.keypoints65.NUM_KEYPOINTS`, `LEFT_HIP`, and `RIGHT_HIP` in validation and root computation for the active project layout.

`validate_sample()` must require:

```text
history_2d: (T, 65, 3)
target_3d: (65, 3)
target_mask: (65,) or (65, 1)
```

- [ ] **Step 7: Update dataset tests**

Modify H3WB and COCO tests so generated caches assert:

```python
inputs_2d.shape[-2:] == (65, 3)
targets_3d.shape[-2:] == (65, 3)
target_masks.shape[-1] == 65
```

Add one H3WB test proving original dense face index `23` is removed and original hand index `91` maps to compact index `23`.

- [ ] **Step 8: Run tests**

Run:

```bash
python -m pytest tests/test_keypoints65.py tests/test_h3wb.py tests/test_coco_wholebody.py tests/test_transforms_validation.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add mypose/data tests
git commit -m "feat: add 65-point dataset remapping"
```

---

### Task 2: Update Losses and Metrics for 65 Points

**Files:**
- Modify: `mypose/utils/metrics.py`
- Modify: `mypose/models/losses.py`
- Modify: `mypose/engine/evaluate.py`
- Test: `tests/test_metrics_losses.py`

**Interfaces:**
- Consumes: `mypose.data.keypoints65.get_part_indices`, `COCO65_EDGES`.
- Produces: evaluation metrics `MPJPE_whole`, `MPJPE_body`, `MPJPE_feet`, `MPJPE_head3`, `MPJPE_left_hand`, `MPJPE_right_hand`, `MPJPE_hands_wrist_aligned`.

- [ ] **Step 1: Write failing metric tests**

Update `tests/test_metrics_losses.py` to assert:

```python
def test_evaluate_reports_65_point_metrics():
    assert "MPJPE_head3" in metrics
    assert "MPJPE_face" not in metrics
    assert "MPJPE_face_nose_aligned" not in metrics
```

Add:

```python
def test_whole_body_loss_uses_65_point_parts_without_dense_face():
    pred = torch.zeros(2, 65, 3)
    target = torch.ones(2, 65, 3)
    mask = torch.ones(2, 65, dtype=torch.bool)
    losses = WholeBodyLoss()(pred, target, mask)
    assert "head3_mpjpe" in losses
    assert "face_mpjpe" not in losses
    assert "face_local" not in losses
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/test_metrics_losses.py -q`

Expected: FAIL because current code reports dense face metrics.

- [ ] **Step 3: Update metrics**

Use `keypoints65.get_part_indices` in `part_mpjpe`. Remove dense face metric paths from `evaluate()`. Add head3 accumulation:

```python
accumulate("MPJPE_head3", _metric_sum_count(pred, target, mask, get_part_indices("head3")))
```

- [ ] **Step 4: Update loss**

Use part weights:

```python
body: 1.0
foot: 1.5
head3: 1.0
left_hand: 2.5
right_hand: 2.5
```

Local losses only:

```python
left_hand_local anchor 23
right_hand_local anchor 44
```

Remove dense face local loss.

- [ ] **Step 5: Update bone loss edges**

Use `COCO65_EDGES`.

- [ ] **Step 6: Run tests**

Run:

```bash
python -m pytest tests/test_metrics_losses.py tests/test_engine_smoke.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add mypose/utils/metrics.py mypose/models/losses.py mypose/engine/evaluate.py tests/test_metrics_losses.py tests/test_engine_smoke.py
git commit -m "feat: update metrics for 65-point layout"
```

---

### Task 3: Add Strictly Causal TCN Lifter

**Files:**
- Create: `mypose/models/causal_tcn_lifter.py`
- Modify: `mypose/engine/train.py`
- Modify: `mypose/engine/evaluate.py`
- Modify: `mypose/engine/infer_stream.py`
- Create: `configs/h3wb_tcn_t81.yaml`
- Create: `configs/h3wb_tcn_t27.yaml`
- Test: `tests/test_causal_tcn_lifter.py`
- Test: `tests/test_engine_smoke.py`
- Test: `tests/test_infer_stream.py`

**Interfaces:**
- Produces: `CausalTCNLifter(num_keypoints=65, in_channels=3, hidden_channels=512, num_blocks=4, kernel_size=3, dropout=0.1)`.
- Produces: `build_model_from_config(cfg) -> torch.nn.Module` supporting `model.type: causal_tcn`.
- Consumes: `(B, T, 65, 3)` histories and returns `(B, 65, 3)`.

- [ ] **Step 1: Write failing TCN tests**

Create `tests/test_causal_tcn_lifter.py`:

```python
import torch

from mypose.models.causal_tcn_lifter import CausalTCNLifter


def test_causal_tcn_outputs_current_65_point_pose():
    model = CausalTCNLifter(hidden_channels=64, num_blocks=2)
    y = model(torch.zeros(2, 81, 65, 3))
    assert y.shape == (2, 65, 3)


def test_causal_tcn_rejects_non_65_points():
    model = CausalTCNLifter(hidden_channels=32, num_blocks=1)
    with pytest.raises(ValueError, match="65"):
        model(torch.zeros(1, 27, 133, 3))


def test_causal_tcn_current_output_does_not_depend_on_future_frames():
    model = CausalTCNLifter(hidden_channels=32, num_blocks=2, kernel_size=3)
    model.eval()
    history = torch.randn(1, 81, 65, 3)
    changed = history.clone()
    changed[:, 80] = changed[:, 80] + 1000.0
    with torch.no_grad():
        before_future = model.forward_sequence(history)[:, :80]
        after_future = model.forward_sequence(changed)[:, :80]
    torch.testing.assert_close(before_future, after_future)


def test_causal_tcn_stream_matches_batch_last_frame():
    torch.manual_seed(7)
    model = CausalTCNLifter(hidden_channels=32, num_blocks=2)
    model.eval()
    history = torch.randn(1, 27, 65, 3)
    with torch.no_grad():
        batch = model(history)
        model.reset_stream()
        stepped = None
        for t in range(history.shape[1]):
            stepped = model.step(history[:, t])
    torch.testing.assert_close(batch, stepped)
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/test_causal_tcn_lifter.py -q`

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement causal residual blocks**

`CausalTCNLifter` implementation:

```python
flatten = nn.Linear(65 * 3, hidden_channels)
blocks = residual causal Conv1d blocks with left padding only
head = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, 65 * 3))
```

Each block:

```python
residual = x
x = F.pad(x, (dilation * (kernel_size - 1), 0))
x = conv(x)
x = GELU + dropout + pointwise conv
return norm((x + residual).transpose(1, 2)).transpose(1, 2)
```

`forward_sequence(history)` returns `(B, T, 65, 3)` predictions for every time step, but uses only causal history. `forward(history)` returns the last time step.

- [ ] **Step 4: Implement streaming**

`step(frame_2d)` appends a detached clone, keeps at most `receptive_field` frames, stacks `(B, T, 65, 3)`, and returns `forward(seq)`.

- [ ] **Step 5: Wire config model factory**

In train/evaluate/infer code, replace direct `HRGCNLifter(...)` construction with:

```python
def build_model_from_config(cfg):
    if cfg["model"]["type"] == "causal_tcn":
        return CausalTCNLifter(...)
    if cfg["model"]["type"] == "hrgcn":
        return HRGCNLifter(...)
    raise ValueError(...)
```

- [ ] **Step 6: Add configs**

`configs/h3wb_tcn_t81.yaml`:

```yaml
data:
  train_cache: data/processed/h3wb_65_train_fold0.npz
  val_cache: data/processed/h3wb_65_val_fold0.npz
  window: 81
model:
  type: causal_tcn
  hidden_channels: 512
  num_blocks: 5
  kernel_size: 3
  dropout: 0.1
loss:
  part_weights:
    body: 1.0
    foot: 1.5
    head3: 1.0
    left_hand: 2.5
    right_hand: 2.5
  local_weights:
    left_hand: 1.0
    right_hand: 1.0
train:
  device: auto
  batch_size: 256
  num_workers: 4
  lr: 0.0005
  weight_decay: 0.01
  epochs: 10
  seed: 0
  out_dir: checkpoints/h3wb_tcn_t81
```

`configs/h3wb_tcn_t27.yaml` is identical except `window: 27` and `out_dir: checkpoints/h3wb_tcn_t27`.

- [ ] **Step 7: Run tests**

Run:

```bash
python -m pytest tests/test_causal_tcn_lifter.py tests/test_engine_smoke.py tests/test_infer_stream.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add mypose/models/causal_tcn_lifter.py mypose/engine configs tests
git commit -m "feat: add causal TCN pose lifter"
```

---

### Task 4: Add Training, Evaluation, and Plotting Workflow

**Files:**
- Create: `tools/plot_prediction.py`
- Modify: `README.md`
- Modify: `docs/datasets.md`
- Test: `tests/test_plot_prediction.py`

**Interfaces:**
- Consumes: `build_model_from_config(cfg)`, `H3WBDataset`, `evaluate()`.
- Produces: `reports/h3wb_tcn_t81_fold0_gt_pred.png`.

- [ ] **Step 1: Write plotting CLI test**

Create `tests/test_plot_prediction.py`:

```python
def test_plot_prediction_help_runs():
    result = subprocess.run(
        [sys.executable, "tools/plot_prediction.py", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout
```

- [ ] **Step 2: Verify test fails**

Run: `python -m pytest tests/test_plot_prediction.py -q`

Expected: FAIL because script does not exist.

- [ ] **Step 3: Implement plotting CLI**

`tools/plot_prediction.py` arguments:

```text
--config PATH
--checkpoint PATH
--cache PATH optional
--index INT default 0
--out PATH default reports/gt_pred.png
```

It loads the model, predicts one sample, plots GT and pred with 65-point edges, and saves PNG.

- [ ] **Step 4: Update README**

Document:

```bash
python tools/download_datasets.py --dataset h3wb --root data/raw
python tools/prepare_h3wb.py --annotations data/raw/h3wb/annotations/h3wb_train.npz --train-out data/processed/h3wb_65_train_fold0.npz --val-out data/processed/h3wb_65_val_fold0.npz --num-folds 5 --val-fold 0
python -m mypose.engine.train --config configs/h3wb_tcn_t81.yaml
python -m mypose.engine.evaluate --config configs/h3wb_tcn_t81.yaml --checkpoint checkpoints/h3wb_tcn_t81/best.pt
python tools/plot_prediction.py --config configs/h3wb_tcn_t81.yaml --checkpoint checkpoints/h3wb_tcn_t81/best.pt --index 100 --out reports/h3wb_tcn_t81_fold0_gt_pred.png
```

Explain that 65-point layout removes dense face and keeps nose/eyes through body indices 0/1/2.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_plot_prediction.py tests/test_inspect_sample.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/plot_prediction.py README.md docs/datasets.md tests/test_plot_prediction.py
git commit -m "docs: add 65-point TCN workflow"
```

---

### Task 5: Verify on Official H3WB Release and Publish

**Files:**
- Modify: none required unless verification reveals a defect.

**Interfaces:**
- Consumes: official `data/raw/h3wb/annotations/h3wb_train.npz`.
- Produces: local untracked caches, checkpoints, reports.

- [ ] **Step 1: Prepare 65-point fold0 caches**

Run:

```bash
python tools/prepare_h3wb.py --annotations data/raw/h3wb/annotations/h3wb_train.npz --train-out data/processed/h3wb_65_train_fold0.npz --val-out data/processed/h3wb_65_val_fold0.npz --num-folds 5 --val-fold 0
```

Expected: writes both cache files.

- [ ] **Step 2: Inspect cache**

Run:

```bash
python tools/inspect_sample.py --cache data/processed/h3wb_65_train_fold0.npz --index 100 --window 81
```

Expected:

```text
history_2d: (81, 65, 3)
target_3d: (65, 3)
valid target points: 65
```

- [ ] **Step 3: Run full tests**

Run:

```bash
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 4: Train 10 epochs on local GPU**

Run:

```bash
python -m mypose.engine.train --config configs/h3wb_tcn_t81.yaml
```

Expected: writes `checkpoints/h3wb_tcn_t81/last.pt` and `best.pt`.

- [ ] **Step 5: Evaluate best checkpoint**

Run:

```bash
python -m mypose.engine.evaluate --config configs/h3wb_tcn_t81.yaml --checkpoint checkpoints/h3wb_tcn_t81/best.pt
```

Expected: prints MPJPE metrics.

- [ ] **Step 6: Plot GT vs prediction**

Run:

```bash
python tools/plot_prediction.py --config configs/h3wb_tcn_t81.yaml --checkpoint checkpoints/h3wb_tcn_t81/best.pt --index 100 --out reports/h3wb_tcn_t81_fold0_gt_pred.png
```

Expected: writes a non-empty PNG.

- [ ] **Step 7: Compare against diagnostics**

Record:

```text
previous 133-point graph epoch10 whole MPJPE: 186.8 mm
linear diagnostic whole MPJPE: about 91.4 mm
new 65-point TCN whole MPJPE: measured value
```

If the new TCN is still worse than the linear diagnostic, do not hide it. Report that the next architecture step should be stronger VideoPose3D-style multi-block/dilated TCN or GraphMLP, not more epochs.

- [ ] **Step 8: Confirm no data artifacts are tracked**

Run:

```bash
git status --short
git ls-files data checkpoints reports
```

Expected: no tracked dataset, checkpoint, or report files.

- [ ] **Step 9: Commit any final fixes**

If no code changes are needed, do not create an empty commit. If fixes were needed:

```bash
git add <changed files>
git commit -m "fix: stabilize 65-point TCN verification"
```

- [ ] **Step 10: Push**

Run:

```bash
git push
```

Expected: updates `origin/codex/133-pose-lifter-pr`.

---

## Self-Review

- Spec coverage: layout reduction, causal TCN, H3WB release cache, metrics, plotting, 10-epoch GPU verification, and artifact safety are covered.
- Placeholder scan: no TODO/TBD placeholders remain.
- Type consistency: all tasks use `(T, 65, 3)` histories, `(65, 3)` targets, and named config files `h3wb_tcn_t81.yaml` / `h3wb_tcn_t27.yaml`.
