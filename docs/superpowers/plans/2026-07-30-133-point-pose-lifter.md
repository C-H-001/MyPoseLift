# 133-Point Pose Lifter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight, strictly causal 2D-to-3D COCO-WholeBody 133-point pose lifter with H3WB 3D supervision, official COCO-WholeBody download support, part-aware losses, and CPU/GPU inference.

**Architecture:** The system uses pure PyTorch modules with explicit dataset parsers and typed sample dictionaries. H3WB provides 3D supervision, COCO-WholeBody provides official 133-point layout and 2D distribution, and the model is an HR-GCN-inspired framewise graph lifter with an optional strictly causal temporal adapter. Tests establish keypoint ordering, root-relative camera transforms, hand/face loss behavior, causal masking, and CPU streaming inference.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, PyYAML, requests, tqdm, pytest.

## Global Constraints

- Do not port MMPose/MMEngine as a framework dependency.
- Do not directly expand VideoPose3D to 133 points as the main method.
- Do not use diffusion, multi-hypothesis sampling, large transformer backbones, or non-causal centered windows.
- Do not predict absolute global translation in the first version.
- Use COCO-WholeBody 133 keypoint layout: body `0-16`, foot `17-22`, face `23-90`, left hand `91-111`, right hand `112-132`.
- Use computed pelvis/root as the midpoint of COCO left hip index `11` and right hip index `12`.
- Prediction target is current-frame root-relative camera-coordinate 3D pose with shape `(133, 3)`.
- Online inference must use only frames `<= t`.
- Large dataset files are never committed to git.

---

## File Structure

- `pyproject.toml`: package metadata, runtime dependencies, pytest configuration.
- `.gitignore`: Python caches, virtual environments, checkpoints, dataset roots, and generated cache files.
- `mypose/data/keypoints133.py`: canonical COCO-WholeBody part slices, names, root definition, graph edges, and validation helpers.
- `mypose/data/transforms.py`: 2D normalization, root-relative conversion, temporal window construction.
- `mypose/data/validation.py`: fail-fast checks for arrays and sample dictionaries.
- `mypose/data/coco_wholebody.py`: COCO-WholeBody JSON parser for official 2D annotations.
- `mypose/data/h3wb.py`: H3WB parser and cache dataset for 2D-to-3D training.
- `mypose/utils/camera.py`: camera/world conversion helpers and unit conversion.
- `mypose/utils/metrics.py`: MPJPE and local aligned metrics.
- `mypose/models/temporal_adapter.py`: strictly causal temporal conv adapter with streaming cache.
- `mypose/models/hrgcn_lifter.py`: HR-GCN-inspired graph lifter and optional temporal adapter wrapper.
- `mypose/models/losses.py`: part-aware whole-body loss.
- `mypose/engine/checkpoint.py`: checkpoint save/load.
- `mypose/engine/train.py`: config-driven training loop.
- `mypose/engine/evaluate.py`: evaluation loop and metric aggregation.
- `mypose/engine/infer_stream.py`: CPU/GPU streaming inference API.
- `tools/download_datasets.py`: active official downloader or guided verification for COCO-WholeBody and H3WB.
- `tools/prepare_coco_wholebody.py`: convert COCO-WholeBody JSON into normalized 2D cache.
- `tools/prepare_h3wb.py`: convert H3WB JSON into unified training cache.
- `tools/inspect_sample.py`: print and optionally render sample diagnostics.
- `configs/h3wb_hrgcn_t1.yaml`: framewise baseline config.
- `configs/h3wb_hrgcn_causal_t27.yaml`: causal temporal ablation config.
- `docs/datasets.md`: source URLs, license notes, directory layout, and preparation commands.
- `tests/fixtures/`: tiny synthetic JSON/NPZ fixtures for deterministic parser tests.
- `tests/test_*.py`: unit tests covering each boundary.

---

### Task 1: Project Package, Keypoint Layout, and Test Harness

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `mypose/__init__.py`
- Create: `mypose/data/__init__.py`
- Create: `mypose/data/keypoints133.py`
- Create: `tests/test_keypoints133.py`

**Interfaces:**
- Produces: `PART_SLICES: dict[str, slice]`
- Produces: `PART_INDICES: dict[str, list[int]]`
- Produces: `LEFT_HIP = 11`, `RIGHT_HIP = 12`
- Produces: `NUM_KEYPOINTS = 133`
- Produces: `validate_keypoints_shape(array, dims, name) -> None`
- Produces: `get_part_indices(part: str) -> list[int]`
- Produces: `COCO_WHOLEBODY_EDGES: list[tuple[int, int]]`

- [ ] **Step 1: Write failing keypoint layout tests**

Create `tests/test_keypoints133.py`:

```python
import numpy as np
import pytest

from mypose.data.keypoints133 import (
    COCO_WHOLEBODY_EDGES,
    LEFT_HIP,
    NUM_KEYPOINTS,
    PART_INDICES,
    PART_SLICES,
    RIGHT_HIP,
    get_part_indices,
    validate_keypoints_shape,
)


def test_part_slices_cover_all_133_keypoints_once():
    covered = []
    for part in ("body", "foot", "face", "left_hand", "right_hand"):
        covered.extend(range(PART_SLICES[part].start, PART_SLICES[part].stop))
    assert covered == list(range(133))
    assert NUM_KEYPOINTS == 133


def test_known_part_boundaries_match_coco_wholebody():
    assert PART_INDICES["body"] == list(range(0, 17))
    assert PART_INDICES["foot"] == list(range(17, 23))
    assert PART_INDICES["face"] == list(range(23, 91))
    assert PART_INDICES["left_hand"] == list(range(91, 112))
    assert PART_INDICES["right_hand"] == list(range(112, 133))
    assert LEFT_HIP == 11
    assert RIGHT_HIP == 12


def test_validate_keypoints_shape_accepts_expected_shapes():
    validate_keypoints_shape(np.zeros((133, 3), dtype=np.float32), dims=2, name="pose")
    validate_keypoints_shape(np.zeros((8, 133, 3), dtype=np.float32), dims=3, name="history")


def test_validate_keypoints_shape_rejects_wrong_count():
    with pytest.raises(ValueError, match="expected 133 keypoints"):
        validate_keypoints_shape(np.zeros((132, 3), dtype=np.float32), dims=2, name="pose")


def test_get_part_indices_rejects_unknown_part():
    with pytest.raises(KeyError, match="unknown keypoint part"):
        get_part_indices("tail")


def test_graph_edges_are_in_bounds_and_nonempty():
    assert len(COCO_WHOLEBODY_EDGES) > 120
    for a, b in COCO_WHOLEBODY_EDGES:
        assert 0 <= a < 133
        assert 0 <= b < 133
        assert a != b
```

- [ ] **Step 2: Run tests and verify they fail because package files do not exist**

Run:

```bash
pytest tests/test_keypoints133.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mypose'`.

- [ ] **Step 3: Create package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mypose"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "torch>=2.2",
  "PyYAML>=6.0",
  "requests>=2.31",
  "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
include = ["mypose*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-q"
```

Create `.gitignore`:

```gitignore
.worktrees/
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
build/
dist/
*.egg-info/
data/raw/
data/processed/
data/cache/
checkpoints/
runs/
*.pt
*.pth
*.npz
```

Create empty `mypose/__init__.py` and `mypose/data/__init__.py`.

- [ ] **Step 4: Implement keypoint layout**

Create `mypose/data/keypoints133.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

NUM_KEYPOINTS = 133
LEFT_HIP = 11
RIGHT_HIP = 12

PART_SLICES: dict[str, slice] = {
    "body": slice(0, 17),
    "foot": slice(17, 23),
    "face": slice(23, 91),
    "left_hand": slice(91, 112),
    "right_hand": slice(112, 133),
}

PART_INDICES: dict[str, list[int]] = {
    name: list(range(part_slice.start, part_slice.stop))
    for name, part_slice in PART_SLICES.items()
}

BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]
FOOT_EDGES = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
FACE_EDGES = [(i, i + 1) for i in range(23, 90)]
LEFT_HAND_EDGES = [
    (91, 92), (92, 93), (93, 94), (94, 95),
    (91, 96), (96, 97), (97, 98), (98, 99),
    (91, 100), (100, 101), (101, 102), (102, 103),
    (91, 104), (104, 105), (105, 106), (106, 107),
    (91, 108), (108, 109), (109, 110), (110, 111),
]
RIGHT_HAND_EDGES = [(a + 21, b + 21) for a, b in LEFT_HAND_EDGES]

COCO_WHOLEBODY_EDGES = BODY_EDGES + FOOT_EDGES + FACE_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES


def get_part_indices(part: str) -> list[int]:
    try:
        return PART_INDICES[part]
    except KeyError as exc:
        known = ", ".join(sorted(PART_INDICES))
        raise KeyError(f"unknown keypoint part {part!r}; expected one of {known}") from exc


def validate_keypoints_shape(array: np.ndarray, dims: int, name: str) -> None:
    if array.ndim != dims:
        raise ValueError(f"{name} expected {dims} dims, got shape {array.shape}")
    keypoint_axis = -2
    if array.shape[keypoint_axis] != NUM_KEYPOINTS:
        raise ValueError(f"{name} expected 133 keypoints, got shape {array.shape}")
    if array.shape[-1] not in (2, 3):
        raise ValueError(f"{name} expected coordinate dimension 2 or 3, got shape {array.shape}")
```

- [ ] **Step 5: Run keypoint tests**

Run:

```bash
pytest tests/test_keypoints133.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit package and keypoint layout**

```bash
git add pyproject.toml .gitignore mypose tests/test_keypoints133.py
git commit -m "feat: add 133-point keypoint layout"
```

---

### Task 2: Coordinate Transforms, Root-Relative Conversion, and Sample Validation

**Files:**
- Create: `mypose/data/transforms.py`
- Create: `mypose/data/validation.py`
- Create: `mypose/utils/__init__.py`
- Create: `mypose/utils/camera.py`
- Create: `tests/test_camera.py`
- Create: `tests/test_transforms_validation.py`

**Interfaces:**
- Consumes: `LEFT_HIP`, `RIGHT_HIP`, `NUM_KEYPOINTS`, `validate_keypoints_shape`
- Produces: `normalize_2d_image(keypoints_xyc: np.ndarray, image_size: tuple[int, int]) -> np.ndarray`
- Produces: `compute_pelvis_root(points_3d: np.ndarray) -> np.ndarray`
- Produces: `make_root_relative(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
- Produces: `meters_to_millimeters(points: np.ndarray) -> np.ndarray`
- Produces: `validate_sample(sample: dict) -> None`
- Produces: `make_causal_window(sequence: np.ndarray, frame_idx: int, window: int) -> np.ndarray`

- [ ] **Step 1: Write failing tests for transforms and validation**

Create `tests/test_camera.py`:

```python
import numpy as np

from mypose.data.transforms import compute_pelvis_root, make_root_relative
from mypose.utils.camera import meters_to_millimeters


def test_compute_pelvis_root_uses_hip_midpoint():
    pose = np.zeros((133, 3), dtype=np.float32)
    pose[11] = [10.0, 2.0, 4.0]
    pose[12] = [14.0, 6.0, 8.0]
    np.testing.assert_allclose(compute_pelvis_root(pose), [12.0, 4.0, 6.0])


def test_make_root_relative_subtracts_pelvis_from_all_points():
    pose = np.zeros((133, 3), dtype=np.float32)
    pose[11] = [2.0, 0.0, 0.0]
    pose[12] = [4.0, 0.0, 0.0]
    pose[0] = [13.0, 5.0, -1.0]
    rel, root = make_root_relative(pose)
    np.testing.assert_allclose(root, [3.0, 0.0, 0.0])
    np.testing.assert_allclose(rel[0], [10.0, 5.0, -1.0])
    np.testing.assert_allclose((rel[11] + rel[12]) / 2.0, [0.0, 0.0, 0.0])


def test_meters_to_millimeters_multiplies_by_1000():
    points = np.array([[1.2, -0.5, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(meters_to_millimeters(points), [[1200.0, -500.0, 0.0]])
```

Create `tests/test_transforms_validation.py`:

```python
import numpy as np
import pytest

from mypose.data.transforms import make_causal_window, normalize_2d_image
from mypose.data.validation import validate_sample


def test_normalize_2d_image_maps_pixels_to_centered_range_and_preserves_confidence():
    keypoints = np.zeros((133, 3), dtype=np.float32)
    keypoints[0] = [320.0, 240.0, 0.75]
    normalized = normalize_2d_image(keypoints, image_size=(640, 480))
    np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.75])


def test_make_causal_window_left_pads_first_frames_with_first_observation():
    seq = np.arange(5 * 133 * 3, dtype=np.float32).reshape(5, 133, 3)
    window = make_causal_window(seq, frame_idx=1, window=4)
    assert window.shape == (4, 133, 3)
    np.testing.assert_allclose(window[0], seq[0])
    np.testing.assert_allclose(window[1], seq[0])
    np.testing.assert_allclose(window[2], seq[0])
    np.testing.assert_allclose(window[3], seq[1])


def test_make_causal_window_rejects_future_frame_request():
    seq = np.zeros((5, 133, 3), dtype=np.float32)
    with pytest.raises(IndexError, match="frame_idx"):
        make_causal_window(seq, frame_idx=5, window=3)


def test_validate_sample_rejects_nan_target():
    sample = {
        "history_2d": np.zeros((1, 133, 3), dtype=np.float32),
        "target_3d": np.zeros((133, 3), dtype=np.float32),
        "target_mask": np.ones((133,), dtype=bool),
        "meta": {"source": "synthetic"},
    }
    sample["target_3d"][0, 0] = np.nan
    with pytest.raises(ValueError, match="target_3d contains non-finite"):
        validate_sample(sample)
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
pytest tests/test_camera.py tests/test_transforms_validation.py -v
```

Expected: FAIL with missing modules/functions.

- [ ] **Step 3: Implement camera utilities**

Create `mypose/utils/__init__.py` as an empty package marker.

Create `mypose/utils/camera.py`:

```python
from __future__ import annotations

import numpy as np


def meters_to_millimeters(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) * 1000.0
```

- [ ] **Step 4: Implement transforms**

Create `mypose/data/transforms.py`:

```python
from __future__ import annotations

import numpy as np

from mypose.data.keypoints133 import LEFT_HIP, RIGHT_HIP, validate_keypoints_shape


def normalize_2d_image(keypoints_xyc: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    keypoints = np.asarray(keypoints_xyc, dtype=np.float32).copy()
    validate_keypoints_shape(keypoints, dims=2, name="keypoints_xyc")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    keypoints[:, 0] = (keypoints[:, 0] / float(width) - 0.5) * 2.0
    keypoints[:, 1] = (keypoints[:, 1] / float(height) - 0.5) * 2.0
    return keypoints


def compute_pelvis_root(points_3d: np.ndarray) -> np.ndarray:
    points = np.asarray(points_3d, dtype=np.float32)
    validate_keypoints_shape(points, dims=2, name="points_3d")
    return (points[LEFT_HIP] + points[RIGHT_HIP]) * 0.5


def make_root_relative(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_3d, dtype=np.float32)
    root = compute_pelvis_root(points)
    return points - root[None, :], root


def make_causal_window(sequence: np.ndarray, frame_idx: int, window: int) -> np.ndarray:
    seq = np.asarray(sequence, dtype=np.float32)
    validate_keypoints_shape(seq, dims=3, name="sequence")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if frame_idx < 0 or frame_idx >= seq.shape[0]:
        raise IndexError(f"frame_idx {frame_idx} outside sequence length {seq.shape[0]}")
    start = frame_idx - window + 1
    indices = [max(0, idx) for idx in range(start, frame_idx + 1)]
    return seq[indices]
```

- [ ] **Step 5: Implement sample validation**

Create `mypose/data/validation.py`:

```python
from __future__ import annotations

import numpy as np

from mypose.data.keypoints133 import NUM_KEYPOINTS, validate_keypoints_shape


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def validate_sample(sample: dict) -> None:
    required = {"history_2d", "target_3d", "target_mask", "meta"}
    missing = required.difference(sample)
    if missing:
        raise ValueError(f"sample missing keys: {sorted(missing)}")

    history = np.asarray(sample["history_2d"])
    target = np.asarray(sample["target_3d"])
    mask = np.asarray(sample["target_mask"])

    validate_keypoints_shape(history, dims=3, name="history_2d")
    validate_keypoints_shape(target, dims=2, name="target_3d")
    if mask.shape not in ((NUM_KEYPOINTS,), (NUM_KEYPOINTS, 1)):
        raise ValueError(f"target_mask expected shape (133,) or (133, 1), got {mask.shape}")
    if mask.astype(bool).sum() == 0:
        raise ValueError("target_mask has no valid keypoints")
    _require_finite("history_2d", history)
    _require_finite("target_3d", target)
```

- [ ] **Step 6: Run transform and validation tests**

Run:

```bash
pytest tests/test_camera.py tests/test_transforms_validation.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit transforms and validation**

```bash
git add mypose/data/transforms.py mypose/data/validation.py mypose/utils tests/test_camera.py tests/test_transforms_validation.py
git commit -m "feat: add pose transforms and validation"
```

---

### Task 3: Metrics and Part-Aware Losses

**Files:**
- Create: `mypose/utils/metrics.py`
- Create: `mypose/models/__init__.py`
- Create: `mypose/models/losses.py`
- Create: `tests/test_metrics_losses.py`

**Interfaces:**
- Consumes: `PART_INDICES`, `LEFT_HIP`, `RIGHT_HIP`
- Produces: `mpjpe(pred, target, mask=None) -> torch.Tensor`
- Produces: `part_mpjpe(pred, target, part, mask=None) -> torch.Tensor`
- Produces: `aligned_mpjpe(pred, target, indices, anchor_index, mask=None) -> torch.Tensor`
- Produces: `WholeBodyLoss(part_weights: dict[str, float], local_weights: dict[str, float], bone_weight: float) -> nn.Module`
- Produces: `WholeBodyLoss.forward(pred, target, target_mask) -> dict[str, torch.Tensor]`

- [ ] **Step 1: Write failing tests for metrics and loss weighting**

Create `tests/test_metrics_losses.py`:

```python
import torch

from mypose.models.losses import WholeBodyLoss
from mypose.utils.metrics import aligned_mpjpe, mpjpe, part_mpjpe


def test_mpjpe_masks_invalid_points():
    pred = torch.zeros(1, 133, 3)
    target = torch.zeros(1, 133, 3)
    target[:, 0] = 10.0
    mask = torch.zeros(1, 133, dtype=torch.bool)
    assert mpjpe(pred, target, mask).item() == 0.0
    mask[:, 0] = True
    assert mpjpe(pred, target, mask).item() > 0.0


def test_part_mpjpe_uses_requested_indices():
    pred = torch.zeros(1, 133, 3)
    target = torch.zeros(1, 133, 3)
    target[:, 91:112] = 2.0
    assert part_mpjpe(pred, target, "left_hand").item() > 0.0
    assert part_mpjpe(pred, target, "body").item() == 0.0


def test_aligned_mpjpe_removes_anchor_translation():
    pred = torch.zeros(1, 133, 3)
    target = torch.zeros(1, 133, 3)
    pred[:, 91:112] = 5.0
    target[:, 91:112] = 7.0
    assert aligned_mpjpe(pred, target, list(range(91, 112)), anchor_index=91).item() == 0.0


def test_wholebody_loss_hand_weight_changes_total_loss():
    pred = torch.zeros(1, 133, 3, requires_grad=True)
    target = torch.zeros(1, 133, 3)
    target[:, 91:112] = 1.0
    mask = torch.ones(1, 133, dtype=torch.bool)
    low = WholeBodyLoss(part_weights={"left_hand": 1.0})
    high = WholeBodyLoss(part_weights={"left_hand": 10.0})
    assert high(pred, target, mask)["total"].item() > low(pred, target, mask)["total"].item()
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
pytest tests/test_metrics_losses.py -v
```

Expected: FAIL with missing modules/functions.

- [ ] **Step 3: Implement metric functions**

Create `mypose/utils/metrics.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

import torch

from mypose.data.keypoints133 import PART_INDICES, get_part_indices


def _valid_mask(mask: torch.Tensor | None, pred: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(pred.shape[:-1], dtype=torch.bool, device=pred.device)
    return mask.to(device=pred.device, dtype=torch.bool)


def mpjpe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    distances = torch.linalg.norm(pred - target, dim=-1)
    valid = _valid_mask(mask, pred)
    if valid.sum() == 0:
        return distances.sum() * 0.0
    return distances[valid].mean()


def part_mpjpe(pred: torch.Tensor, target: torch.Tensor, part: str, mask: torch.Tensor | None = None) -> torch.Tensor:
    indices = get_part_indices(part)
    part_mask = None if mask is None else mask[:, indices]
    return mpjpe(pred[:, indices], target[:, indices], part_mask)


def aligned_mpjpe(
    pred: torch.Tensor,
    target: torch.Tensor,
    indices: Sequence[int],
    anchor_index: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    anchor_pred = pred[:, anchor_index:anchor_index + 1]
    anchor_target = target[:, anchor_index:anchor_index + 1]
    local_pred = pred[:, indices] - anchor_pred
    local_target = target[:, indices] - anchor_target
    local_mask = None if mask is None else mask[:, indices]
    return mpjpe(local_pred, local_target, local_mask)
```

- [ ] **Step 4: Implement WholeBodyLoss**

Create `mypose/models/__init__.py` as an empty package marker.

Create `mypose/models/losses.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from mypose.data.keypoints133 import COCO_WHOLEBODY_EDGES
from mypose.utils.metrics import aligned_mpjpe, mpjpe, part_mpjpe


DEFAULT_PART_WEIGHTS = {
    "body": 1.0,
    "foot": 1.5,
    "face": 2.0,
    "left_hand": 2.5,
    "right_hand": 2.5,
}


class WholeBodyLoss(nn.Module):
    def __init__(
        self,
        part_weights: dict[str, float] | None = None,
        local_weights: dict[str, float] | None = None,
        bone_weight: float = 0.01,
    ) -> None:
        super().__init__()
        weights = DEFAULT_PART_WEIGHTS.copy()
        if part_weights:
            weights.update(part_weights)
        self.part_weights = weights
        self.local_weights = {"face": 1.0, "left_hand": 1.0, "right_hand": 1.0}
        if local_weights:
            self.local_weights.update(local_weights)
        self.bone_weight = float(bone_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        losses["whole"] = mpjpe(pred, target, target_mask)
        total = losses["whole"]
        for part, weight in self.part_weights.items():
            value = part_mpjpe(pred, target, part, target_mask)
            losses[f"{part}_mpjpe"] = value
            total = total + float(weight) * value
        losses["face_local"] = aligned_mpjpe(pred, target, list(range(23, 91)), anchor_index=30, mask=target_mask)
        losses["left_hand_local"] = aligned_mpjpe(pred, target, list(range(91, 112)), anchor_index=91, mask=target_mask)
        losses["right_hand_local"] = aligned_mpjpe(pred, target, list(range(112, 133)), anchor_index=112, mask=target_mask)
        for name, weight in self.local_weights.items():
            total = total + float(weight) * losses[f"{name}_local"]
        losses["bone"] = self._bone_loss(pred, target)
        losses["total"] = total + self.bone_weight * losses["bone"]
        return losses

    def _bone_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        edge_index = torch.tensor(COCO_WHOLEBODY_EDGES, dtype=torch.long, device=pred.device)
        pred_len = torch.linalg.norm(pred[:, edge_index[:, 0]] - pred[:, edge_index[:, 1]], dim=-1)
        target_len = torch.linalg.norm(target[:, edge_index[:, 0]] - target[:, edge_index[:, 1]], dim=-1)
        return torch.mean(torch.abs(pred_len - target_len))
```

- [ ] **Step 5: Run metric and loss tests**

Run:

```bash
pytest tests/test_metrics_losses.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit metrics and losses**

```bash
git add mypose/utils/metrics.py mypose/models tests/test_metrics_losses.py
git commit -m "feat: add whole-body metrics and losses"
```

---

### Task 4: Official Dataset Downloaders, Dataset Documentation, and COCO Parser

**Files:**
- Create: `tools/download_datasets.py`
- Create: `tools/prepare_coco_wholebody.py`
- Create: `mypose/data/coco_wholebody.py`
- Create: `docs/datasets.md`
- Create: `tests/fixtures/coco_wholebody_tiny.json`
- Create: `tests/test_coco_wholebody.py`

**Interfaces:**
- Consumes: `normalize_2d_image`, `validate_keypoints_shape`
- Produces: `download_file(url: str, dest: Path) -> None`
- Produces: `download_coco_wholebody(root: Path, with_images: bool) -> None`
- Produces: `download_h3wb(root: Path) -> None`
- Produces: `load_coco_wholebody_annotations(path: Path) -> list[dict]`
- Produces: `CocoWholeBodyDataset(annotation_file: Path, image_root: Path | None = None)`

- [ ] **Step 1: Write failing COCO parser test and tiny fixture**

Create `tests/fixtures/coco_wholebody_tiny.json`:

```json
{
  "images": [
    {"id": 7, "file_name": "000000000007.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 7,
      "bbox": [100, 50, 200, 300],
      "keypoints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "foot_kpts": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "face_kpts": [],
      "lefthand_kpts": [],
      "righthand_kpts": []
    }
  ]
}
```

Create `tests/test_coco_wholebody.py`:

```python
from pathlib import Path

import numpy as np

from mypose.data.coco_wholebody import CocoWholeBodyDataset, load_coco_wholebody_annotations


def test_load_coco_wholebody_annotations_pads_missing_optional_parts():
    samples = load_coco_wholebody_annotations(Path("tests/fixtures/coco_wholebody_tiny.json"))
    assert len(samples) == 1
    sample = samples[0]
    assert sample["keypoints_2d"].shape == (133, 3)
    assert sample["image_size"] == (640, 480)
    assert sample["meta"]["image_id"] == 7
    np.testing.assert_allclose(sample["keypoints_2d"][0], [0.0, 0.0, 0.0])


def test_dataset_returns_normalized_history_sample():
    dataset = CocoWholeBodyDataset(Path("tests/fixtures/coco_wholebody_tiny.json"))
    sample = dataset[0]
    assert sample["history_2d"].shape == (1, 133, 3)
    assert sample["meta"]["source"] == "coco-wholebody"
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
pytest tests/test_coco_wholebody.py -v
```

Expected: FAIL with missing parser module.

- [ ] **Step 3: Implement COCO-WholeBody parser**

Create `mypose/data/coco_wholebody.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mypose.data.transforms import normalize_2d_image
from mypose.data.validation import validate_sample


def _reshape_part(values: list[float], expected_points: int) -> np.ndarray:
    if not values:
        return np.zeros((expected_points, 3), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    if arr.shape[0] != expected_points:
        raise ValueError(f"expected {expected_points} points, got {arr.shape[0]}")
    return arr


def _annotation_to_133(annotation: dict[str, Any]) -> np.ndarray:
    body = _reshape_part(annotation.get("keypoints", []), 17)
    foot = _reshape_part(annotation.get("foot_kpts", []), 6)
    face = _reshape_part(annotation.get("face_kpts", []), 68)
    left = _reshape_part(annotation.get("lefthand_kpts", []), 21)
    right = _reshape_part(annotation.get("righthand_kpts", []), 21)
    return np.concatenate([body, foot, face, left, right], axis=0).astype(np.float32)


def load_coco_wholebody_annotations(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in payload["images"]}
    samples = []
    for ann in payload["annotations"]:
        image = images[ann["image_id"]]
        keypoints = _annotation_to_133(ann)
        samples.append({
            "keypoints_2d": keypoints,
            "image_size": (int(image["width"]), int(image["height"])),
            "bbox": ann.get("bbox"),
            "meta": {
                "source": "coco-wholebody",
                "image_id": ann["image_id"],
                "annotation_id": ann["id"],
                "file_name": image["file_name"],
            },
        })
    return samples


class CocoWholeBodyDataset:
    def __init__(self, annotation_file: Path, image_root: Path | None = None) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_root = image_root
        self.samples = load_coco_wholebody_annotations(self.annotation_file)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        raw = self.samples[index]
        keypoints = normalize_2d_image(raw["keypoints_2d"], raw["image_size"])
        sample = {
            "history_2d": keypoints[None, :, :],
            "target_3d": np.zeros((133, 3), dtype=np.float32),
            "target_mask": np.zeros((133,), dtype=bool),
            "meta": raw["meta"],
        }
        sample["meta"]["source"] = "coco-wholebody"
        return sample
```

- [ ] **Step 4: Implement dataset downloader command**

Create `tools/download_datasets.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

COCO_IMAGE_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}

COCO_WHOLEBODY_SOURCES = {
    "official_repo": "https://github.com/jin-s13/COCO-WholeBody",
    "official_readme_downloads": "https://github.com/jin-s13/COCO-WholeBody#download",
}

H3WB_SOURCES = {
    "official_repo": "https://github.com/wholebody3d/wholebody3d",
    "official_readme_downloads": "https://github.com/wholebody3d/wholebody3d#download",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists: {dest}")
        return
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        with dest.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))


def download_coco_wholebody(root: Path, with_images: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if with_images:
        for split, url in COCO_IMAGE_URLS.items():
            download_file(url, root / "images" / Path(urlparse(url).path).name)
    print("COCO-WholeBody annotations are hosted by the official repository via OneDrive, Google Drive, BaiduPan, and OpenXLab.")
    print(f"Open: {COCO_WHOLEBODY_SOURCES['official_readme_downloads']}")
    print(f"Place train/val annotation JSON files under: {root / 'annotations'}")


def download_h3wb(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    print("H3WB annotations and test sets are hosted by the official repository via Google Drive.")
    print(f"Open: {H3WB_SOURCES['official_readme_downloads']}")
    print(f"Place 2Dto3D_train.json and available test JSON files under: {root / 'annotations'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["coco-wholebody", "h3wb"], required=True)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--with-images", action="store_true")
    args = parser.parse_args()
    if args.dataset == "coco-wholebody":
        download_coco_wholebody(args.root / "coco-wholebody", args.with_images)
    else:
        download_h3wb(args.root / "h3wb")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement COCO preparation command**

Create `tools/prepare_coco_wholebody.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mypose.data.coco_wholebody import load_coco_wholebody_annotations
from mypose.data.transforms import normalize_2d_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    samples = load_coco_wholebody_annotations(args.annotations)
    inputs = np.stack([normalize_2d_image(s["keypoints_2d"], s["image_size"]) for s in samples])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, inputs_2d=inputs)
    print(f"wrote {args.out} with inputs_2d shape {inputs.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Document official sources and license constraints**

Create `docs/datasets.md`:

```markdown
# Datasets

## COCO-WholeBody

Official repository: https://github.com/jin-s13/COCO-WholeBody

Images are COCO 2017 train/val images from http://images.cocodataset.org.
The official COCO-WholeBody repository lists OneDrive, Google Drive, BaiduPan, and OpenXLab annotation downloads. The annotations are for research and non-commercial use; commercial usage requires contacting the dataset owners. The annotations use the COCO-WholeBody 133 layout: 17 body, 6 foot, 68 face, and 42 hand keypoints.

Commands:

```bash
python tools/download_datasets.py --dataset coco-wholebody --root data/raw --with-images
python tools/prepare_coco_wholebody.py --annotations data/raw/coco-wholebody/annotations/coco_wholebody_train_v1.0.json --out data/processed/coco_wholebody_train.npz
```

## H3WB

Official repository: https://github.com/wholebody3d/wholebody3d

H3WB extends Human3.6M to 133 whole-body 3D keypoints with the same skeleton layout as COCO-WholeBody. The official repository lists Google Drive downloads for H3WB annotations and the 2D-to-3D task. Human3.6M images are governed by the Human3.6M license.

Expected files for this project:

```text
data/raw/h3wb/annotations/2Dto3D_train.json
data/raw/h3wb/annotations/2Dto3D_test_2d.json
```

Commands:

```bash
python tools/download_datasets.py --dataset h3wb --root data/raw
python tools/prepare_h3wb.py --annotations data/raw/h3wb/annotations/2Dto3D_train.json --out data/processed/h3wb_2dto3d_train.npz
```
```

- [ ] **Step 7: Run parser tests**

Run:

```bash
pytest tests/test_coco_wholebody.py -v
```

Expected: PASS.

- [ ] **Step 8: Run downloader help smoke test**

Run:

```bash
python tools/download_datasets.py --help
```

Expected: command prints `--dataset {coco-wholebody,h3wb}`.

- [ ] **Step 9: Commit download tooling and COCO parser**

```bash
git add tools/download_datasets.py tools/prepare_coco_wholebody.py mypose/data/coco_wholebody.py docs/datasets.md tests/fixtures/coco_wholebody_tiny.json tests/test_coco_wholebody.py
git commit -m "feat: add dataset download and coco parser"
```

---

### Task 5: H3WB Parser, Cache Dataset, and Preparation Command

**Files:**
- Create: `mypose/data/h3wb.py`
- Create: `tools/prepare_h3wb.py`
- Create: `tests/fixtures/h3wb_tiny.json`
- Create: `tests/test_h3wb.py`

**Interfaces:**
- Consumes: `make_root_relative`, `normalize_2d_image`, `make_causal_window`, `validate_sample`
- Produces: `load_h3wb_json(path: Path) -> list[dict]`
- Produces: `H3WBDataset(cache_file: Path, window: int) -> Dataset`
- Produces cache fields: `inputs_2d`, `targets_3d`, `target_masks`, `frame_ids`, `metas`

- [ ] **Step 1: Write failing H3WB tests and fixture**

Create `tests/fixtures/h3wb_tiny.json`:

```json
{
  "sample_000001": {
    "image_path": "S1/Directions/000001.jpg",
    "bbox": [0, 0, 640, 480],
    "keypoint_2d": {
      "0": {"x": 320, "y": 240},
      "11": {"x": 300, "y": 300},
      "12": {"x": 340, "y": 300}
    },
    "keypoint_3d": {
      "0": {"x": 0, "y": 100, "z": 1000},
      "11": {"x": -100, "y": 0, "z": 1000},
      "12": {"x": 100, "y": 0, "z": 1000}
    }
  }
}
```

Create `tests/test_h3wb.py`:

```python
from pathlib import Path

import numpy as np

from mypose.data.h3wb import H3WBDataset, load_h3wb_json, write_h3wb_cache


def test_load_h3wb_json_fills_missing_points_and_root_relative_target():
    samples = load_h3wb_json(Path("tests/fixtures/h3wb_tiny.json"))
    assert len(samples) == 1
    sample = samples[0]
    assert sample["history_2d"].shape == (1, 133, 3)
    assert sample["target_3d"].shape == (133, 3)
    np.testing.assert_allclose((sample["target_3d"][11] + sample["target_3d"][12]) / 2.0, [0.0, 0.0, 0.0])
    assert sample["target_mask"][0]
    assert not sample["target_mask"][25]


def test_h3wb_dataset_reads_npz_cache_with_causal_window(tmp_path):
    cache = tmp_path / "h3wb.npz"
    write_h3wb_cache(Path("tests/fixtures/h3wb_tiny.json"), cache)
    dataset = H3WBDataset(cache, window=3)
    sample = dataset[0]
    assert sample["history_2d"].shape == (3, 133, 3)
    assert sample["target_3d"].shape == (133, 3)
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
pytest tests/test_h3wb.py -v
```

Expected: FAIL with missing H3WB parser.

- [ ] **Step 3: Implement H3WB parser and cache dataset**

Create `mypose/data/h3wb.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from mypose.data.transforms import make_causal_window, make_root_relative, normalize_2d_image
from mypose.data.validation import validate_sample


def _dict_points(entry: dict[str, Any], dims: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros((133, dims), dtype=np.float32)
    mask = np.zeros((133,), dtype=bool)
    for key, value in entry.items():
        idx = int(key)
        if idx < 0 or idx >= 133:
            raise ValueError(f"keypoint index {idx} outside 0..132")
        if dims == 2:
            points[idx] = [float(value["x"]), float(value["y"])]
        else:
            points[idx] = [float(value["x"]), float(value["y"]), float(value["z"])]
        mask[idx] = True
    return points, mask


def load_h3wb_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for sample_id, item in payload.items():
        xy, mask2d = _dict_points(item.get("keypoint_2d") or item.get("keypont_2d"), 2)
        xyz, mask3d = _dict_points(item.get("keypoint_3d") or item.get("keypont_3d"), 3)
        xyc = np.concatenate([xy, mask2d[:, None].astype(np.float32)], axis=1)
        bbox = item.get("bbox", [0, 0, 1, 1])
        width = max(1, int(bbox[2] if len(bbox) == 4 else 1))
        height = max(1, int(bbox[3] if len(bbox) == 4 else 1))
        norm_2d = normalize_2d_image(xyc, (width, height))
        rel_3d, _ = make_root_relative(xyz)
        sample = {
            "history_2d": norm_2d[None, :, :],
            "target_3d": rel_3d,
            "target_mask": mask3d,
            "meta": {
                "source": "h3wb",
                "sample_id": sample_id,
                "image_path": item.get("image_path", ""),
            },
        }
        validate_sample(sample)
        samples.append(sample)
    return samples


def write_h3wb_cache(annotation_file: Path, out_file: Path) -> None:
    samples = load_h3wb_json(annotation_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        inputs_2d=np.stack([s["history_2d"][0] for s in samples]),
        targets_3d=np.stack([s["target_3d"] for s in samples]),
        target_masks=np.stack([s["target_mask"] for s in samples]),
        metas=np.asarray([s["meta"] for s in samples], dtype=object),
    )


class H3WBDataset(Dataset):
    def __init__(self, cache_file: Path, window: int) -> None:
        payload = np.load(cache_file, allow_pickle=True)
        self.inputs_2d = payload["inputs_2d"].astype(np.float32)
        self.targets_3d = payload["targets_3d"].astype(np.float32)
        self.target_masks = payload["target_masks"].astype(bool)
        self.metas = payload["metas"]
        self.window = int(window)

    def __len__(self) -> int:
        return self.inputs_2d.shape[0]

    def __getitem__(self, index: int) -> dict:
        sample = {
            "history_2d": make_causal_window(self.inputs_2d, index, self.window),
            "target_3d": self.targets_3d[index],
            "target_mask": self.target_masks[index],
            "meta": dict(self.metas[index]),
        }
        validate_sample(sample)
        return sample
```

- [ ] **Step 4: Implement H3WB preparation command**

Create `tools/prepare_h3wb.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from mypose.data.h3wb import write_h3wb_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_h3wb_cache(args.annotations, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run H3WB tests**

Run:

```bash
pytest tests/test_h3wb.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit H3WB parser and preparation command**

```bash
git add mypose/data/h3wb.py tools/prepare_h3wb.py tests/fixtures/h3wb_tiny.json tests/test_h3wb.py
git commit -m "feat: add h3wb parser and cache dataset"
```

---

### Task 6: Causal Temporal Adapter and HR-GCN-Inspired Lifter

**Files:**
- Create: `mypose/models/temporal_adapter.py`
- Create: `mypose/models/hrgcn_lifter.py`
- Create: `tests/test_causal_window.py`
- Create: `tests/test_model_forward.py`

**Interfaces:**
- Consumes: `COCO_WHOLEBODY_EDGES`
- Produces: `CausalTemporalAdapter(in_channels: int, hidden_channels: int, kernel_size: int = 3, dilation: int = 1)`
- Produces: `CausalTemporalAdapter.forward(x: torch.Tensor) -> torch.Tensor` where `x` shape is `(B, T, J, C)`
- Produces: `CausalTemporalAdapter.reset_stream() -> None`
- Produces: `CausalTemporalAdapter.step(frame: torch.Tensor) -> torch.Tensor`
- Produces: `HRGCNLifter(num_keypoints: int = 133, in_channels: int = 3, hidden_channels: int = 128, use_temporal: bool = False)`
- Produces: `HRGCNLifter.forward(history_2d: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
- Produces: `HRGCNLifter.forward_frame(frame_2d: torch.Tensor) -> torch.Tensor`
- Produces: `HRGCNLifter.reset_stream() -> None`
- Produces: `HRGCNLifter.step(frame_2d: torch.Tensor) -> torch.Tensor`

- [ ] **Step 1: Write failing causal and forward tests**

Create `tests/test_causal_window.py`:

```python
import torch

from mypose.models.temporal_adapter import CausalTemporalAdapter


def test_temporal_adapter_output_does_not_depend_on_future_frames():
    torch.manual_seed(0)
    model = CausalTemporalAdapter(in_channels=3, hidden_channels=8, kernel_size=3)
    model.eval()
    x1 = torch.randn(1, 5, 133, 3)
    x2 = x1.clone()
    x2[:, 4] = x2[:, 4] + 1000.0
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    torch.testing.assert_close(y1[:, :4], y2[:, :4])


def test_temporal_adapter_stream_step_matches_batch_last_frame():
    torch.manual_seed(1)
    model = CausalTemporalAdapter(in_channels=3, hidden_channels=8, kernel_size=3)
    model.eval()
    seq = torch.randn(1, 4, 133, 3)
    with torch.no_grad():
        batch = model(seq)[:, -1]
        model.reset_stream()
        stepped = None
        for idx in range(seq.shape[1]):
            stepped = model.step(seq[:, idx])
    torch.testing.assert_close(batch, stepped)
```

Create `tests/test_model_forward.py`:

```python
import torch

from mypose.models.hrgcn_lifter import HRGCNLifter


def test_framewise_lifter_outputs_133_3():
    model = HRGCNLifter(use_temporal=False, hidden_channels=32)
    x = torch.randn(2, 1, 133, 3)
    y = model(x)
    assert y.shape == (2, 133, 3)


def test_temporal_lifter_outputs_133_3_on_cpu():
    model = HRGCNLifter(use_temporal=True, hidden_channels=32)
    x = torch.randn(2, 5, 133, 3)
    y = model(x)
    assert y.shape == (2, 133, 3)


def test_streaming_lifter_step_outputs_current_pose():
    model = HRGCNLifter(use_temporal=True, hidden_channels=32)
    model.reset_stream()
    y = model.step(torch.randn(1, 133, 3))
    assert y.shape == (1, 133, 3)
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
pytest tests/test_causal_window.py tests/test_model_forward.py -v
```

Expected: FAIL with missing model modules.

- [ ] **Step 3: Implement causal temporal adapter**

Create `mypose/models/temporal_adapter.py`:

```python
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalTemporalAdapter(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = 0
        self.input = nn.Linear(in_channels, hidden_channels)
        self.depthwise = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,
            padding=padding,
        )
        self.pointwise = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.output = nn.Linear(hidden_channels, in_channels)
        self.activation = nn.GELU()
        self._stream: list[torch.Tensor] = []

    @property
    def receptive_field(self) -> int:
        return (self.kernel_size - 1) * self.dilation + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, joints, channels = x.shape
        h = self.input(x).permute(0, 2, 3, 1).reshape(bsz * joints, -1, steps)
        left_pad = self.receptive_field - 1
        h = F.pad(h, (left_pad, 0))
        h = self.depthwise(h)
        h = self.activation(self.pointwise(h))
        h = h.reshape(bsz, joints, -1, steps).permute(0, 3, 1, 2)
        return x + self.output(h)

    def reset_stream(self) -> None:
        self._stream = []

    def step(self, frame: torch.Tensor) -> torch.Tensor:
        self._stream.append(frame)
        if len(self._stream) > self.receptive_field:
            self._stream = self._stream[-self.receptive_field:]
        seq = torch.stack(self._stream, dim=1)
        return self.forward(seq)[:, -1]
```

- [ ] **Step 4: Implement HR-GCN-inspired lifter**

Create `mypose/models/hrgcn_lifter.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from mypose.data.keypoints133 import COCO_WHOLEBODY_EDGES, NUM_KEYPOINTS
from mypose.models.temporal_adapter import CausalTemporalAdapter


class GraphBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(channels, channels)
        self.neighbor_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        edges = torch.tensor(COCO_WHOLEBODY_EDGES, dtype=torch.long)
        reverse = edges[:, [1, 0]]
        self.register_buffer("edges", torch.cat([edges, reverse], dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = self.edges[:, 0]
        dst = self.edges[:, 1]
        messages = torch.zeros_like(x)
        messages.index_add_(1, dst, x[:, src])
        degree = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        messages = messages / degree.clamp_min(1.0)[None, :, None]
        return self.norm(x + self.act(self.self_proj(x) + self.neighbor_proj(messages)))


class HRGCNLifter(nn.Module):
    def __init__(
        self,
        num_keypoints: int = NUM_KEYPOINTS,
        in_channels: int = 3,
        hidden_channels: int = 128,
        use_temporal: bool = False,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.use_temporal = use_temporal
        self.temporal = CausalTemporalAdapter(in_channels, hidden_channels, kernel_size=3) if use_temporal else None
        self.input = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([GraphBlock(hidden_channels), GraphBlock(hidden_channels), GraphBlock(hidden_channels)])
        self.body_head = nn.Linear(hidden_channels, 3)
        self.fine_head = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.GELU(), nn.Linear(hidden_channels, 3))

    def forward(self, history_2d: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = history_2d
        if self.temporal is not None:
            x = self.temporal(x)
        frame = x[:, -1]
        return self.forward_frame(frame)

    def forward_frame(self, frame_2d: torch.Tensor) -> torch.Tensor:
        h = self.input(frame_2d)
        for block in self.blocks:
            h = block(h)
        out = self.body_head(h)
        fine_indices = list(range(23, 133))
        out[:, fine_indices] = self.fine_head(h[:, fine_indices])
        return out

    def reset_stream(self) -> None:
        if self.temporal is not None:
            self.temporal.reset_stream()

    def step(self, frame_2d: torch.Tensor) -> torch.Tensor:
        if self.temporal is None:
            return self.forward_frame(frame_2d)
        adapted = self.temporal.step(frame_2d)
        return self.forward_frame(adapted)
```

- [ ] **Step 5: Run causal and model tests**

Run:

```bash
pytest tests/test_causal_window.py tests/test_model_forward.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit temporal adapter and lifter**

```bash
git add mypose/models/temporal_adapter.py mypose/models/hrgcn_lifter.py tests/test_causal_window.py tests/test_model_forward.py
git commit -m "feat: add causal hrgcn lifter"
```

---

### Task 7: Training, Evaluation, Checkpointing, and Configs

**Files:**
- Create: `mypose/engine/__init__.py`
- Create: `mypose/engine/checkpoint.py`
- Create: `mypose/engine/evaluate.py`
- Create: `mypose/engine/train.py`
- Create: `configs/h3wb_hrgcn_t1.yaml`
- Create: `configs/h3wb_hrgcn_causal_t27.yaml`
- Create: `tests/test_engine_smoke.py`

**Interfaces:**
- Consumes: `H3WBDataset`, `HRGCNLifter`, `WholeBodyLoss`, metrics
- Produces: `save_checkpoint(path, model, optimizer, epoch, metrics) -> None`
- Produces: `load_checkpoint(path, model, optimizer=None) -> dict`
- Produces: `evaluate(model, dataloader, device) -> dict[str, float]`
- Produces: CLI `python -m mypose.engine.train --config configs/h3wb_hrgcn_t1.yaml`

- [ ] **Step 1: Write failing engine smoke tests**

Create `tests/test_engine_smoke.py`:

```python
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from mypose.engine.checkpoint import load_checkpoint, save_checkpoint
from mypose.engine.evaluate import evaluate
from mypose.models.hrgcn_lifter import HRGCNLifter


class TinyPoseDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "history_2d": torch.zeros(1, 133, 3),
            "target_3d": torch.zeros(133, 3),
            "target_mask": torch.ones(133, dtype=torch.bool),
            "meta": {"index": index},
        }


def test_checkpoint_roundtrip(tmp_path):
    model = HRGCNLifter(hidden_channels=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, epoch=3, metrics={"MPJPE_whole": 1.5})
    state = load_checkpoint(path, model, optimizer)
    assert state["epoch"] == 3
    assert state["metrics"]["MPJPE_whole"] == 1.5


def test_evaluate_returns_part_metrics():
    loader = DataLoader(TinyPoseDataset(), batch_size=2)
    model = HRGCNLifter(hidden_channels=16)
    metrics = evaluate(model, loader, device=torch.device("cpu"))
    assert "MPJPE_whole" in metrics
    assert "MPJPE_hands_wrist_aligned" in metrics
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
pytest tests/test_engine_smoke.py -v
```

Expected: FAIL with missing engine modules.

- [ ] **Step 3: Implement checkpoint utilities**

Create `mypose/engine/__init__.py` as an empty package marker.

Create `mypose/engine/checkpoint.py`:

```python
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state
```

- [ ] **Step 4: Implement evaluation**

Create `mypose/engine/evaluate.py`:

```python
from __future__ import annotations

import torch

from mypose.utils.metrics import aligned_mpjpe, mpjpe, part_mpjpe


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {
        "MPJPE_whole": 0.0,
        "MPJPE_body": 0.0,
        "MPJPE_feet": 0.0,
        "MPJPE_face": 0.0,
        "MPJPE_face_nose_aligned": 0.0,
        "MPJPE_left_hand": 0.0,
        "MPJPE_right_hand": 0.0,
        "MPJPE_hands_wrist_aligned": 0.0,
    }
    count = 0
    for batch in dataloader:
        history = batch["history_2d"].to(device=device, dtype=torch.float32)
        target = batch["target_3d"].to(device=device, dtype=torch.float32)
        mask = batch["target_mask"].to(device=device, dtype=torch.bool)
        pred = model(history)
        totals["MPJPE_whole"] += mpjpe(pred, target, mask).item()
        totals["MPJPE_body"] += part_mpjpe(pred, target, "body", mask).item()
        totals["MPJPE_feet"] += part_mpjpe(pred, target, "foot", mask).item()
        totals["MPJPE_face"] += part_mpjpe(pred, target, "face", mask).item()
        totals["MPJPE_face_nose_aligned"] += aligned_mpjpe(pred, target, list(range(23, 91)), 30, mask).item()
        totals["MPJPE_left_hand"] += part_mpjpe(pred, target, "left_hand", mask).item()
        totals["MPJPE_right_hand"] += part_mpjpe(pred, target, "right_hand", mask).item()
        left = aligned_mpjpe(pred, target, list(range(91, 112)), 91, mask)
        right = aligned_mpjpe(pred, target, list(range(112, 133)), 112, mask)
        totals["MPJPE_hands_wrist_aligned"] += ((left + right) * 0.5).item()
        count += 1
    return {name: value / max(1, count) for name, value in totals.items()}
```

- [ ] **Step 5: Implement training CLI**

Create `mypose/engine/train.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from mypose.data.h3wb import H3WBDataset
from mypose.engine.checkpoint import save_checkpoint
from mypose.engine.evaluate import evaluate
from mypose.models.hrgcn_lifter import HRGCNLifter
from mypose.models.losses import WholeBodyLoss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    requested_device = cfg["train"]["device"]
    selected_device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else "cpu" if requested_device == "auto" else requested_device
    device = torch.device(selected_device)
    train_set = H3WBDataset(Path(cfg["data"]["train_cache"]), window=int(cfg["data"]["window"]))
    train_loader = DataLoader(train_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"]["num_workers"]))
    model = HRGCNLifter(
        hidden_channels=int(cfg["model"]["hidden_channels"]),
        use_temporal=bool(cfg["model"]["use_temporal"]),
    ).to(device)
    criterion = WholeBodyLoss(part_weights=cfg["loss"]["part_weights"], local_weights=cfg["loss"]["local_weights"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    out_dir = Path(cfg["train"]["out_dir"])
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            history = batch["history_2d"].to(device=device, dtype=torch.float32)
            target = batch["target_3d"].to(device=device, dtype=torch.float32)
            mask = batch["target_mask"].to(device=device, dtype=torch.bool)
            losses = criterion(model(history), target, mask)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
        metrics = evaluate(model, train_loader, device)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch=epoch, metrics=metrics)
        print({"epoch": epoch, **metrics})


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add baseline and causal configs**

Create `configs/h3wb_hrgcn_t1.yaml`:

```yaml
data:
  train_cache: data/processed/h3wb_2dto3d_train.npz
  window: 1
model:
  hidden_channels: 128
  use_temporal: false
loss:
  part_weights:
    body: 1.0
    foot: 1.5
    face: 2.0
    left_hand: 2.5
    right_hand: 2.5
  local_weights:
    face: 1.0
    left_hand: 1.0
    right_hand: 1.0
train:
  device: auto
  batch_size: 64
  num_workers: 4
  lr: 0.0003
  weight_decay: 0.01
  epochs: 80
  out_dir: checkpoints/h3wb_hrgcn_t1
```

Create `configs/h3wb_hrgcn_causal_t27.yaml`:

```yaml
data:
  train_cache: data/processed/h3wb_2dto3d_train.npz
  window: 27
model:
  hidden_channels: 128
  use_temporal: true
loss:
  part_weights:
    body: 1.0
    foot: 1.5
    face: 2.0
    left_hand: 2.5
    right_hand: 2.5
  local_weights:
    face: 1.0
    left_hand: 1.0
    right_hand: 1.0
train:
  device: auto
  batch_size: 64
  num_workers: 4
  lr: 0.0003
  weight_decay: 0.01
  epochs: 80
  out_dir: checkpoints/h3wb_hrgcn_causal_t27
```

- [ ] **Step 7: Run engine tests**

Run:

```bash
pytest tests/test_engine_smoke.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit training and evaluation engine**

```bash
git add mypose/engine configs tests/test_engine_smoke.py
git commit -m "feat: add training and evaluation engine"
```

---

### Task 8: Streaming Inference API and Sample Inspection Tool

**Files:**
- Create: `mypose/engine/infer_stream.py`
- Create: `tools/inspect_sample.py`
- Create: `tests/test_infer_stream.py`

**Interfaces:**
- Consumes: `HRGCNLifter`, `load_checkpoint`, `H3WBDataset`
- Produces: `PoseStream(model: HRGCNLifter, device: torch.device)`
- Produces: `PoseStream.reset() -> None`
- Produces: `PoseStream.step(frame_2d) -> np.ndarray`
- Produces: CLI `python tools/inspect_sample.py --cache data/processed/h3wb_2dto3d_train.npz --index 0`

- [ ] **Step 1: Write failing streaming inference test**

Create `tests/test_infer_stream.py`:

```python
import numpy as np
import torch

from mypose.engine.infer_stream import PoseStream
from mypose.models.hrgcn_lifter import HRGCNLifter


def test_pose_stream_accepts_numpy_frame_and_returns_numpy_pose():
    model = HRGCNLifter(hidden_channels=16, use_temporal=True)
    stream = PoseStream(model, device=torch.device("cpu"))
    frame = np.zeros((133, 3), dtype=np.float32)
    pred = stream.step(frame)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (133, 3)


def test_pose_stream_reset_keeps_api_usable():
    model = HRGCNLifter(hidden_channels=16, use_temporal=True)
    stream = PoseStream(model, device=torch.device("cpu"))
    stream.step(np.zeros((133, 3), dtype=np.float32))
    stream.reset()
    pred = stream.step(np.zeros((133, 3), dtype=np.float32))
    assert pred.shape == (133, 3)
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
pytest tests/test_infer_stream.py -v
```

Expected: FAIL with missing streaming module.

- [ ] **Step 3: Implement streaming inference**

Create `mypose/engine/infer_stream.py`:

```python
from __future__ import annotations

import numpy as np
import torch

from mypose.data.keypoints133 import validate_keypoints_shape
from mypose.models.hrgcn_lifter import HRGCNLifter


class PoseStream:
    def __init__(self, model: HRGCNLifter, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.model.reset_stream()

    @torch.no_grad()
    def step(self, frame_2d: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(frame_2d, np.ndarray):
            validate_keypoints_shape(frame_2d, dims=2, name="frame_2d")
            frame = torch.from_numpy(frame_2d).to(device=self.device, dtype=torch.float32)
        else:
            frame = frame_2d.to(device=self.device, dtype=torch.float32)
        pred = self.model.step(frame.unsqueeze(0))
        return pred.squeeze(0).cpu().numpy()
```

- [ ] **Step 4: Implement sample inspection tool**

Create `tools/inspect_sample.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from mypose.data.h3wb import H3WBDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--window", type=int, default=1)
    args = parser.parse_args()
    dataset = H3WBDataset(args.cache, window=args.window)
    sample = dataset[args.index]
    print(f"history_2d: {sample['history_2d'].shape}")
    print(f"target_3d: {sample['target_3d'].shape}")
    print(f"valid target points: {int(sample['target_mask'].sum())}")
    print(f"meta: {sample['meta']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run streaming tests**

Run:

```bash
pytest tests/test_infer_stream.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit streaming inference and inspection**

```bash
git add mypose/engine/infer_stream.py tools/inspect_sample.py tests/test_infer_stream.py
git commit -m "feat: add streaming inference"
```

---

### Task 9: Final Verification, Documentation Check, and Baseline Readiness

**Files:**
- Modify: `docs/datasets.md`
- Create: `README.md`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: setup, preparation, training, evaluation, and inference commands documented in `README.md`.

- [ ] **Step 1: Create README with exact workflow commands**

Create `README.md`:

```markdown
# MyPoseLift

Lightweight 2D-to-3D COCO-WholeBody 133-point pose lifting with H3WB supervision.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Data

```bash
python tools/download_datasets.py --dataset coco-wholebody --root data/raw --with-images
python tools/download_datasets.py --dataset h3wb --root data/raw
python tools/prepare_h3wb.py --annotations data/raw/h3wb/annotations/2Dto3D_train.json --out data/processed/h3wb_2dto3d_train.npz
```

## Train

```bash
python -m mypose.engine.train --config configs/h3wb_hrgcn_t1.yaml
python -m mypose.engine.train --config configs/h3wb_hrgcn_causal_t27.yaml
```

## Inspect

```bash
python tools/inspect_sample.py --cache data/processed/h3wb_2dto3d_train.npz --index 0
```

## Test

```bash
pytest -v
```
```

- [ ] **Step 2: Run the full test suite**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 3: Run import smoke test**

Run:

```bash
python -c "from mypose.models.hrgcn_lifter import HRGCNLifter; HRGCNLifter(hidden_channels=16); print('ok')"
```

Expected: command prints `ok`.

- [ ] **Step 4: Run CPU forward smoke test**

Run:

```bash
python -c "import torch; from mypose.models.hrgcn_lifter import HRGCNLifter; m=HRGCNLifter(hidden_channels=16,use_temporal=True); y=m(torch.zeros(1,3,133,3)); print(tuple(y.shape))"
```

Expected: command prints `(1, 133, 3)`.

- [ ] **Step 5: Confirm no dataset files are staged**

Run:

```bash
git status --short
```

Expected: no staged files under `data/raw`, `data/processed`, or `data/cache`.

- [ ] **Step 6: Commit final docs**

```bash
git add README.md docs/datasets.md
git commit -m "docs: add pose lifter workflow"
```

---

## Self-Review

Spec coverage:

- H3WB 3D supervision: Task 5.
- COCO-WholeBody official 133-point download and parser: Task 4.
- COCO-WholeBody keypoint layout: Task 1.
- Root-relative camera-coordinate target: Task 2 and Task 5.
- Hand/face local losses and metrics: Task 3 and Task 7.
- HR-GCN-inspired framewise model: Task 6.
- Strictly causal temporal adapter and streaming cache: Task 6 and Task 8.
- CPU/GPU training and inference: Task 7 and Task 8.
- Dataset files excluded from git: Task 1 and Task 9.
- Tests for keypoint layout, camera/root transform, losses, causal window, parsers, model forward, and streaming inference: Tasks 1 through 8.

Type consistency:

- Dataset samples use `history_2d`, `target_3d`, `target_mask`, and `meta` throughout.
- Model forward consumes `history_2d` as `(B, T, 133, 3)` and returns `(B, 133, 3)`.
- Streaming consumes a single frame `(133, 3)` and returns `(133, 3)`.
- Metrics and loss functions consume batched `(B, 133, 3)` tensors.
