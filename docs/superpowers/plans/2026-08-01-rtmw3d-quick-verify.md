# RTMW3D Quick Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a minimal, reproducible RTMW3D-L runtime that supports single-image, video, and webcam RGB-to-133-point 3D inference with CPU/GPU timing.

**Architecture:** MyPoseLift owns a small adapter and CLI. MMPose, MMDetection, and their runtime dependencies remain external and are imported lazily only when the demo runs. The adapter uses the official RTMW3D-L configuration/checkpoint pair and validates the 133-point output, while tests cover defaults, input modes, result conversion, and benchmark statistics without downloading models.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, PyTorch, MMPose/MMDetection only for runtime inference, pytest.

## Global Constraints

- Do not copy MMPose source code into MyPoseLift.
- Do not introduce a training pipeline in this phase.
- Do not alter the existing H3WB/COCO3D data files or download large datasets.
- Use the official RTMW3D-L model/config from the MMPose v1.3.2 project.
- Preserve COCO-WholeBody 133-point ordering and camera-relative 3D output.
- Webcam inference must process current frames only; no future-frame buffering.
- Benchmark timing must separate model inference from optional visualization and report p50/p95 latency and FPS.
- Large weights and generated outputs must remain ignored by Git.

## File Map

- Create `rtmw3d/__init__.py`: public package exports.
- Create `rtmw3d/defaults.py`: official configs, checkpoint URLs, thresholds, and 133-point metadata.
- Create `rtmw3d/types.py`: typed runtime/result configuration dataclasses.
- Create `rtmw3d/adapter.py`: lazy MMPose/MMDetection loader and one-frame RGB-to-3D inference adapter.
- Create `rtmw3d/benchmark.py`: latency statistics and formatting.
- Create `tools/rtmw3d_demo.py`: webcam/video/image CLI with 3D skeleton visualization and timing overlay.
- Create `tests/test_rtmw3d_defaults.py`: default URLs, keypoint metadata, and mode validation.
- Create `tests/test_rtmw3d_benchmark.py`: percentile/FPS behavior.
- Create `tests/test_rtmw3d_adapter.py`: result conversion and dependency error behavior using small local doubles.
- Create `docs/rtmw3d.md`: install, model files, commands, data-flow checks, and limitations.
- Modify `README.md`: quick-start link and commands.
- Modify `.gitignore`: runtime caches, checkpoints, and demo outputs.

### Task 1: Runtime contract and test harness

Write failing tests for default model metadata, supported input modes, result conversion, missing runtime dependency errors, and benchmark statistics. Implement only the dataclasses, constants, validation helpers, and benchmark calculator needed by those tests. Run the focused tests and then the full suite.

### Task 2: Official MMPose adapter

Implement lazy imports so importing `rtmw3d` works without MMPose installed. Build detector and RTMW3D pose estimator from explicit config/checkpoint paths, run top-down inference on one frame, convert official `PoseDataSample` output to a stable NumPy result with shape `(N, 133, 3)`, and preserve scores/bboxes when present. Add clear errors for missing packages, missing files, and unexpected keypoint shapes. Run adapter unit tests without model downloads.

### Task 3: Demo and measurement path

Implement `tools/rtmw3d_demo.py` for `--input webcam`, image, or video. Use the same one-frame adapter for all modes. Render the 2D camera frame and a simple 3D skeleton side by side, display current and aggregate timing, and ensure benchmark measurements exclude OpenCV drawing. Add a `--no-show`/save path for headless smoke tests and a bounded `--benchmark-frames` option.

### Task 4: Documentation and verification

Document the exact MMPose/MMDetection installation choices, official URLs, expected local files, commands for CPU/GPU/webcam/video/image, and the data-flow audit: BGR-to-RGB handling, detector person filtering, bbox propagation, RTMW3D 384x288 input, COCO-WholeBody 133 ordering, root-relative camera coordinates, and no temporal look-ahead. Run unit tests, import smoke tests, CLI help, and a model-backed smoke test when runtime dependencies and weights are available. Report unverified speed claims separately from measured latency.

