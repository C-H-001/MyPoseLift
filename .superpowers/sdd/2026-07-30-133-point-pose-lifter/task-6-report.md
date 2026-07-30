# Task 6 Report: Causal Temporal Adapter and HR-GCN-Inspired Lifter

## Status

DONE_WITH_CONCERNS

## Implemented

- Added `CausalTemporalAdapter` with a left-padded depthwise temporal convolution, pointwise projection, residual output, receptive-field property, and streaming cache/reset API.
- Added `GraphBlock` with bidirectional COCO-WholeBody edge aggregation, degree normalization, residual projection, LayerNorm, and GELU activation.
- Added `HRGCNLifter` with optional causal temporal adaptation, three graph blocks, body/fine output heads, framewise inference, and streaming/reset APIs.
- Added the required causal-window and model-forward tests.

## Causality

The temporal adapter applies only left padding. Its output at frame `t` therefore uses the current frame and prior frames within the receptive field. Streaming `step` retains at most the receptive field and returns the current frame output. The lifter selects the final adapted frame for `forward`, and `step` processes one current frame through the same framewise graph lifter.

## Verification

Required command attempted:

```text
pytest tests/test_causal_window.py tests/test_model_forward.py -v
```

Result: blocked during test collection by the known host environment issue. Native Torch import failed with:

```text
OSError: [WinError 1114] ... Error loading "E:\Anaconda\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
```

No test cases executed because both test modules failed while importing `torch`.

Fallback command run:

```text
python -m py_compile mypose\models\temporal_adapter.py mypose\models\hrgcn_lifter.py tests\test_causal_window.py tests\test_model_forward.py
```

Result: exit code `0`.

Additional verification:

- `git diff --check`: exit code `0`.
- `git show --format= --check HEAD`: exit code `0`.
- No dataset files were added or staged.

## Commit

`79010e5 feat: add causal hrgcn lifter`

## Concern

Runtime Torch tests and CPU forward/streaming behavior could not be verified on this host because `torch` cannot load `c10.dll` (`WinError 1114`).

---

## Review Fixes

### Findings addressed

1. `HRGCNLifter` now rejects any `num_keypoints` value other than the project-required 133 with `ValueError("num_keypoints must be 133, got ...")`. This makes the fixed COCO-WholeBody graph edges and fine-keypoint indices explicit and safe.
2. Added lifter-level tests for streaming equivalence and future-frame isolation. The streaming test compares repeated `step` calls with `forward(history)` for the current pose. The causality test changes a later frame and verifies the earlier history output is unchanged.

### Focused tests added

- `test_lifter_rejects_non_133_keypoint_count`
- `test_streaming_lifter_matches_batch_current_pose`
- `test_lifter_output_at_frame_does_not_depend_on_future_frames`

### Verification after fixes

Required test command attempted again:

```text
pytest tests/test_causal_window.py tests/test_model_forward.py -v
```

Result: blocked during collection with the same host issue. Pytest collected 0 items and reported 2 import errors, both caused by:

```text
OSError: [WinError 1114] ... Error loading "E:\Anaconda\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
```

Fallback command:

```text
python -m py_compile mypose\models\hrgcn_lifter.py mypose\models\temporal_adapter.py tests\test_causal_window.py tests\test_model_forward.py
```

Result: exit code `0`; no Python compiler errors or output.

Additional verification: `git diff --check` exited `0`.

---

## Review Fix Round 2

### Finding addressed

Corrected `test_lifter_output_at_frame_does_not_depend_on_future_frames` so it passes both complete five-frame sequences through the `HRGCNLifter` instance's configured temporal adapter. Only frame 4 differs between the sequences, and the test compares adapted outputs for frames 0 through 3. The changed future frame is therefore included in computation but excluded from the assertion, making the test detect right-padded or centered temporal leakage.

### Verification

Required test command attempted:

```text
pytest tests/test_causal_window.py tests/test_model_forward.py -v
```

Result: blocked during collection. Pytest collected 0 items and reported 2 import errors because native Torch failed to load:

```text
OSError: [WinError 1114] ... Error loading "E:\Anaconda\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
```

Fallback command:

```text
python -m py_compile mypose\models\hrgcn_lifter.py mypose\models\temporal_adapter.py tests\test_causal_window.py tests\test_model_forward.py
```

Result: exit code `0`; no Python compiler errors or output.

Additional verification: `git diff --check` exited `0`.
