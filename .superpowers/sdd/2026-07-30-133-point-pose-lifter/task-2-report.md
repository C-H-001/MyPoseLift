# Task 2 Report: Coordinate Transforms, Root-Relative Conversion, and Sample Validation

## Scope
Implemented the requested transform, root-relative, and sample validation helpers for 133-point poses, plus focused camera/unit conversion support and tests.

## Files Added
- `mypose/data/transforms.py`
- `mypose/data/validation.py`
- `mypose/utils/__init__.py`
- `mypose/utils/camera.py`
- `tests/test_camera.py`
- `tests/test_transforms_validation.py`

## Step-by-Step Execution
- Added tests from the brief for pelvis root computation, root-relative conversion, meter-to-mm conversion, 2D normalization, causal windowing, and sample validation error paths.
- Implemented `mypose/utils/camera.py` with `meters_to_millimeters`.
- Implemented `mypose/data/transforms.py` with:
  - `normalize_2d_image`
  - `compute_pelvis_root`
  - `make_root_relative`
  - `make_causal_window`
- Implemented `mypose/data/validation.py` with:
  - `_require_finite`
  - `validate_sample`
- Ran the required command:
  - `pytest tests/test_camera.py tests/test_transforms_validation.py -v`

## Test Result
- 7 passed in 0.47s

## Self-Review
- Functions and validation behavior align with the provided Task 2 interfaces and constraints.
- New modules only depend on existing `mypose.data.keypoints133` helpers (`LEFT_HIP`, `RIGHT_HIP`, `NUM_KEYPOINTS`, `validate_keypoints_shape`) as required.
- No dataset files or `data/` content were modified.
- No regression concerns identified in scope.
