# Task 4 Report

## Status

DONE_WITH_CONCERNS

## Implemented

- Added `mypose.data.coco_wholebody` with COCO-WholeBody annotation loading and a dataset wrapper.
- Added robust zero-padding for missing optional face and hand parts so every sample has `(133, 3)` 2D keypoints.
- Added official-source guided download tooling for COCO-WholeBody and H3WB, including optional COCO 2017 image downloads.
- Added the COCO preparation CLI that writes normalized `inputs_2d` arrays to compressed NPZ output.
- Added dataset source, license, expected-file, and command documentation.
- Added the tiny COCO fixture and parser tests.
- Kept generated outputs in the system temp directory; no dataset files were added under `data/`.

## Verification

- `pytest tests/test_coco_wholebody.py -v`: 2 passed.
- `python tools/download_datasets.py --help`: passed; displayed `--dataset {coco-wholebody,h3wb}`.
- `python tools/prepare_coco_wholebody.py --annotations tests/fixtures/coco_wholebody_tiny.json --out <system-temp>/coco_wholebody_tiny.npz`: passed; wrote shape `(1, 133, 3)`.
- `pytest tests/test_keypoints133.py tests/test_transforms_validation.py tests/test_camera.py tests/test_coco_wholebody.py -v`: 19 passed.
- `git diff --check`: passed.
- `pytest -v`: blocked during collection by the environment’s PyTorch DLL initialization, `WinError 1114` loading `E:\Anaconda\Lib\site-packages\torch\lib\c10.dll` from `tests/test_metrics_losses.py`.

## Commit

- `feat: add dataset download and coco parser`
