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

## Review Fixes

### Findings addressed

1. COCO-WholeBody and H3WB download flows now create `annotations/`, verify every expected non-empty annotation file, print the official repository/download anchors and missing paths, and raise `FileNotFoundError` unless `--no-verify` is explicitly supplied. Direct cloud-hosted annotation downloads remain guided/manual because the official sources use OneDrive, Google Drive, BaiduPan, or OpenXLab flows.
2. Removed the Task 5-only `tools/prepare_h3wb.py` command from the Task 4 workflow and documented that H3WB preparation is introduced by Task 5.
3. `download_file` streams into a `.part` file, atomically replaces the destination only after completion, and removes partial output on failure.
4. Documented the controller-verified COCO-WholeBody sources, OneDrive/Google Drive/BaiduPan/OpenXLab locations, BaiduPan password `pu6j`, SenseTime Research ownership, CC BY 4.0 annotation license, research/non-commercial restriction, and Flickr image terms. H3WB Google Drive, split filenames, MIT project license, and Human3.6M image license are also documented.

### Additional verification

- `pytest tests/test_coco_wholebody.py -v`: 5 passed.
- `pytest tests/test_keypoints133.py tests/test_transforms_validation.py tests/test_camera.py tests/test_coco_wholebody.py -v`: 22 passed.
- `python tools/download_datasets.py --help`: passed; displayed `--no-verify` and the required dataset choices.
- Offline smoke test for `download_coco_wholebody(..., no_verify=True)`: passed; created the annotation directory and printed official anchors.
- `git diff --check`: passed.
- `pytest -v`: still blocked during collection by PyTorch `WinError 1114` loading `E:\Anaconda\Lib\site-packages\torch\lib\c10.dll`; no Task 4 test failure was reached.

## Review Fix Round 2

### Finding addressed

COCO-WholeBody now has an active OpenXLab/OpenDataLab annotation path. The CLI defaults to `--annotation-source openxlab` for COCO and runs:

`openxlab dataset get --dataset-repo OpenDataLab/COCO-WholeBody --target-path <root>`

The implementation checks for the executable, verifies both downloaded COCO JSON files and their top-level `images` and `annotations` lists, and raises a setup error containing `pip install openxlab`, `openxlab login`, and the exact command when the CLI is missing. `--annotation-source manual` remains available for OneDrive, Google Drive, BaiduPan, and manual OpenXLab downloads. H3WB remains explicit manual verification.

### Additional verification

- `pytest tests/test_coco_wholebody.py -v`: 7 passed, including OpenXLab command construction and missing-CLI error tests without network access.
- `pytest tests/test_keypoints133.py tests/test_transforms_validation.py tests/test_camera.py tests/test_coco_wholebody.py -v`: 24 passed.
- `python tools/download_datasets.py --help`: passed; displayed `--annotation-source {openxlab,manual}`.
- `git diff --check`: passed.
- `pytest -v`: blocked during collection by the environment’s PyTorch `WinError 1114` loading `E:\Anaconda\Lib\site-packages\torch\lib\c10.dll`.
