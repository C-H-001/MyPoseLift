# Datasets

## COCO-WholeBody

Official repository: https://github.com/jin-s13/COCO-WholeBody

Images are COCO 2017 train/val images from the [COCO 2017 website](http://images.cocodataset.org).
The official README provides annotation downloads through OneDrive Train/Validation, Google Drive Train/Validation, BaiduPan Train&Validation (password `pu6j`), and OpenXLab. The annotations belong to SenseTime Research, are licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/), and are restricted to research and non-commercial use. The images follow Flickr terms. The annotations use the COCO-WholeBody 133 layout: 17 body, 6 foot, 68 face, and 42 hand keypoints.

Commands:

```bash
pip install openxlab
openxlab login
python tools/download_datasets.py --dataset coco-wholebody --root data/raw --with-images
python tools/prepare_coco_wholebody.py --annotations data/raw/coco-wholebody/annotations/coco_wholebody_train_v1.0.json --out data/processed/coco_wholebody_train.npz
```

The preparation command converts the official annotation JSON into the
project's cached 65-keypoint representation. Dense face points are removed;
body indices 0, 1, and 2 retain the nose, left eye, and right eye. Keep downloaded images,
annotations, and generated caches under `data/`; dataset files are not part of
the repository.

The default COCO annotation source actively runs the official OpenDataLab/OpenXLab command:

```bash
openxlab dataset get --dataset-repo OpenDataLab/COCO-WholeBody --target-path data/raw/coco-wholebody
```

The downloader then validates both expected JSON files and their top-level `images` and `annotations` lists. For manual OneDrive, Google Drive, BaiduPan (password `pu6j`), or OpenXLab downloads, use explicit guided verification:

```bash
python tools/download_datasets.py --dataset coco-wholebody --root data/raw --annotation-source manual
```

Use `--no-verify` only to create the directory structure before placing files; it does not claim that annotations are present.

## H3WB

Official repository: https://github.com/wholebody3d/wholebody3d

H3WB extends Human3.6M to 133 whole-body 3D keypoints with the same skeleton
layout as COCO-WholeBody. The current public release provides
`h3wb_train.npz` for train/validation and separate task-specific test files for
leaderboard evaluation. The H3WB project is MIT licensed; images follow the
Human3.6M license.

Expected files for this project:

```text
data/raw/h3wb/annotations/h3wb_train.npz
```

Commands:

```bash
python tools/download_datasets.py --dataset h3wb --root data/raw
python tools/prepare_h3wb.py --annotations data/raw/h3wb/annotations/h3wb_train.npz --train-out data/processed/h3wb_65_train_fold0.npz --val-out data/processed/h3wb_65_val_fold0.npz --num-folds 5 --val-fold 0
```

The downloader actively fetches `h3wb_train.npz` from the official GitHub
release. The preparation command creates the 65-point cache consumed by the training
configurations in `configs/`.

## Causal TCN workflow

```bash
python tools/download_datasets.py --dataset h3wb --root data/raw
python tools/prepare_h3wb.py --annotations data/raw/h3wb/annotations/h3wb_train.npz --train-out data/processed/h3wb_65_train_fold0.npz --val-out data/processed/h3wb_65_val_fold0.npz --num-folds 5 --val-fold 0
python -m mypose.engine.train --config configs/h3wb_tcn_t81.yaml
python -m mypose.engine.evaluate --config configs/h3wb_tcn_t81.yaml --checkpoint checkpoints/h3wb_tcn_t81/best.pt
python tools/plot_prediction.py --config configs/h3wb_tcn_t81.yaml --checkpoint checkpoints/h3wb_tcn_t81/best.pt --index 100 --out reports/h3wb_tcn_t81_fold0_gt_pred.png
```

For the release NPZ, the parser consumes per-camera `pose_2d` and `camera_3d`
arrays from `train_data`. For older JSON annotations, it consumes the official
`keypoints_2d` and `keypoints_3d` fields.
Pixel coordinates are normalized using per-sample `image_width` /
`image_height`, `width` / `height`, or `image_size` metadata. When those are
absent, the four documented Human3.6M camera IDs use their 1000-pixel camera
resolutions (heights 1000 or 1002). Bounding boxes are never treated as image
dimensions, and samples without a valid normalization basis are rejected.

H3WB does not provide an official validation partition and recommends
five-fold cross-validation. The preparation command sorts sequence IDs
deterministically, assigns complete sequences to folds, and writes disjoint
train and validation caches. Run `--val-fold 0` through `4` for a complete
five-fold result. The released test annotations remain reserved for official
test evaluation.
