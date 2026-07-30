# Datasets

## COCO-WholeBody

Official repository: https://github.com/jin-s13/COCO-WholeBody

Images are COCO 2017 train/val images from the [COCO 2017 website](http://images.cocodataset.org).
The official README provides annotation downloads through OneDrive Train/Validation, Google Drive Train/Validation, BaiduPan Train&Validation (password `pu6j`), and OpenXLab. The annotations belong to SenseTime Research, are licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/), and are restricted to research and non-commercial use. The images follow Flickr terms. The annotations use the COCO-WholeBody 133 layout: 17 body, 6 foot, 68 face, and 42 hand keypoints.

Commands:

```bash
python tools/download_datasets.py --dataset coco-wholebody --root data/raw --with-images
python tools/prepare_coco_wholebody.py --annotations data/raw/coco-wholebody/annotations/coco_wholebody_train_v1.0.json --out data/processed/coco_wholebody_train.npz
```

The downloader creates `annotations/`, verifies both expected JSON files, and prints the official manual-download links when files are missing. Use `--no-verify` only to create the directory structure before placing manually downloaded files.

## H3WB

Official repository: https://github.com/wholebody3d/wholebody3d

H3WB extends Human3.6M to 133 whole-body 3D keypoints with the same skeleton layout as COCO-WholeBody. The official README provides Google Drive downloads. `2Dto3D_train.json` is used for train/validation and `2Dto3D_test_2d.json` is used for the test leaderboard. The H3WB project is MIT licensed; images follow the Human3.6M license.

Expected files for this project:

```text
data/raw/h3wb/annotations/2Dto3D_train.json
data/raw/h3wb/annotations/2Dto3D_test_2d.json
```

Commands:

```bash
python tools/download_datasets.py --dataset h3wb --root data/raw
```

The downloader verifies both expected JSON files and prints the official Google Drive link when either is missing. H3WB preparation is introduced by Task 5; `tools/prepare_h3wb.py` is not part of Task 4.
