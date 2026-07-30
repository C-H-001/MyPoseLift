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
