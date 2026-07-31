# MyPoseLift

Lightweight 2D-to-3D COCO-WholeBody 133-point pose lifting with H3WB supervision.

## Setup

Create a Windows virtual environment and install the project in editable mode:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

The editable install is intentional: it keeps the `mypose` package importable
when running the module, training, and inspection commands from this checkout.

## Data

See [docs/datasets.md](docs/datasets.md) for licensing and source details.

The recommended local layout is:

```text
data/
  raw/
    h3wb/
      annotations/
        h3wb_train.npz
    coco-wholebody/
      annotations/
        coco_wholebody_train_v1.0.json
        coco_wholebody_val_v1.0.json
      images/
        train2017.zip
        val2017.zip
  processed/
    h3wb_2dto3d_train_fold0.npz
    h3wb_2dto3d_val_fold0.npz
```

`tools/download_datasets.py --root` controls where raw datasets are checked or
downloaded. Training does not read raw JSON directly; it reads the prepared
`.npz` caches configured by `data.train_cache` and `data.val_cache`.

```bash
python tools/download_datasets.py --dataset coco-wholebody --root data/raw --with-images
python tools/download_datasets.py --dataset h3wb --root data/raw
python tools/prepare_h3wb.py --annotations data/raw/h3wb/annotations/h3wb_train.npz --train-out data/processed/h3wb_2dto3d_train_fold0.npz --val-out data/processed/h3wb_2dto3d_val_fold0.npz --num-folds 5 --val-fold 0
```

Downloaded datasets and generated caches belong under `data/`, which is not
committed to git. H3WB does not publish a validation split, so preparation
uses deterministic sequence-level folds. Report all five held-out folds for
cross-validation results; a single fold is suitable for development only.

The current official H3WB release publishes `h3wb_train.npz`, whose
`train_data` tree contains per-camera `pose_2d` and `camera_3d` arrays. The
preparation tool also keeps compatibility with the older JSON shape where
samples contain:

```text
keypoints_2d
keypoints_3d
```

For temporal training (`window > 1`), every sample must also provide sequence
and frame metadata. The parser accepts explicit `sequence_id` or `video_id`, or
derives a sequence from `subject + action + camera`. It accepts `frame_id`,
`frame_idx`, `frame_index`, or a sortable frame name in `image_path`.

2D pixel normalization requires image dimensions from one of:

```text
image_width + image_height
width + height
image_size
```

If these fields are absent, only recognized Human3.6M camera IDs are accepted.
Bounding boxes are never treated as image dimensions.

To use server paths, write caches wherever you want and point the config at
those files:

```bash
python tools/prepare_h3wb.py --annotations /data/H3WB/annotations/h3wb_train.npz --train-out /data/MyPoseLift/processed/h3wb_train_fold0.npz --val-out /data/MyPoseLift/processed/h3wb_val_fold0.npz --num-folds 5 --val-fold 0
```

```yaml
data:
  train_cache: /data/MyPoseLift/processed/h3wb_train_fold0.npz
  val_cache: /data/MyPoseLift/processed/h3wb_val_fold0.npz
  window: 27
```

## Train

```bash
python -m mypose.engine.train --config configs/h3wb_hrgcn_t1.yaml
python -m mypose.engine.train --config configs/h3wb_hrgcn_causal_t27.yaml
```

Each training epoch evaluates only `data.val_cache`. Training writes
`last.pt` every epoch and updates `best.pt` when validation `MPJPE_whole`
improves. Resume from the epoch after a saved checkpoint with:

```bash
python -m mypose.engine.train --config configs/h3wb_hrgcn_causal_t27.yaml --resume checkpoints/h3wb_hrgcn_causal_t27/last.pt
```

The causal T=27 config uses a strictly left-padded 27-frame temporal kernel,
so its configured receptive field is genuinely 27 frames.

## Evaluate

```bash
python -m mypose.engine.evaluate --config configs/h3wb_hrgcn_t1.yaml --checkpoint <path-to-checkpoint>
```

Standalone evaluation reads `data.val_cache` by default. `--cache` is an
explicit override for smoke tests or another prepared fold.

## Inspect

```bash
python tools/inspect_sample.py --cache data/processed/h3wb_2dto3d_train_fold0.npz --index 0
```

## Inference

Streaming inference is provided by `mypose.engine.infer_stream.PoseStream`.
Pass each `(133, 3)` 2D frame to `PoseStream.step()` in timestamp order; it
returns the current `(133, 3)` root-relative 3D prediction and only retains
causal history.

## Test

```bash
pytest -v
```
