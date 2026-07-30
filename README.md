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

```bash
python tools/download_datasets.py --dataset coco-wholebody --root data/raw --with-images
python tools/download_datasets.py --dataset h3wb --root data/raw
python tools/prepare_h3wb.py --annotations data/raw/h3wb/annotations/2Dto3D_train.json --out data/processed/h3wb_2dto3d_train.npz
```

Downloaded datasets and generated caches belong under `data/`, which is not
committed to git.

## Train

```bash
python -m mypose.engine.train --config configs/h3wb_hrgcn_t1.yaml
python -m mypose.engine.train --config configs/h3wb_hrgcn_causal_t27.yaml
```

Each training epoch evaluates the training cache and writes metrics and the
latest checkpoint below the configured `out_dir`.

## Inspect

```bash
python tools/inspect_sample.py --cache data/processed/h3wb_2dto3d_train.npz --index 0
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
