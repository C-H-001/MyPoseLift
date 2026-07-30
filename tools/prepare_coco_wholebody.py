from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mypose.data.coco_wholebody import load_coco_wholebody_annotations
from mypose.data.transforms import normalize_2d_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    samples = load_coco_wholebody_annotations(args.annotations)
    inputs = np.stack([normalize_2d_image(s["keypoints_2d"], s["image_size"]) for s in samples])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, inputs_2d=inputs)
    print(f"wrote {args.out} with inputs_2d shape {inputs.shape}")


if __name__ == "__main__":
    main()
