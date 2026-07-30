from __future__ import annotations

import argparse
from pathlib import Path

from mypose.data.h3wb import H3WBDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--window", type=int, default=1)
    args = parser.parse_args()
    dataset = H3WBDataset(args.cache, window=args.window)
    sample = dataset[args.index]
    print(f"history_2d: {sample['history_2d'].shape}")
    print(f"target_3d: {sample['target_3d'].shape}")
    print(f"valid target points: {int(sample['target_mask'].sum())}")
    print(f"meta: {sample['meta']}")


if __name__ == "__main__":
    main()
