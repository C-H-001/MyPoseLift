from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mypose.data.h3wb import write_h3wb_cache, write_h3wb_fold_caches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument(
        "--out",
        type=Path,
        help="write one unsplit cache for parser smoke tests",
    )
    outputs.add_argument(
        "--train-out",
        type=Path,
        help="write the training side of a deterministic sequence fold",
    )
    parser.add_argument("--val-out", type=Path)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--val-fold", type=int, default=0)
    args = parser.parse_args()
    if args.out is not None:
        if args.val_out is not None:
            parser.error("--val-out requires --train-out")
        write_h3wb_cache(args.annotations, args.out)
        print(f"wrote {args.out}")
        return
    if args.val_out is None:
        parser.error("--train-out requires --val-out")
    write_h3wb_fold_caches(
        args.annotations,
        args.train_out,
        args.val_out,
        num_folds=args.num_folds,
        val_fold=args.val_fold,
    )
    print(f"wrote train cache {args.train_out}")
    print(f"wrote validation cache {args.val_out}")


if __name__ == "__main__":
    main()
