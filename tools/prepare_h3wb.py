from __future__ import annotations

import argparse
from pathlib import Path

from mypose.data.h3wb import write_h3wb_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_h3wb_cache(args.annotations, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
