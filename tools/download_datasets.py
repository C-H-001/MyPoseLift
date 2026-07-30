from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

COCO_IMAGE_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}

COCO_WHOLEBODY_SOURCES = {
    "official_repo": "https://github.com/jin-s13/COCO-WholeBody",
    "official_readme_downloads": "https://github.com/jin-s13/COCO-WholeBody#download",
}

H3WB_SOURCES = {
    "official_repo": "https://github.com/wholebody3d/wholebody3d",
    "official_readme_downloads": "https://github.com/wholebody3d/wholebody3d#download",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists: {dest}")
        return
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        with dest.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))


def download_coco_wholebody(root: Path, with_images: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if with_images:
        for split, url in COCO_IMAGE_URLS.items():
            download_file(url, root / "images" / Path(urlparse(url).path).name)
    print("COCO-WholeBody annotations are hosted by the official repository via OneDrive, Google Drive, BaiduPan, and OpenXLab.")
    print(f"Open: {COCO_WHOLEBODY_SOURCES['official_readme_downloads']}")
    print(f"Place train/val annotation JSON files under: {root / 'annotations'}")


def download_h3wb(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    print("H3WB annotations and test sets are hosted by the official repository via Google Drive.")
    print(f"Open: {H3WB_SOURCES['official_readme_downloads']}")
    print(f"Place 2Dto3D_train.json and available test JSON files under: {root / 'annotations'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["coco-wholebody", "h3wb"], required=True)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--with-images", action="store_true")
    args = parser.parse_args()
    if args.dataset == "coco-wholebody":
        download_coco_wholebody(args.root / "coco-wholebody", args.with_images)
    else:
        download_h3wb(args.root / "h3wb")


if __name__ == "__main__":
    main()
