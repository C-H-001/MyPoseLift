from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlparse

import requests
from tqdm import tqdm

COCO_IMAGE_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}

COCO_ANNOTATION_FILENAMES = (
    "coco_wholebody_train_v1.0.json",
    "coco_wholebody_val_v1.0.json",
)

H3WB_ANNOTATION_FILENAMES = (
    "2Dto3D_train.json",
    "2Dto3D_test_2d.json",
)

COCO_WHOLEBODY_SOURCES = {
    "official_repo": "https://github.com/jin-s13/COCO-WholeBody",
    "official_readme_downloads": "https://github.com/jin-s13/COCO-WholeBody#download",
    "official_opendatalab": "https://opendatalab.com/OpenDataLab/COCO-WholeBody/download",
}

OPENXLAB_COCO_REPO = "OpenDataLab/COCO-WholeBody"

H3WB_SOURCES = {
    "official_repo": "https://github.com/wholebody3d/wholebody3d",
    "official_readme_downloads": "https://github.com/wholebody3d/wholebody3d#download",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists: {dest}")
        return
    partial = dest.with_name(f"{dest.name}.part")
    partial.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", "0"))
            with partial.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _verify_expected_files(
    root: Path,
    filenames: tuple[str, ...],
    sources: dict[str, str],
    dataset: str,
    validator=None,
) -> None:
    missing = []
    invalid = []
    for filename in filenames:
        path = root / filename
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(path)
        elif validator is not None:
            try:
                validator(path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                invalid.append((path, str(error)))
    if invalid:
        details = "; ".join(f"{path}: {error}" for path, error in invalid)
        raise ValueError(f"invalid {dataset} annotation files: {details}")
    if not missing:
        return
    print(f"Missing {dataset} annotation files:")
    for path in missing:
        print(f"- {path}")
    print("Download them from the official sources:")
    for name, url in sources.items():
        print(f"- {name}: {url}")
    raise FileNotFoundError(
        f"missing {dataset} annotation files; place the files above under {root} "
        "or use --no-verify for guided setup"
    )


def _validate_coco_annotation(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list) or not isinstance(payload.get("annotations"), list):
        raise ValueError("expected top-level 'images' and 'annotations' lists")


def _download_coco_annotations_openxlab(root: Path) -> None:
    executable = shutil.which("openxlab")
    command = [
        executable or "openxlab",
        "dataset",
        "get",
        "--dataset-repo",
        OPENXLAB_COCO_REPO,
        "--target-path",
        str(root),
    ]
    if executable is None:
        command_text = " ".join(command)
        raise RuntimeError(
            "OpenXLab CLI is required for active COCO-WholeBody annotation download. "
            "Install it with `pip install openxlab`, run `openxlab login`, then run: "
            f"`{command_text}`"
        )
    subprocess.run(command, check=True)


def download_coco_wholebody(
    root: Path,
    with_images: bool,
    no_verify: bool = False,
    annotation_source: str = "manual",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    annotation_root = root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    if with_images:
        for split, url in COCO_IMAGE_URLS.items():
            download_file(url, root / "images" / Path(urlparse(url).path).name)
    if no_verify:
        print(f"Place train/val annotation JSON files under: {annotation_root}")
        for name, url in COCO_WHOLEBODY_SOURCES.items():
            print(f"- {name}: {url}")
    elif annotation_source == "openxlab":
        _download_coco_annotations_openxlab(root)
        _verify_expected_files(annotation_root, COCO_ANNOTATION_FILENAMES, COCO_WHOLEBODY_SOURCES, "COCO-WholeBody", _validate_coco_annotation)
    elif annotation_source == "manual":
        _verify_expected_files(annotation_root, COCO_ANNOTATION_FILENAMES, COCO_WHOLEBODY_SOURCES, "COCO-WholeBody", _validate_coco_annotation)
    else:
        raise ValueError(f"unknown COCO annotation source: {annotation_source}")


def download_h3wb(root: Path, no_verify: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    annotation_root = root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    if no_verify:
        print(f"Place 2Dto3D_train.json and 2Dto3D_test_2d.json under: {annotation_root}")
        for name, url in H3WB_SOURCES.items():
            print(f"- {name}: {url}")
    else:
        _verify_expected_files(annotation_root, H3WB_ANNOTATION_FILENAMES, H3WB_SOURCES, "H3WB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["coco-wholebody", "h3wb"], required=True)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--with-images", action="store_true")
    parser.add_argument("--annotation-source", choices=["openxlab", "manual"], default=None)
    parser.add_argument("--no-verify", action="store_true", help="create directories and print manual download instructions without checking files")
    args = parser.parse_args()
    if args.dataset == "coco-wholebody":
        annotation_source = args.annotation_source or "openxlab"
        download_coco_wholebody(args.root / "coco-wholebody", args.with_images, args.no_verify, annotation_source)
    else:
        if args.annotation_source == "openxlab":
            parser.error("--annotation-source openxlab is only supported for --dataset coco-wholebody")
        download_h3wb(args.root / "h3wb", args.no_verify)


if __name__ == "__main__":
    main()
