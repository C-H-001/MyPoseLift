from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from mypose.data.keypoints65 import COCO65_EDGES


def _device_from_config(cfg: dict) -> torch.device:
    import torch

    requested = cfg["train"]["device"]
    selected = (
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested == "auto"
        else requested
    )
    return torch.device(selected)


def _plot_pose(axis, pose, color: str, label: str) -> None:
    axis.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=10, color=color, label=label)
    for start, end in COCO65_EDGES:
        axis.plot(
            pose[[start, end], 0], pose[[start, end], 1], pose[[start, end], 2],
            color=color, linewidth=0.8, alpha=0.8,
        )


def plot_prediction(
    config_path: Path,
    checkpoint_path: Path,
    cache_path: Path | None,
    index: int,
    output_path: Path,
) -> None:
    import torch

    from mypose.data.h3wb import H3WBDataset
    from mypose.engine import build_model_from_config
    from mypose.engine.checkpoint import load_checkpoint

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = _device_from_config(cfg)
    cache = cache_path or Path(cfg["data"]["val_cache"])
    dataset = H3WBDataset(cache, window=int(cfg["data"]["window"]))
    sample = dataset[index]
    model = build_model_from_config(cfg).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    history = torch.from_numpy(sample["history_2d"]).unsqueeze(0).to(device=device)
    with torch.no_grad():
        prediction = model(history)[0].cpu().numpy()

    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 6))
    axes = [figure.add_subplot(1, 2, position, projection="3d") for position in (1, 2)]
    _plot_pose(axes[0], sample["target_3d"], "tab:blue", "ground truth")
    _plot_pose(axes[1], prediction, "tab:orange", "prediction")
    for axis, title in zip(axes, ("Ground truth", "Prediction")):
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.legend()
    figure.suptitle(f"65-point pose prediction (sample {index})")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a 65-point H3WB prediction")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/gt_pred.png"))
    args = parser.parse_args()
    plot_prediction(args.config, args.checkpoint, args.cache, args.index, args.out)


if __name__ == "__main__":
    main()
