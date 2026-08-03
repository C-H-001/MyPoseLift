"""Validation-time 3D pose visualization and per-sample error analysis."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _first_instance(value: Any, dimensions: int) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == dimensions + 1:
        array = array[0]
    if array.ndim != dimensions:
        raise ValueError(f"expected {dimensions}D pose data, got {array.shape}")
    return array.astype(np.float32, copy=False)


def _procrustes_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_mean = pred.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    pred_norm = np.linalg.norm(pred_centered)
    if pred_norm < 1e-8:
        return np.repeat(target_mean, len(pred), axis=0)

    covariance = pred_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = np.sum((pred_centered @ rotation) * target_centered)
    scale /= np.sum(pred_centered**2)
    return scale * (pred_centered @ rotation) + target_mean


def compute_sample_metrics(
    pred: np.ndarray, target: np.ndarray, visible: np.ndarray | None = None
) -> dict[str, float]:
    """Compute MPJPE and Procrustes MPJPE in millimeters."""

    pred = _first_instance(pred, 2)
    target = _first_instance(target, 2)
    if pred.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} vs {target.shape}")
    if visible is None:
        mask = np.ones(pred.shape[0], dtype=bool)
    else:
        mask = _as_numpy(visible).reshape(-1) > 0
        if len(mask) != len(pred):
            raise ValueError(f"visibility shape mismatch: {mask.shape} vs {pred.shape}")
    if not np.any(mask):
        return {"mpjpe_mm": float("nan"), "p_mpjpe_mm": float("nan"), "valid_keypoints": 0}

    pred_valid = pred[mask]
    target_valid = target[mask]
    mpjpe = np.linalg.norm(pred_valid - target_valid, axis=-1).mean()
    aligned = _procrustes_align(pred_valid, target_valid)
    p_mpjpe = np.linalg.norm(aligned - target_valid, axis=-1).mean()
    return {
        "mpjpe_mm": float(mpjpe * 1000.0),
        "p_mpjpe_mm": float(p_mpjpe * 1000.0),
        "valid_keypoints": int(mask.sum()),
    }


def _plot_skeleton(ax, points: np.ndarray, parents: Sequence[int], color: str, label: str):
    for child, parent in enumerate(parents):
        if child == parent:
            continue
        ax.plot(
            [points[child, 0], points[parent, 0]],
            [points[child, 1], points[parent, 1]],
            [points[child, 2], points[parent, 2]],
            color=color,
            linewidth=1.2,
            alpha=0.8,
        )
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8, color=color, label=label)


def _draw_sample(
    sample: dict[str, Any], output_path: Path, parents: Sequence[int], epoch: int
) -> None:
    import matplotlib.pyplot as plt

    target = sample["target"]
    pred = sample["pred"]
    keypoints_2d = sample["keypoints_2d"]
    fig = plt.figure(figsize=(14, 9))
    ax_2d = fig.add_subplot(221)
    ax_gt = fig.add_subplot(222, projection="3d")
    ax_pred = fig.add_subplot(223, projection="3d")
    ax_overlay = fig.add_subplot(224, projection="3d")

    ax_2d.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], s=10, color="#1f77b4")
    for child, parent in enumerate(parents):
        if child != parent:
            ax_2d.plot(
                [keypoints_2d[child, 0], keypoints_2d[parent, 0]],
                [keypoints_2d[child, 1], keypoints_2d[parent, 1]],
                color="#1f77b4",
                linewidth=1.0,
            )
    ax_2d.invert_yaxis()
    ax_2d.set_title("2D validation input")
    ax_2d.set_aspect("equal", adjustable="datalim")

    _plot_skeleton(ax_gt, target, parents, "#2ca02c", "GT")
    ax_gt.set_title("GT 3D")
    _plot_skeleton(ax_pred, pred, parents, "#d62728", "Pred")
    ax_pred.set_title("Pred 3D")
    _plot_skeleton(ax_overlay, target, parents, "#2ca02c", "GT")
    _plot_skeleton(ax_overlay, pred, parents, "#d62728", "Pred")
    ax_overlay.set_title("GT vs Pred")
    for axis in (ax_gt, ax_pred, ax_overlay):
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.legend(loc="upper right")

    metrics = sample["metrics"]
    fig.suptitle(
        f"epoch {epoch} | sample {sample['index']} | "
        f"MPJPE {metrics['mpjpe_mm']:.1f} mm | "
        f"P-MPJPE {metrics['p_mpjpe_mm']:.1f} mm"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _write_html(output_dir: Path, epoch: int, samples: list[dict[str, Any]]) -> None:
    cards = []
    for sample in samples:
        metrics = sample["metrics"]
        image = html.escape(sample["image"])
        cards.append(
            "<article>"
            f"<h2>Sample {sample['index']}</h2>"
            f"<p>MPJPE: {metrics['mpjpe_mm']:.2f} mm; "
            f"P-MPJPE: {metrics['p_mpjpe_mm']:.2f} mm; "
            f"valid joints: {metrics['valid_keypoints']}</p>"
            f'<img src="{image}" alt="validation sample {sample["index"]}">'
            f"<p>{html.escape(sample.get('img_path', ''))}</p></article>"
        )
    document = (
        "<!doctype html><meta charset='utf-8'><title>RTMW3D validation</title>"
        "<style>body{font-family:Arial;margin:24px;background:#f4f4f4}"
        "article{background:white;padding:16px;margin:16px 0}"
        "img{max-width:100%;height:auto}</style>"
        f"<h1>RTMW3D validation epoch {epoch}</h1>" + "".join(cards)
    )
    (output_dir / f"epoch_{epoch:03d}.html").write_text(document, encoding="utf-8")


def _sample_from_data(data_sample: Any, output: Any, index: int) -> dict[str, Any]:
    gt_instances = data_sample.gt_instances
    pred_instances = output.pred_instances
    target = _first_instance(gt_instances.lifting_target, 2)
    pred = _first_instance(pred_instances.keypoints, 2)
    visible = getattr(gt_instances, "lifting_target_visible", None)
    keypoints_2d = _first_instance(gt_instances.keypoints, 2)
    metrics = compute_sample_metrics(pred, target, visible)
    return {
        "index": index,
        "img_path": str(data_sample.metainfo.get("img_path", "")),
        "target": target - target.mean(axis=0, keepdims=True),
        "pred": pred - pred.mean(axis=0, keepdims=True),
        "keypoints_2d": keypoints_2d,
        "metrics": metrics,
    }


try:
    from mmengine.hooks import Hook
    from mmpose.registry import HOOKS

    @HOOKS.register_module()
    class ValidationSampleVisualizationHook(Hook):
        """Save three deterministic validation samples after every val epoch."""

        def __init__(self, joint_parents: Sequence[int], num_samples: int = 3,
                     out_dir: str = "val_visualizations"):
            self.joint_parents = list(joint_parents)
            self.num_samples = num_samples
            self.out_dir = out_dir
            self.samples: list[dict[str, Any]] = []

        def before_val_epoch(self, runner) -> None:
            self.samples = []

        def after_val_iter(self, runner, batch_idx, data_batch, outputs) -> None:
            remaining = self.num_samples - len(self.samples)
            if remaining <= 0:
                return
            for data_sample, output in zip(
                data_batch["data_samples"][:remaining], outputs[:remaining]
            ):
                self.samples.append(
                    _sample_from_data(data_sample, output, len(self.samples))
                )

        def after_val_epoch(self, runner, metrics=None) -> None:
            if not self.samples:
                return
            output_dir = Path(runner.work_dir) / runner.timestamp / self.out_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            epoch = int(runner.epoch) + 1
            serializable = []
            for sample in self.samples:
                image_name = f"epoch_{epoch:03d}_sample_{sample['index']:02d}.png"
                _draw_sample(sample, output_dir / image_name,
                             self.joint_parents, epoch)
                serializable.append({
                    "index": sample["index"],
                    "img_path": sample["img_path"],
                    "metrics": sample["metrics"],
                    "image": image_name,
                })
            summary = {
                "epoch": epoch,
                "samples": serializable,
                "mean_mpjpe_mm": float(np.mean([
                    item["metrics"]["mpjpe_mm"] for item in serializable
                ])),
                "mean_p_mpjpe_mm": float(np.mean([
                    item["metrics"]["p_mpjpe_mm"] for item in serializable
                ])),
            }
            (output_dir / f"epoch_{epoch:03d}.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            _write_html(output_dir, epoch, serializable)
            runner.logger.info(
                "Saved %d validation visualizations to %s; mean MPJPE %.2f mm",
                len(serializable), output_dir, summary["mean_mpjpe_mm"]
            )
except ImportError:
    # The pure metric helpers remain usable in the lightweight test install.
    pass

