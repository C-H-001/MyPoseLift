"""Minimal image/video/webcam demo for the official RTMW3D-L checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = None
_CV2_IMPORT_ERROR = None

from rtmw3d import (
    DEFAULT_DETECTOR_CHECKPOINT_URL,
    DEFAULT_DETECTOR_CONFIG,
    DEFAULT_POSE_CHECKPOINT_URL,
    DEFAULT_POSE_CONFIG,
    PoseResult,
    RuntimeConfig,
    RTMW3DAdapter,
    summarize_latencies,
)


BODY_LINKS = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
    (14, 16),
)
FOOT_LINKS = ((17, 18), (18, 19), (20, 21), (21, 22))


def _require_cv2():
    global cv2, _CV2_IMPORT_ERROR
    if cv2 is None and _CV2_IMPORT_ERROR is None:
        try:
            import cv2 as cv

            cv2 = cv
        except Exception as exc:  # pragma: no cover - depends on local wheel ABI
            _CV2_IMPORT_ERROR = exc
    if cv2 is None:
        raise RuntimeError(
            "The RTMW3D demo requires a working opencv-python installation. "
            "The current import failed; reinstall opencv-python with a NumPy "
            "compatible wheel."
        ) from _CV2_IMPORT_ERROR
    return cv2


def input_kind(value: str) -> str:
    """Classify an input without opening a model."""

    if value.lower() == "webcam":
        return "webcam"
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {value}")
    cv = _require_cv2()
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    return "image" if image is not None else "video"


def project_3d_points(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Project camera-relative points to a stable orthographic canvas."""

    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {values.shape}")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("canvas size must be positive")
    if values.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)

    # Keep camera Y as the display vertical axis and rotate X/Z for depth.
    centered = values - np.nanmean(values, axis=0, keepdims=True)
    yaw = np.deg2rad(30.0)
    screen_x = np.cos(yaw) * centered[:, 0] + np.sin(yaw) * centered[:, 2]
    screen_y = centered[:, 1]
    scale = float(np.nanmax(np.abs(np.column_stack((screen_x, screen_y)))))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    screen_x = screen_x / scale
    screen_y = screen_y / scale
    return np.column_stack((width * (0.5 + 0.38 * screen_x), height * (0.5 - 0.38 * screen_y))).astype(np.int32)


def _draw_links(canvas: np.ndarray, points: np.ndarray, links: Iterable[tuple[int, int]], color: tuple[int, int, int]) -> None:
    for left, right in links:
        if left < len(points) and right < len(points):
            cv2.line(canvas, tuple(points[left]), tuple(points[right]), color, 2, cv2.LINE_AA)


def draw_result(canvas: np.ndarray, result: PoseResult, *, kpt_thr: float = 0.2) -> None:
    """Draw 2D keypoints on a BGR canvas; 3D is drawn by draw_3d_panel."""

    for person_index, points_3d in enumerate(result.keypoints_3d):
        if result.keypoints_2d is None:
            points = points_3d[:, :2].astype(np.int32)
        else:
            points = result.keypoints_2d[person_index].astype(np.int32)
        scores = None if result.scores is None else result.scores[person_index]
        _draw_links(canvas, points, BODY_LINKS, (50, 220, 50))
        _draw_links(canvas, points, FOOT_LINKS, (50, 220, 220))
        for keypoint_index, point in enumerate(points):
            if scores is None or scores[keypoint_index] >= kpt_thr:
                cv2.circle(canvas, tuple(point), 2, (0, 180, 255), -1, cv2.LINE_AA)


def draw_3d_panel(panel: np.ndarray, result: PoseResult, *, kpt_thr: float = 0.2) -> None:
    """Draw a compact orthographic 3D skeleton view on a black panel."""

    height, width = panel.shape[:2]
    for person_index, points_3d in enumerate(result.keypoints_3d):
        points = project_3d_points(points_3d, (width, height))
        scores = None if result.scores is None else result.scores[person_index]
        _draw_links(panel, points, BODY_LINKS, (80, 220, 80))
        _draw_links(panel, points, FOOT_LINKS, (80, 220, 220))
        for keypoint_index, point in enumerate(points):
            if scores is None or scores[keypoint_index] >= kpt_thr:
                depth = float(points_3d[keypoint_index, 2])
                intensity = int(np.clip(180 + depth * 0.5, 80, 255))
                cv2.circle(panel, tuple(point), 2, (intensity, 160, 255), -1, cv2.LINE_AA)


def render_frame(frame_bgr: np.ndarray, result: PoseResult, latency_ms: float, aggregate: str) -> np.ndarray:
    """Create the side-by-side 2D and 3D visualization."""

    left = frame_bgr.copy()
    right = np.zeros_like(left)
    draw_result(left, result)
    draw_3d_panel(right, result)
    cv2.putText(left, "2D camera view", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(right, "3D camera-relative view", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(right, f"inference {latency_ms:.1f} ms", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(right, aggregate, (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return np.concatenate((left, right), axis=1)


def _sync(device: str) -> None:
    if not device.lower().startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, OSError):
        pass


def _configure_cpu_threads(device: str, thread_count: int) -> None:
    if not device.lower().startswith("cpu") or thread_count <= 0:
        return
    try:
        import torch

        torch.set_num_threads(thread_count)
        torch.set_num_interop_threads(1)
    except (ImportError, OSError, RuntimeError):
        pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="image/video path or webcam")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="PyTorch intra-op CPU threads; 0 keeps the library default",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="use CUDA automatic mixed precision for lower latency",
    )
    parser.add_argument("--det-config", default=DEFAULT_DETECTOR_CONFIG)
    parser.add_argument("--det-checkpoint", default=DEFAULT_DETECTOR_CHECKPOINT_URL)
    parser.add_argument("--pose-config", default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--pose-checkpoint", default=DEFAULT_POSE_CHECKPOINT_URL)
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--max-instances", type=int, default=1)
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="skip RTMDet and use the whole frame as one person bbox; for one centered person",
    )
    parser.add_argument("--benchmark-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--output-root", default="", help="directory for rendered image/video")
    parser.add_argument("--no-show", action="store_true")
    return parser


def _aggregate(latencies: list[float]) -> str:
    if not latencies:
        return "no timing samples"
    stats = summarize_latencies(latencies)
    return f"p50 {stats.p50_ms:.1f} ms | p95 {stats.p95_ms:.1f} ms | {stats.fps:.1f} FPS"


def _adapter_from_args(args: argparse.Namespace) -> RTMW3DAdapter:
    return RTMW3DAdapter(RuntimeConfig(
        detector_config=args.det_config,
        detector_checkpoint=args.det_checkpoint,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_checkpoint,
        device=args.device,
        bbox_thr=args.bbox_thr,
        max_instances=args.max_instances,
        use_full_frame=args.full_frame,
    ))


def _predict(
    adapter: RTMW3DAdapter,
    frame: np.ndarray,
    device: str,
    amp: bool = False,
) -> tuple[PoseResult, float]:
    _sync(device)
    started = time.perf_counter()
    context = nullcontext()
    if amp and device.lower().startswith("cuda"):
        import torch

        context = torch.autocast(device_type="cuda", dtype=torch.float16)
    with context:
        result = adapter.predict(frame)
    _sync(device)
    return result, (time.perf_counter() - started) * 1000.0


def _run_image(args: argparse.Namespace, adapter: RTMW3DAdapter, latencies: list[float]) -> None:
    cv = _require_cv2()
    frame = cv.imread(args.input, cv.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"unable to read image: {args.input}")
    result, latency = _predict(adapter, frame, args.device, args.amp)
    latencies.append(latency)
    rendered = render_frame(frame, result, latency, _aggregate(latencies))
    if args.output_root:
        output = Path(args.output_root)
        output.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(output / f"{Path(args.input).stem}_rtmw3d.jpg"), rendered)
    if not args.no_show:
        cv.imshow("RTMW3D", rendered)
        cv.waitKey(0)


def _run_stream(args: argparse.Namespace, adapter: RTMW3DAdapter, latencies: list[float], kind: str) -> None:
    cv = _require_cv2()
    capture = cv.VideoCapture(0 if kind == "webcam" else args.input)
    if not capture.isOpened():
        raise ValueError(f"unable to open {kind}: {args.input}")
    writer = None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            result, latency = _predict(adapter, frame, args.device, args.amp)
            if args.benchmark_frames <= 0 or len(latencies) < args.benchmark_frames:
                latencies.append(latency)
            rendered = render_frame(frame, result, latency, _aggregate(latencies))
            if args.output_root:
                output = Path(args.output_root)
                output.mkdir(parents=True, exist_ok=True)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output / ("webcam_rtmw3d.mp4" if kind == "webcam" else f"{Path(args.input).stem}_rtmw3d.mp4")),
                        cv.VideoWriter_fourcc(*"mp4v"),
                        capture.get(cv2.CAP_PROP_FPS) or 30.0,
                        (rendered.shape[1], rendered.shape[0]),
                    )
                writer.write(rendered)
            if not args.no_show:
                cv.imshow("RTMW3D", rendered)
                if cv.waitKey(1) & 0xFF == 27:
                    break
            if args.benchmark_frames > 0 and len(latencies) >= args.benchmark_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv.destroyAllWindows()
        print(f"processed {frame_index} frames; {_aggregate(latencies)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cpu_threads < 0:
        raise ValueError("--cpu-threads must be non-negative")
    _configure_cpu_threads(args.device, args.cpu_threads)
    kind = input_kind(args.input)
    adapter = _adapter_from_args(args)
    latencies: list[float] = []
    if kind == "image":
        _run_image(args, adapter, latencies)
        print(_aggregate(latencies))
    else:
        _run_stream(args, adapter, latencies, kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
