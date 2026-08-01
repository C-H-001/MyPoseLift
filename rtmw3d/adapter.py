"""Small, lazy adapter around the official MMPose RTMW3D pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlparse

import numpy as np

from .defaults import NUM_KEYPOINTS
from .types import PoseResult, RuntimeConfig, RuntimeDependencyError
from .types import require_runtime_dependency


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_model_reference(value: str, label: str) -> None:
    if _is_url(value):
        return
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {value}")


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _get_field(container: Any, name: str) -> Any:
    if container is None:
        return None
    try:
        return getattr(container, name)
    except AttributeError:
        pass
    if hasattr(container, "get"):
        return container.get(name)
    return None


def _as_rows(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if array is None:
        raise ValueError(f"detector output is missing {name}")
    if array.ndim == 1:
        array = array[None, ...]
    if array.ndim != 2:
        raise ValueError(f"detector {name} must be a 2D array; got {array.shape}")
    return array


class RTMW3DAdapter:
    """Run one BGR frame through RTMDet person detection and RTMW3D lifting.

    MMPose and MMDetection are imported only while constructing the adapter. This
    keeps package import and unit tests usable on machines without the optional
    OpenMMLab runtime.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        detector: Any = None,
        pose_estimator: Any = None,
        inference_detector: Optional[Callable[..., Any]] = None,
        inference_topdown: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config

        # Validate explicitly supplied local references before importing optional
        # runtimes, so a typo is reported as a file error rather than an install
        # error.
        if detector is not None:
            _validate_model_reference(config.detector_config, "detector config")
            _validate_model_reference(config.detector_checkpoint, "detector checkpoint")
        if pose_estimator is not None:
            _validate_model_reference(config.pose_config, "pose config")
            _validate_model_reference(config.pose_checkpoint, "pose checkpoint")

        detector_apis = None
        if detector is None or inference_detector is None:
            detector_apis = require_runtime_dependency(
                "mmdet.apis", "MMDetection RTMW3D runtime"
            )
        if detector is None:
            _validate_model_reference(config.detector_config, "detector config")
            _validate_model_reference(config.detector_checkpoint, "detector checkpoint")
            init_detector = getattr(detector_apis, "init_detector", None)
            if init_detector is None:
                raise RuntimeDependencyError(
                    "MMDetection runtime does not expose mmdet.apis.init_detector"
                )
            detector = init_detector(
                config.detector_config,
                config.detector_checkpoint,
                device=config.device,
            )
        if inference_detector is None:
            inference_detector = getattr(detector_apis, "inference_detector", None)
            if inference_detector is None:
                raise RuntimeDependencyError(
                    "MMDetection runtime does not expose mmdet.apis.inference_detector"
                )

        pose_apis = None
        if pose_estimator is None or inference_topdown is None:
            pose_apis = require_runtime_dependency(
                "mmpose.apis", "MMPose RTMW3D runtime"
            )
        if pose_estimator is None:
            _validate_model_reference(config.pose_config, "pose config")
            _validate_model_reference(config.pose_checkpoint, "pose checkpoint")
            init_model = getattr(pose_apis, "init_model", None)
            if init_model is None:
                raise RuntimeDependencyError(
                    "MMPose runtime does not expose mmpose.apis.init_model"
                )
            pose_estimator = init_model(
                config.pose_config,
                config.pose_checkpoint,
                device=config.device,
            )
        if inference_topdown is None:
            inference_topdown = getattr(pose_apis, "inference_topdown", None)
            if inference_topdown is None:
                raise RuntimeDependencyError(
                    "MMPose runtime does not expose mmpose.apis.inference_topdown"
                )

        self.detector = detector
        self.pose_estimator = pose_estimator
        self._inference_detector = inference_detector
        self._inference_topdown = inference_topdown

    def predict(self, frame_bgr: np.ndarray) -> PoseResult:
        """Predict camera-relative 3D COCO-WholeBody points for one BGR frame."""

        if not isinstance(frame_bgr, np.ndarray):
            raise ValueError("frame must be a BGR image ndarray")
        if (
            frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.shape[0] == 0
            or frame_bgr.shape[1] == 0
        ):
            raise ValueError("frame must be a non-empty BGR image ndarray with 3 channels")

        detection = self._inference_detector(self.detector, frame_bgr)
        person_bboxes = self._select_person_bboxes(detection)
        if person_bboxes.shape[0] == 0:
            return self._empty_result()

        samples = self._inference_topdown(
            self.pose_estimator,
            frame_bgr,
            person_bboxes[:, :4],
            bbox_format="xyxy",
        )
        if samples is None:
            raise ValueError("pose inference returned no PoseDataSample collection")
        return self._convert_pose_samples(samples, person_bboxes)

    def _select_person_bboxes(self, detection: Any) -> np.ndarray:
        instances = _get_field(detection, "pred_instances")
        if instances is None:
            instances = detection
        labels = _to_numpy(_get_field(instances, "labels"))
        bboxes = _as_rows(_get_field(instances, "bboxes"), name="bboxes")
        scores = _to_numpy(_get_field(instances, "scores"))
        if scores is None and bboxes.shape[1] == 5:
            scores = bboxes[:, 4]
            bboxes = bboxes[:, :4]
        if scores is None:
            raise ValueError("detector output is missing scores")
        scores = np.asarray(scores).reshape(-1)
        if labels is None:
            raise ValueError("detector output is missing labels")
        labels = np.asarray(labels).reshape(-1)
        if bboxes.shape[1] != 4:
            raise ValueError(f"detector bboxes must have four coordinates; got {bboxes.shape}")
        if not (len(labels) == len(scores) == len(bboxes)):
            raise ValueError("detector labels, scores, and bboxes have inconsistent lengths")

        keep = (labels == 0) & (scores >= self.config.bbox_thr)
        selected = np.flatnonzero(keep)
        if selected.size == 0:
            return np.empty((0, 5), dtype=np.float32)
        selected = selected[np.argsort(scores[selected])[::-1]]
        selected = selected[: self.config.max_instances]
        return np.concatenate(
            [np.asarray(bboxes[selected], dtype=np.float32), scores[selected, None].astype(np.float32)],
            axis=1,
        )

    def _convert_pose_samples(
        self, samples: Iterable[Any], person_bboxes: np.ndarray
    ) -> PoseResult:
        keypoints: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        bboxes: list[np.ndarray] = []
        for index, sample in enumerate(list(samples)):
            instances = _get_field(sample, "pred_instances")
            if instances is None:
                instances = sample
            points = _get_field(instances, "keypoints_3d")
            if points is None:
                points = _get_field(instances, "keypoints")
            points = _to_numpy(points)
            if points is None:
                raise ValueError("pose output is missing keypoints_3d/keypoints")
            if points.ndim == 2:
                points = points[None, ...]
            if points.ndim != 3 or points.shape[1] != NUM_KEYPOINTS:
                raise ValueError(
                    "pose keypoints must have shape (N, 133, 3); "
                    f"got {points.shape}"
                )
            if points.shape[2] != 3:
                raise ValueError(
                    "pose keypoints must have shape (N, 133, 3); "
                    f"got {points.shape}"
                )

            point_scores = _to_numpy(_get_field(instances, "keypoint_scores"))
            if point_scores is None:
                point_scores = np.ones(points.shape[:2], dtype=np.float32)
            if point_scores.ndim == 1:
                point_scores = point_scores[None, ...]
            if point_scores.shape != points.shape[:2]:
                raise ValueError(
                    "pose keypoint_scores must have shape "
                    f"{points.shape[:2]}; got {point_scores.shape}"
                )

            for person_index in range(points.shape[0]):
                keypoints.append(points[person_index])
                scores.append(point_scores[person_index])
                bbox_index = min(index, len(person_bboxes) - 1)
                bboxes.append(person_bboxes[bbox_index])

        if not keypoints:
            return self._empty_result()
        return PoseResult.from_arrays(
            np.stack(keypoints), scores=np.stack(scores), bboxes=np.stack(bboxes)
        )

    @staticmethod
    def _empty_result() -> PoseResult:
        return PoseResult.from_arrays(
            np.empty((0, NUM_KEYPOINTS, 3), dtype=np.float32),
            scores=np.empty((0, NUM_KEYPOINTS), dtype=np.float32),
            bboxes=np.empty((0, 5), dtype=np.float32),
        )


__all__ = ["RTMW3DAdapter"]
