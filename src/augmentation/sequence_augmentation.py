"""Temporally consistent mock data for VideoPose3D-style pose lifting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Projection = Literal["perspective", "weak"]


@dataclass(frozen=True)
class CameraModel:
    """Simple camera model whose output is normalized to ``[-1, 1]``."""

    focal_length: float
    image_size: tuple[int, int]
    principal_point: tuple[float, float] | None = None
    camera_distance: float = 5.0
    projection: Projection = "perspective"

    def __post_init__(self) -> None:
        if not np.isfinite(self.focal_length) or self.focal_length <= 0:
            raise ValueError("focal_length must be positive")
        if len(self.image_size) != 2 or any(int(value) <= 0 for value in self.image_size):
            raise ValueError("image_size must contain two positive values")
        if not np.isfinite(self.camera_distance) or self.camera_distance <= 0:
            raise ValueError("camera_distance must be positive")
        if self.projection not in ("perspective", "weak"):
            raise ValueError("projection must be 'perspective' or 'weak'")
        if self.principal_point is not None:
            if len(self.principal_point) != 2 or not np.all(
                np.isfinite(self.principal_point)
            ):
                raise ValueError("principal_point must contain two finite values")

    @property
    def resolved_principal_point(self) -> tuple[float, float]:
        if self.principal_point is not None:
            return self.principal_point
        width, height = self.image_size
        return (width / 2.0, height / 2.0)


@dataclass(frozen=True)
class AugmentationConfig:
    """Sampling controls for one augmented motion clip."""

    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_range: tuple[float, float] = (1.0, 1.0)
    residual_rho: float = 0.0
    residual_scale: float = 0.0
    dropout_probability: float = 0.0
    max_dropout_span: int = 1
    root_index: int = 0
    center_frame: int | None = None

    def __post_init__(self) -> None:
        if len(self.rotation_degrees) != 3 or not np.all(
            np.isfinite(self.rotation_degrees)
        ):
            raise ValueError("rotation_degrees must contain three finite values")
        if any(value < 0 for value in self.rotation_degrees):
            raise ValueError("rotation_degrees cannot be negative")
        if len(self.scale_range) != 2 or not np.all(np.isfinite(self.scale_range)):
            raise ValueError("scale_range must contain two finite values")
        lower, upper = self.scale_range
        if lower <= 0 or upper < lower:
            raise ValueError("scale_range must be positive and ordered")
        if not 0 <= self.residual_rho <= 1:
            raise ValueError("residual_rho must be in [0, 1]")
        if self.residual_scale < 0:
            raise ValueError("residual_scale cannot be negative")
        if not 0 <= self.dropout_probability <= 1:
            raise ValueError("dropout_probability must be in [0, 1]")
        if int(self.max_dropout_span) < 1:
            raise ValueError("max_dropout_span must be at least one")
        if int(self.root_index) < 0:
            raise ValueError("root_index cannot be negative")
        if self.center_frame is not None and int(self.center_frame) < 0:
            raise ValueError("center_frame cannot be negative")


@dataclass(frozen=True)
class ResidualBank:
    """Empirical 2D detector residuals in the caller's coordinate system."""

    residuals: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.residuals, dtype=np.float32, copy=True)
        if values.ndim != 3 or values.shape[-1] != 2 or values.shape[0] == 0:
            raise ValueError("residuals must have shape (N, J, 2)")
        if not np.all(np.isfinite(values)):
            raise ValueError("residuals must contain only finite values")
        values.setflags(write=False)
        object.__setattr__(self, "residuals", values)

    @classmethod
    def from_array(cls, residuals: np.ndarray) -> "ResidualBank":
        return cls(residuals)

    @property
    def keypoint_count(self) -> int:
        return int(self.residuals.shape[1])

    def sample(
        self,
        frames: int,
        rng: np.random.Generator | None = None,
        rho: float = 0.0,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Sample residuals with an AR(1)-style temporal correlation."""

        if frames < 1:
            raise ValueError("frames must be positive")
        if not 0 <= rho <= 1:
            raise ValueError("rho must be in [0, 1]")
        if scale < 0:
            raise ValueError("scale cannot be negative")
        generator = np.random.default_rng() if rng is None else rng
        indices = generator.integers(0, len(self.residuals), size=frames)
        fresh = self.residuals[indices].astype(np.float32, copy=True)
        mean = self.residuals.mean(axis=0, dtype=np.float32)
        sampled = np.empty_like(fresh)
        state = fresh[0]
        sampled[0] = state
        innovation_scale = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
        for frame in range(1, frames):
            centered = rho * (state - mean) + innovation_scale * (fresh[frame] - mean)
            state = mean + centered
            sampled[frame] = state
        return mean + float(scale) * (sampled - mean)


@dataclass(frozen=True)
class AugmentedSequence:
    """Generated sample suitable for a VideoPose3D dataset adapter."""

    keypoints_2d: np.ndarray
    clean_keypoints_2d: np.ndarray
    target_3d: np.ndarray
    visibility: np.ndarray
    pose_3d: np.ndarray
    rotation: np.ndarray
    scale: float
    center_frame: int


def _rotation_matrix(angles_degrees: np.ndarray) -> np.ndarray:
    angles = np.deg2rad(angles_degrees)
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def _validate_pose(pose_3d: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_3d, dtype=np.float32)
    if pose.ndim != 3 or pose.shape[-1] != 3 or pose.shape[0] == 0:
        raise ValueError("pose_3d must have shape (T, J, 3)")
    if pose.shape[1] == 0:
        raise ValueError("pose_3d must contain at least one keypoint")
    if not np.all(np.isfinite(pose)):
        raise ValueError("pose_3d must contain only finite values")
    return pose.copy()


def _project(pose_3d: np.ndarray, camera: CameraModel) -> np.ndarray:
    width, height = camera.image_size
    cx, cy = camera.resolved_principal_point
    camera_pose = pose_3d.copy()
    camera_pose[..., 2] += camera.camera_distance
    depth = camera_pose[..., 2]
    if camera.projection == "perspective":
        if np.any(depth <= 0):
            raise ValueError("projection requires positive depth")
        x_pixels = camera.focal_length * camera_pose[..., 0] / depth + cx
        y_pixels = camera.focal_length * camera_pose[..., 1] / depth + cy
    else:
        x_pixels = camera.focal_length * camera_pose[..., 0] / camera.camera_distance + cx
        y_pixels = camera.focal_length * camera_pose[..., 1] / camera.camera_distance + cy
    projected = np.stack(
        (2.0 * x_pixels / width - 1.0, 2.0 * y_pixels / height - 1.0), axis=-1
    )
    return projected.astype(np.float32, copy=False)


def _dropout_mask(
    frames: int,
    keypoints: int,
    probability: float,
    max_span: int,
    rng: np.random.Generator,
) -> np.ndarray:
    visibility = np.ones((frames, keypoints), dtype=bool)
    if probability == 0:
        return visibility
    for joint in range(keypoints):
        frame = 0
        while frame < frames:
            if rng.random() < probability:
                span = int(rng.integers(1, max_span + 1))
                visibility[frame : min(frames, frame + span), joint] = False
                frame += span
            else:
                frame += 1
    return visibility


def augment_sequence(
    pose_3d: np.ndarray,
    camera: CameraModel,
    config: AugmentationConfig | None = None,
    residual_bank: ResidualBank | None = None,
    rng: np.random.Generator | None = None,
) -> AugmentedSequence:
    """Create a temporally consistent 2D/3D training sample."""

    pose = _validate_pose(pose_3d)
    options = AugmentationConfig() if config is None else config
    if options.root_index >= pose.shape[1]:
        raise ValueError("root_index is outside the pose keypoint range")
    center_frame = pose.shape[0] // 2 if options.center_frame is None else options.center_frame
    if center_frame >= pose.shape[0]:
        raise ValueError("center_frame is outside the clip range")
    if residual_bank is not None and residual_bank.keypoint_count != pose.shape[1]:
        raise ValueError("residual bank keypoint count does not match pose_3d")
    if options.residual_scale > 0 and residual_bank is None:
        raise ValueError("residual_bank is required when residual_scale is positive")

    generator = np.random.default_rng() if rng is None else rng
    rotation_limits = np.asarray(options.rotation_degrees, dtype=np.float32)
    angles = generator.uniform(-rotation_limits, rotation_limits)
    rotation = _rotation_matrix(angles)
    scale = float(generator.uniform(*options.scale_range))
    transformed_pose = (pose @ rotation.T) * scale
    clean_2d = _project(transformed_pose, camera)
    keypoints_2d = clean_2d.copy()
    if residual_bank is not None and options.residual_scale > 0:
        keypoints_2d += residual_bank.sample(
            pose.shape[0],
            rng=generator,
            rho=options.residual_rho,
            scale=options.residual_scale,
        )
    visibility = _dropout_mask(
        pose.shape[0],
        pose.shape[1],
        options.dropout_probability,
        options.max_dropout_span,
        generator,
    )
    target = transformed_pose[center_frame].copy()
    target -= target[options.root_index]
    return AugmentedSequence(
        keypoints_2d=keypoints_2d,
        clean_keypoints_2d=clean_2d,
        target_3d=target,
        visibility=visibility,
        pose_3d=transformed_pose,
        rotation=rotation,
        scale=scale,
        center_frame=center_frame,
    )


__all__ = [
    "AugmentedSequence",
    "AugmentationConfig",
    "CameraModel",
    "ResidualBank",
    "augment_sequence",
]
