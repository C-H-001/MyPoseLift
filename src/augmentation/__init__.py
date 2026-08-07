"""数据增强: 时序一致的运动数据增强 (3D -> 投影 2D + 噪声 + dropout)"""
from .sequence_augmentation import (
    AugmentedSequence,
    AugmentationConfig,
    CameraModel,
    ResidualBank,
    augment_sequence,
)

__all__ = [
    "AugmentedSequence",
    "AugmentationConfig",
    "CameraModel",
    "ResidualBank",
    "augment_sequence",
]
