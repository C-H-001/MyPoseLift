"""Small H3WB pipeline compatibility transforms."""

from __future__ import annotations

from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS

from .target_weights import scale_2d_only_target_weights


@TRANSFORMS.register_module()
class SetCausalTargetIndex(BaseTransform):
    """Add the target index expected by MMPose's KeypointConverter.

    H36MWholeBodyDataset already stores ``lifting_target`` for the causal
    frame, but unlike the base mocap dataset it does not expose ``target_idx``.
    The converter needs that index when it remaps ``keypoints_3d``.
    """

    def transform(self, results: dict) -> dict:
        results.setdefault('target_idx', [-1])
        return results


@TRANSFORMS.register_module()
class SelectTransformedKeypoints(BaseTransform):
    """Apply the same source selection to affine-transformed 2D points."""

    def __init__(self, indices: list[int]):
        self.indices = indices

    def transform(self, results: dict) -> dict:
        if 'transformed_keypoints' in results:
            results['transformed_keypoints'] = results[
                'transformed_keypoints'][:, self.indices]
        return results


@TRANSFORMS.register_module()
class Scale2DOnlyTargetWeights(BaseTransform):
    """Scale XY supervision for samples that have no metric depth labels."""

    def __init__(self, weight: float = 0.25):
        self.weight = weight

    def transform(self, results: dict) -> dict:
        return scale_2d_only_target_weights(results, self.weight)
