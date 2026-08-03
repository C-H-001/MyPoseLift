"""H36W-backed H3WB dataset wrapper with explicit subject/image filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mmpose.datasets.datasets.body3d.h36m_dataset import Human36mDataset
from mmpose.datasets.datasets.wholebody3d.h3wb_dataset import H36MWholeBodyDataset
from mmpose.datasets.dataset_wrappers import CombinedDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class H36WWholeBodyDataset(H36MWholeBodyDataset):
    """Use selected H36M subjects and skip unavailable extracted frames."""

    def __init__(self, subjects: list[str], strict_images: bool = True, **kwargs):
        self.camera_order_id = ['54138969', '55011271', '58860488', '60457274']
        self.subjects = list(subjects)
        self.strict_images = strict_images
        self._h36w_subset_frac = float(kwargs.pop('subset_frac', 1.0))
        if not 0 < self._h36w_subset_frac <= 1:
            raise ValueError('subset_frac must be in the interval (0, 1]')
        test_mode = kwargs.pop('test_mode', False)
        # Bypass H36MWholeBodyDataset.__init__, which hardcodes S1/S5/S6 or S7.
        Human36mDataset.__init__(self, test_mode=test_mode, **kwargs)
        self.subset_frac = self._h36w_subset_frac

    def _load_annotations(self):
        instance_list, image_list = super()._load_annotations()
        if not self.strict_images:
            return instance_list, image_list

        filtered = []
        for instance in instance_list:
            if all(Path(path).is_file() for path in instance['img_paths']):
                filtered.append(instance)
        removed = len(instance_list) - len(filtered)
        if removed:
            print(
                f'H36WWholeBodyDataset skipped {removed} samples with missing images'
            )
        if self._h36w_subset_frac == 1.0:
            return filtered, image_list

        # H36MWholeBodyDataset builds instances directly and therefore bypasses
        # Human36mDataset.get_sequence_indices(), where subset_frac is normally
        # applied. Keep a deterministic, evenly spaced subset in every video so
        # the reduced experiment still covers all subjects, actions and cameras.
        groups: dict[tuple[str, str], list[dict]] = {}
        for instance in filtered:
            image_path = Path(instance['img_path'])
            group_key = (image_path.parents[2].name, image_path.parent.name)
            groups.setdefault(group_key, []).append(instance)

        selected = []
        for group in groups.values():
            keep = max(1, int(round(len(group) * self._h36w_subset_frac)))
            positions = np.linspace(0, len(group) - 1, keep).round().astype(int)
            selected.extend(group[index] for index in positions)

        print(
            f'H36WWholeBodyDataset subset_frac={self._h36w_subset_frac} '
            f'selected {len(selected)}/{len(filtered)} samples'
        )
        return selected, image_list


@DATASETS.register_module()
class RelativeRatioCombinedDataset(CombinedDataset):
    """Combine datasets with ratios relative to one reference dataset.

    MMPose's ``CombinedDataset.sample_ratio_factor`` scales each dataset by
    its own raw length.  That makes ``0.05`` COCO samples much more frequent
    than its name suggests when COCO is much larger than H3WB.  This wrapper
    keeps the upstream indexing and resampling behavior but interprets each
    ratio as a fraction of the selected reference dataset's effective length.
    """

    def __init__(
        self,
        datasets: list,
        sample_ratio_factor: list[float] | None = None,
        reference_dataset: int = 0,
        **kwargs,
    ):
        super().__init__(
            datasets=datasets,
            sample_ratio_factor=None,
            **kwargs,
        )
        if sample_ratio_factor is None:
            return
        if len(sample_ratio_factor) != len(self.datasets):
            raise ValueError(
                "sample_ratio_factor must match the number of datasets"
            )
        if not 0 <= reference_dataset < len(self.datasets):
            raise ValueError(
                f"reference_dataset out of range: {reference_dataset}"
            )
        if min(sample_ratio_factor) < 0:
            raise ValueError("sample_ratio_factor cannot be negative")

        self.resample = True
        self._lens_ori = list(self._lens)
        reference_len = self._lens_ori[reference_dataset]
        self._lens = [
            round(reference_len * ratio) for ratio in sample_ratio_factor
        ]
        self._len = sum(self._lens)
