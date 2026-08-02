"""H36W-backed H3WB dataset wrapper with explicit subject/image filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mmpose.datasets.datasets.body3d.h36m_dataset import Human36mDataset
from mmpose.datasets.datasets.wholebody3d.h3wb_dataset import H36MWholeBodyDataset
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
