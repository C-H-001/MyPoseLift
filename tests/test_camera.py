import numpy as np

from mypose.data.transforms import compute_pelvis_root, make_root_relative
from mypose.utils.camera import meters_to_millimeters


def test_compute_pelvis_root_uses_hip_midpoint():
    pose = np.zeros((133, 3), dtype=np.float32)
    pose[11] = [10.0, 2.0, 4.0]
    pose[12] = [14.0, 6.0, 8.0]
    np.testing.assert_allclose(compute_pelvis_root(pose), [12.0, 4.0, 6.0])


def test_make_root_relative_subtracts_pelvis_from_all_points():
    pose = np.zeros((133, 3), dtype=np.float32)
    pose[11] = [2.0, 0.0, 0.0]
    pose[12] = [4.0, 0.0, 0.0]
    pose[0] = [13.0, 5.0, -1.0]
    rel, root = make_root_relative(pose)
    np.testing.assert_allclose(root, [3.0, 0.0, 0.0])
    np.testing.assert_allclose(rel[0], [10.0, 5.0, -1.0])
    np.testing.assert_allclose((rel[11] + rel[12]) / 2.0, [0.0, 0.0, 0.0])


def test_meters_to_millimeters_multiplies_by_1000():
    points = np.array([[1.2, -0.5, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(meters_to_millimeters(points), [[1200.0, -500.0, 0.0]])
