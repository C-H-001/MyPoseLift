import numpy as np
import pytest

from tools.rtmw3d_demo import input_kind, project_3d_points


def test_project_3d_points_is_stable_for_empty_and_regular_inputs():
    empty = project_3d_points(np.empty((0, 3), dtype=np.float32), (320, 240))
    assert empty.shape == (0, 2)

    points = np.array([[-1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    projected = project_3d_points(points, (320, 240))
    assert projected.shape == (2, 2)
    assert np.all(projected >= 0)
    assert np.all(projected[:, 0] < 320)
    assert np.all(projected[:, 1] < 240)


def test_input_kind_rejects_missing_path_before_model_loading(tmp_path):
    with pytest.raises(FileNotFoundError, match="input file not found"):
        input_kind(str(tmp_path / "missing.mp4"))
