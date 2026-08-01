from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rtmw3d.adapter import RTMW3DAdapter
from rtmw3d.types import RuntimeConfig, RuntimeDependencyError


def _install_fake_runtime(monkeypatch, *, detector_result, pose_samples):
    calls = {"detector_init": None, "pose_init": None}
    mmdet_apis = ModuleType("mmdet.apis")
    mmpose_apis = ModuleType("mmpose.apis")

    def init_detector(config, checkpoint, device):
        calls["detector_init"] = (config, checkpoint, device)
        return object()

    def init_model(config, checkpoint, device):
        calls["pose_init"] = (config, checkpoint, device)
        return object()

    mmdet_apis.init_detector = init_detector
    mmdet_apis.inference_detector = lambda model, image: detector_result
    mmpose_apis.init_model = init_model
    mmpose_apis.inference_topdown = (
        lambda model, image, bboxes, bbox_format="xyxy": pose_samples
    )

    monkeypatch.setitem(__import__("sys").modules, "mmdet.apis", mmdet_apis)
    monkeypatch.setitem(__import__("sys").modules, "mmpose.apis", mmpose_apis)
    return calls


def _sample(keypoints_3d, scores):
    keypoints_3d = np.asarray(keypoints_3d)
    return SimpleNamespace(
        pred_instances=SimpleNamespace(
            keypoints_3d=keypoints_3d,
            transformed_keypoints=keypoints_3d[..., :2],
            keypoint_scores=np.asarray(scores),
        )
    )


def test_adapter_initializes_optional_runtimes_lazily_and_predicts_133_points(
    monkeypatch, tmp_path
):
    detector_result = SimpleNamespace(
        pred_instances=SimpleNamespace(
            labels=np.array([0, 1, 0]),
            bboxes=np.array([[1, 2, 10, 20], [3, 4, 30, 40], [5, 6, 50, 60]]),
            scores=np.array([0.9, 0.99, 0.4]),
        )
    )
    pose_samples = [
        _sample(np.ones((1, 133, 3)), np.full((1, 133), 0.8)),
        _sample(np.full((1, 133, 3), 2), np.full((1, 133), 0.7)),
    ]
    calls = _install_fake_runtime(
        monkeypatch, detector_result=detector_result, pose_samples=pose_samples
    )
    detector_config = tmp_path / "detector.py"
    pose_config = tmp_path / "pose.py"
    detector_checkpoint = tmp_path / "detector.pth"
    pose_checkpoint = tmp_path / "pose.pth"
    for path in (detector_config, pose_config, detector_checkpoint, pose_checkpoint):
        path.touch()
    config = RuntimeConfig(
        detector_config=str(detector_config),
        detector_checkpoint=str(detector_checkpoint),
        pose_config=str(pose_config),
        pose_checkpoint=str(pose_checkpoint),
        device="cuda:0",
        bbox_thr=0.3,
        max_instances=2,
    )

    adapter = RTMW3DAdapter(config)
    result = adapter.predict(np.zeros((32, 32, 3), dtype=np.uint8))

    assert calls["detector_init"] == (
        str(detector_config),
        str(detector_checkpoint),
        "cuda:0",
    )
    assert calls["pose_init"] == (str(pose_config), str(pose_checkpoint), "cuda:0")
    assert result.keypoints_3d.shape == (2, 133, 3)
    assert result.keypoints_2d.shape == (2, 133, 2)
    assert result.scores.shape == (2, 133)
    assert result.bboxes.shape == (2, 5)
    np.testing.assert_allclose(result.bboxes[:, 4], [0.9, 0.4])


def test_adapter_accepts_keypoints_alias_and_returns_empty_result_when_no_person(
    monkeypatch, tmp_path
):
    detector_result = SimpleNamespace(
        pred_instances=SimpleNamespace(
            labels=np.array([1]),
            bboxes=np.array([[1, 2, 10, 20]]),
            scores=np.array([0.99]),
        )
    )
    calls = _install_fake_runtime(
        monkeypatch, detector_result=detector_result, pose_samples=[]
    )
    paths = []
    for name in ("detector.py", "pose.py", "detector.pth", "pose.pth"):
        path = tmp_path / name
        path.touch()
        paths.append(str(path))
    adapter = RTMW3DAdapter(
        RuntimeConfig(
            detector_config=paths[0],
            pose_config=paths[1],
            detector_checkpoint=paths[2],
            pose_checkpoint=paths[3],
        )
    )
    empty = adapter.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert empty.keypoints_3d.shape == (0, 133, 3)
    assert empty.keypoints_2d.shape == (0, 133, 2)
    assert empty.scores.shape == (0, 133)
    assert empty.bboxes.shape == (0, 5)
    assert calls["pose_init"] is not None


def test_adapter_uses_keypoints_when_keypoints_3d_is_absent(monkeypatch, tmp_path):
    detector_result = SimpleNamespace(
        pred_instances=SimpleNamespace(
            labels=np.array([0]),
            bboxes=np.array([[1, 2, 10, 20]]),
            scores=np.array([0.8]),
        )
    )
    pose_samples = [
        SimpleNamespace(
            pred_instances=SimpleNamespace(
                keypoints=np.zeros((1, 133, 3), dtype=np.float64),
                keypoint_scores=np.ones((1, 133), dtype=np.float64),
            )
        )
    ]
    _install_fake_runtime(
        monkeypatch, detector_result=detector_result, pose_samples=pose_samples
    )
    paths = []
    for name in ("detector.py", "pose.py", "detector.pth", "pose.pth"):
        path = tmp_path / name
        path.touch()
        paths.append(str(path))

    result = RTMW3DAdapter(
        RuntimeConfig(
            detector_config=paths[0],
            pose_config=paths[1],
            detector_checkpoint=paths[2],
            pose_checkpoint=paths[3],
        )
    ).predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert result.keypoints_3d.dtype == np.float32
    np.testing.assert_array_equal(result.scores, np.ones((1, 133), dtype=np.float32))


def test_adapter_reports_missing_dependency_and_local_model_files(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__("sys").modules, "mmdet.apis", None)
    monkeypatch.setitem(__import__("sys").modules, "mmpose.apis", None)
    with pytest.raises(RuntimeDependencyError, match="MMDetection"):
        RTMW3DAdapter(RuntimeConfig())

    with pytest.raises(FileNotFoundError, match="detector config"):
        RTMW3DAdapter(
            RuntimeConfig(
                detector_config=str(tmp_path / "missing.py"),
                detector_checkpoint="https://example.invalid/model.pth",
                pose_config=str(tmp_path / "missing_pose.py"),
                pose_checkpoint="https://example.invalid/pose.pth",
            ),
            detector=object(),
            pose_estimator=object(),
        )


def test_adapter_rejects_invalid_frame_and_malformed_pose_output(
    monkeypatch, tmp_path
):
    detector_result = SimpleNamespace(
        pred_instances=SimpleNamespace(
            labels=np.array([0]),
            bboxes=np.array([[1, 2, 10, 20]]),
            scores=np.array([0.8]),
        )
    )
    _install_fake_runtime(
        monkeypatch,
        detector_result=detector_result,
        pose_samples=[_sample(np.zeros((1, 17, 3)), np.ones((1, 17)))],
    )
    paths = []
    for name in ("detector.py", "pose.py", "detector.pth", "pose.pth"):
        path = tmp_path / name
        path.touch()
        paths.append(str(path))
    adapter = RTMW3DAdapter(
        RuntimeConfig(
            detector_config=paths[0],
            pose_config=paths[1],
            detector_checkpoint=paths[2],
            pose_checkpoint=paths[3],
        )
    )

    with pytest.raises(ValueError, match="BGR image"):
        adapter.predict(np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="133"):
        adapter.predict(np.zeros((16, 16, 3), dtype=np.uint8))


def test_adapter_rejects_2d_keypoints_instead_of_padding_depth(
    monkeypatch, tmp_path
):
    detector_result = SimpleNamespace(
        pred_instances=SimpleNamespace(
            labels=np.array([0]),
            bboxes=np.array([[1, 2, 10, 20]]),
            scores=np.array([0.8]),
        )
    )
    _install_fake_runtime(
        monkeypatch,
        detector_result=detector_result,
        pose_samples=[
            SimpleNamespace(
                pred_instances=SimpleNamespace(
                    keypoints=np.zeros((1, 133, 2), dtype=np.float32),
                    keypoint_scores=np.ones((1, 133), dtype=np.float32),
                )
            )
        ],
    )
    paths = []
    for name in ("detector.py", "pose.py", "detector.pth", "pose.pth"):
        path = tmp_path / name
        path.touch()
        paths.append(str(path))
    adapter = RTMW3DAdapter(
        RuntimeConfig(
            detector_config=paths[0],
            pose_config=paths[1],
            detector_checkpoint=paths[2],
            pose_checkpoint=paths[3],
        )
    )

    with pytest.raises(ValueError, match="133"):
        adapter.predict(np.zeros((16, 16, 3), dtype=np.uint8))
