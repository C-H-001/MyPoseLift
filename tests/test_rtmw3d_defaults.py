import pytest

from rtmw3d.defaults import (
    COCO_WHOLEBODY_GROUPS,
    DEFAULT_DETECTOR_CHECKPOINT_URL,
    DEFAULT_DETECTOR_CONFIG,
    DEFAULT_POSE_CHECKPOINT_URL,
    DEFAULT_POSE_CONFIG,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    SUPPORTED_INPUT_MODES,
    validate_input_mode,
)


def test_official_rtmw3d_defaults_and_133_point_layout():
    assert NUM_KEYPOINTS == 133
    assert len(KEYPOINT_NAMES) == NUM_KEYPOINTS
    assert KEYPOINT_NAMES[:3] == ("nose", "left_eye", "right_eye")
    assert COCO_WHOLEBODY_GROUPS == {
        "body": (0, 17),
        "foot": (17, 23),
        "face": (23, 91),
        "left_hand": (91, 112),
        "right_hand": (112, 133),
    }
    assert DEFAULT_POSE_CONFIG.endswith(
        "rtmw3d-l_8xb64_cocktail14-384x288.py"
    )
    assert DEFAULT_POSE_CHECKPOINT_URL.startswith("https://download.openmmlab.com/")
    assert DEFAULT_DETECTOR_CONFIG.endswith("rtmdet_m_640-8xb32_coco-person.py")
    assert DEFAULT_DETECTOR_CHECKPOINT_URL.startswith("https://download.openmmlab.com/")


def test_input_mode_validation_is_explicit():
    assert SUPPORTED_INPUT_MODES == ("webcam", "image", "video")
    for mode in SUPPORTED_INPUT_MODES:
        assert validate_input_mode(mode) == mode

    with pytest.raises(ValueError, match="Unsupported input mode"):
        validate_input_mode("stream")
