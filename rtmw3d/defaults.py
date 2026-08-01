"""Stable defaults shared by the RTMW3D adapter and command-line tools."""

from typing import Final

NUM_KEYPOINTS: Final = 133
RTMW3D_INPUT_SIZE: Final = (288, 384)  # width, height; filename is height x width

DEFAULT_POSE_CONFIG: Final = (
    "projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py"
)
DEFAULT_POSE_CHECKPOINT_URL: Final = (
    "https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/"
    "rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
)
DEFAULT_DETECTOR_CONFIG: Final = (
    "projects/rtmpose3d/demo/rtmdet_m_640-8xb32_coco-person.py"
)
DEFAULT_DETECTOR_CHECKPOINT_URL: Final = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
    "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
)

SUPPORTED_INPUT_MODES: Final = ("webcam", "image", "video")

_BODY_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
_FOOT_NAMES = (
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)

# COCO-WholeBody stores 68 face landmarks followed by 21 landmarks for each hand.
KEYPOINT_NAMES: Final = (
    _BODY_NAMES
    + _FOOT_NAMES
    + tuple(f"face_{index}" for index in range(68))
    + tuple(f"left_hand_{index}" for index in range(21))
    + tuple(f"right_hand_{index}" for index in range(21))
)

COCO_WHOLEBODY_GROUPS: Final = {
    "body": (0, 17),
    "foot": (17, 23),
    "face": (23, 91),
    "left_hand": (91, 112),
    "right_hand": (112, 133),
}


def validate_input_mode(mode: str) -> str:
    """Return a supported input mode or fail before opening any model/runtime."""

    if mode not in SUPPORTED_INPUT_MODES:
        supported = ", ".join(SUPPORTED_INPUT_MODES)
        raise ValueError(f"Unsupported input mode {mode!r}; expected one of: {supported}")
    return mode
