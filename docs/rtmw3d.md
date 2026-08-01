# RTMW3D Quick Verification

This document describes the smallest reproducible RGB-to-133-3D path in
MyPoseLift. It deliberately keeps MMPose and MMDetection outside this
repository. The local code is only an adapter and demo surface; the official
model, preprocessing, detector, and model configs remain external.

## 1. Runtime Layout

The expected local files are:

```text
MyPoseLift/
  rtmw3d/                         # local adapter and stable result types
  tools/rtmw3d_demo.py            # local image/video/webcam entry point
  external/mmpose/                 # external checkout, ignored by Git
    projects/rtmpose3d/
      configs/rtmw3d-l_8xb64_cocktail14-384x288.py
      demo/rtmdet_m_640-8xb32_coco-person.py
  weights/
    rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth
    rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
```

`external/mmpose` and `weights/` are local runtime assets and are ignored by
Git. They are not copied into MyPoseLift and are not required for unit tests.

## 2. Install External Runtime

Use a fresh Python 3.10+ environment. Install PyTorch first using the command
from the [official PyTorch selector](https://pytorch.org/get-started/locally/).
For CPU-only use, select CPU. For CUDA use, select a PyTorch build compatible
with the installed NVIDIA driver. The PyTorch choice is intentionally separate
from this repository.

Then install the OpenMMLab stack with versions compatible with MMPose v1.3.2:

```bash
python -m pip install -U openmim
mim install "mmengine>=0.10.0,<1.0.0"
mim install "mmcv>=2.0.1,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0"
python -m pip install "mmpose==1.3.2"
python -m pip install -e ".[test,demo]"
```

On the Windows machine used for this workspace, the existing Anaconda base
environment had incompatible Torch/OpenMP DLLs and two OpenCV distributions
claiming the same `cv2` module. The reproducible local fix is an isolated
environment:

```powershell
E:\Anaconda\python.exe -m venv .venv-rtmw3d
.\.venv-rtmw3d\Scripts\python.exe -m pip install -U pip
.\.venv-rtmw3d\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
.\.venv-rtmw3d\Scripts\python.exe -m pip install -e ".[test,demo]"
```

For CUDA, replace the CPU Torch command with the exact command produced by the
PyTorch selector for the installed NVIDIA driver. Install only one of
`opencv-python` or `opencv-python-headless`; this demo uses `opencv-python`.
The isolated environment was verified with Torch `2.7.1+cpu`, NumPy `2.2.6`,
OpenCV `4.13.0`, and all 16 repository tests passing.

MMCV contains compiled operators and must match the installed PyTorch/CUDA
combination. If MIM cannot find a compatible wheel, follow the [official MMCV
installation matrix](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html)
for that exact PyTorch and CUDA build. On CPU, use the CPU PyTorch build and
verify that the selected model does not request a missing GPU-only operator.

The version choice follows the MMPose v1.3.2 installation model: MMPose 1.x,
MMDetection 3.x, and MMCV 2.x. Pinning the upper bounds avoids silently
selecting the incompatible newer MMCV/MMDetection combinations reported by the
OpenMMLab packages.

## 3. External Checkout and Weights

Clone the exact MMPose tag for the configs. From the repository root:

```bash
git clone --branch v1.3.2 --depth 1 \
  https://github.com/open-mmlab/mmpose.git external/mmpose
mkdir -p weights
```

Download the official RTMW3D-L checkpoint:

```text
https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth
```

Download the official RTMDet-m person checkpoint:

```text
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
```

With `curl`:

```bash
curl -L -o weights/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth \
  https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth
curl -L -o weights/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
```

The config paths passed to the local adapter are:

```text
external/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py
external/mmpose/projects/rtmpose3d/demo/rtmdet_m_640-8xb32_coco-person.py
```

The detector config is the official RTMDet person config used by the RTMPose3D
demo. It is not a MyPoseLift config.

## 4. Run Inference

The local demo accepts an input image path, video path, or the literal
`webcam`. The examples below use the local external config and checkpoint
locations explicitly so that a different checkout or weight directory cannot
be selected accidentally.

Image on CPU:

```bash
python tools/rtmw3d_demo.py \
  --input path/to/image.jpg \
  --device cpu \
  --det-config external/mmpose/projects/rtmpose3d/demo/rtmdet_m_640-8xb32_coco-person.py \
  --det-checkpoint weights/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  --pose-config external/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py \
  --pose-checkpoint weights/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth \
  --output-root outputs/image
```

Video on CUDA:

```bash
python tools/rtmw3d_demo.py \
  --input path/to/video.mp4 \
  --device cuda:0 \
  --det-config external/mmpose/projects/rtmpose3d/demo/rtmdet_m_640-8xb32_coco-person.py \
  --det-checkpoint weights/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  --pose-config external/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py \
  --pose-checkpoint weights/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth \
  --output-root outputs/video \
  --benchmark-frames 100
```

Webcam on CPU:

```bash
python tools/rtmw3d_demo.py \
  --input webcam \
  --device cpu \
  --det-config external/mmpose/projects/rtmpose3d/demo/rtmdet_m_640-8xb32_coco-person.py \
  --det-checkpoint weights/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  --pose-config external/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py \
  --pose-checkpoint weights/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth \
  --benchmark-frames 100
```

For headless runs, use the demo's `--no-show` option and an output directory.
The demo's visualizer is not included in model-only latency; drawing and video
encoding are reported separately when available.

## 5. Data and Coordinate Contract

The adapter accepts one OpenCV BGR `numpy.ndarray` per call. It passes that
current BGR frame through the official top-down pipeline: RTMDet selects COCO
`person` boxes, then RTMW3D predicts the 3D whole-body points for each selected
box. The official model name uses 384x288 in height-by-width notation; the
corresponding 2D image tensor in the config is (width=288, height=384).

The stable output has shape `(N, 133, 3)` with float coordinates and optional
per-keypoint scores and `(N, 5)` detector boxes. The 133-point order is the
COCO-WholeBody order:

```text
0:16    body (17 points)
17:22   foot (6 points)
23:90   face (68 points)
91:111  left hand (21 points)
112:132 right hand (21 points)
```

The 3D values are camera-relative model coordinates, not Human3.6M global
world coordinates and not metric depth from a calibrated camera. The official
visualization may negate/reorder axes and rebase the lowest point for display;
that display transform must not be confused with the raw adapter output.

Each frame is processed independently. There is no future-frame look-ahead,
temporal smoothing, or temporal lifting state in this quick verification. This
is necessary for a fair current-frame latency measurement and is different from
the earlier 2D-sequence-to-3D-lifter experiments.

## 6. Benchmark Interpretation

Use `--benchmark-frames` to collect a bounded number of model-only samples.
The summary reports mean latency, p50, p95, and FPS computed as `1000 / mean_ms`.
The reported model latency excludes display drawing, keyboard polling, disk I/O,
and video encoding. End-to-end webcam responsiveness can therefore be slower.

Run CPU and CUDA separately on the target machine. Warm up first, report the
device, PyTorch version, input resolution, detector threshold, number of people,
and whether CUDA synchronization was used. Do not treat the official H3WB MPJPE
or COCO-WholeBody AP as a local accuracy result, and do not claim that CPU
real-time or sub-30 ms GPU performance has been achieved until measured here.

## 7. References

- [Official MMPose v1.3.2 RTMPose3D README](https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/projects/rtmpose3d/README.md)
- [Official RTMPose3D demo](https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/projects/rtmpose3d/demo/body3d_img2pose_demo.py)
- [MMPose v1.3.2 installation guide](https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/docs/en/installation.md)
- [RTMW technical report](https://arxiv.org/abs/2407.08634)
