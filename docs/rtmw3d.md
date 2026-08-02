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

For the Windows RTX 4060 environment used for the current verification, the
CUDA runtime was installed with:

```powershell
.\.venv-rtmw3d\Scripts\python.exe -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1+cu126 torchvision==0.22.1+cu126
.\.venv-rtmw3d\Scripts\python.exe -m pip install --force-reinstall --no-deps --extra-index-url https://miropsota.github.io/torch_packages_builder "mmcv==2.2.0+pt2.7.1cu126"
.\.venv-rtmw3d\Scripts\python.exe -m pip install --force-reinstall --no-deps "mmdet==3.3.0"
```

The second wheel is a third-party Windows build with the CUDA operators needed
by RTMDet, including CUDA NMS. With this wheel, the tested MMDetection version
is `3.3.0`; its local package version guard must allow MMCV 2.2.0. This is an
environment workaround, not a source change in MyPoseLift. Always verify the
operator directly before running the detector:

```powershell
$p = Resolve-Path ".\.venv-rtmw3d\Lib\site-packages\mmdet\__init__.py"
(Get-Content $p -Raw).Replace("mmcv_maximum_version = '2.2.0'", "mmcv_maximum_version = '2.3.0'") | Set-Content $p -NoNewline
```

```powershell
.\.venv-rtmw3d\Scripts\python.exe -c "import torch; from mmcv.ops import nms; b=torch.tensor([[0.,0.,10.,10.]],device='cuda'); s=torch.tensor([.9],device='cuda'); print(nms(b,s,.5)[0].device)"
```
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

### Video file to annotated video

This is the file-based video workflow, not webcam capture. The program reads
the input video frame by frame, runs RTMDet person detection followed by the
pretrained RTMW3D pose model, draws the 2D detections on the left and a
camera-relative 3D skeleton projection on the right, and writes one annotated
MP4. The default output name is `<input-stem>_rtmw3d.mp4` under the directory
passed to `--output-root`.

On Windows PowerShell, run this as one line after downloading the two official
weights into `weights\`:

```powershell
python tools\rtmw3d_demo.py --input E:\videos\input.mp4 --device cuda:0 --det-config external\mmpose\projects\rtmpose3d\demo\rtmdet_m_640-8xb32_coco-person.py --det-checkpoint weights\rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth --pose-config external\mmpose\projects\rtmpose3d\configs\rtmw3d-l_8xb64_cocktail14-384x288.py --pose-checkpoint weights\rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth --output-root outputs\video --no-show
```

The resulting file is:

```text
outputs\video\input_rtmw3d.mp4
```

Use `--device cpu` for CPU inference. Use `--amp` with CUDA to benchmark FP16
autocast. Use `--max-instances 1` when only the highest-confidence person is
needed. For a single centered person, `--full-frame` skips RTMDet and uses the
whole image as the box, which is faster but does not handle multiple people or
large background regions reliably.

Do not pass `--benchmark-frames` when the goal is to process the complete
video. That option intentionally stops after the requested number of frames;
it is only for a bounded latency test. Add `--benchmark-frames 100` to process
and save only the first 100 frames while printing p50/p95 latency and FPS.

The output is a visualization video, not a serialized 3D result file. The raw
per-frame model output has shape `(N, 133, 3)` and is camera-relative. If later
applications need the coordinates, the adapter can be called directly and
the returned arrays should be saved separately; the MP4 is intended for visual
inspection.

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

For a single centered person, skip the RTMDet-m detector during a quick CPU
check. This removes the detector cost, but the whole frame becomes the person
box, so it is not a replacement for detector-based multi-person inference:

```bash
python tools/rtmw3d_demo.py --input webcam --device cpu --full-frame --cpu-threads 4 --pose-config external/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py --pose-checkpoint weights/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth --benchmark-frames 10
```

On CPU, `--cpu-threads 4` is a practical laptop starting point; benchmark
`2`, `4`, and the machine's physical-core count instead of assuming one value
is universally fastest. RTMW3D-L itself is still a relatively heavy model, so
this mode does not promise 30 FPS on CPU.

On CUDA, `--amp` enables FP16 autocast for a latency check. Compare it against
the default FP32 path on the target GPU; rerun the benchmark after changing
this flag.

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

## 8. H3WB 68-Point Fine-Tuning and Mixed 2D Training

The fine-tuning config is
`configs/rtmw3d/rtmw3d-m_68_h3wb-256x192.py`. It uses the same RTMW3D-M-sized
whole-body 2D backbone initialization, an M-sized CSPNeXt/PAFPN, and a
68-channel RTMW3D head. The retained order is:

```text
0:22     body + feet
23:25    nose, left eye, right eye (the selected face points)
26:67    left and right hands
```

The other 65 face landmarks are removed. Nose and both eyes remain because
they are H3WB indices 0, 1, and 2. The reduced bone graph is remapped from the
official 133-point graph; it is not a 68-channel output layer with the old
133-point loss left unchanged. BoneLoss is intentionally disabled because the
upstream implementation computes bone lengths after averaging over the batch,
which can cancel per-sample errors. The active loss is the weighted SimCC 3D
classification loss.

### Diverse 2D body data

When available, the training config mixes H3WB 3D data with COCO-WholeBody,
COCO Body, MPII, CrowdPose, PoseTrack18, AIC, and JHMDB. The official source
keypoint orders are converted to the common COCO-133 order, and the common
133-to-68 selection is applied afterward. This prevents each source dataset
from silently using a different body order. 2D-only samples receive zero Z
weights from `SimCC3DLabel`, so they provide image and XY supervision without
inventing depth targets. The auxiliary source ratio is 0.05 per dataset; this
keeps H3WB 3D supervision from being drowned out by COCO-scale collections.

The config checks asset paths at load time and only adds a source when both its
annotation file and image directory exist. Set these roots as needed:

```powershell
$env:COCO_BODY_ROOT='E:\datasets\coco'
$env:COCO_WHOLEBODY_ANN='E:\datasets\coco\annotations\coco_wholebody_train_v1.0.json'
$env:MPII_ROOT='E:\datasets\mmpose-data'
$env:CROWDPOSE_ROOT='E:\datasets\mmpose-data'
$env:POSETRACK_ROOT='E:\datasets\mmpose-data'
$env:AIC_ROOT='E:\datasets\mmpose-data'
$env:JHMDB_ROOT='E:\datasets\mmpose-data'
```

For the first diversity experiment, COCO 2017 is sufficient:

```text
E:\datasets\coco\
  annotations\person_keypoints_train2017.json
  annotations\coco_wholebody_train_v1.0.json   # optional
  train2017\*.jpg
```

The official COCO Body annotation file is `person_keypoints_train2017.json`.
COCO-WholeBody uses `coco_wholebody_train_v1.0.json` with the same COCO 2017
image directory. Do not place the object-detection-only COCO3D JSON files in
this directory: they do not contain the human keypoint fields needed by this
pipeline.

### H3WB annotation preparation

H3WB provides the 2D/3D annotations, but the RGB images must be downloaded
separately from the licensed Human3.6M distribution. The expected extracted
layout is:

```text
<H3WB_ROOT>/
  original/
    S1/Images/Directions.54138969/frame_0005.jpg
    S5/Images/...
    S6/Images/...
    S7/Images/...
  annotation_body3d/h3wb_train_bbox.npz
```

Prepare the MMPose-compatible bbox field from the official H3WB NPZ:

```powershell
python tools/prepare_h3wb_mmpose.py --input data/raw/h3wb/h3wb_train.npz --output data/h36m/annotation_body3d/h3wb_train_bbox.npz
```

For the flat H36W image archives downloaded from Baidu Netdisk, normalize the
JPEG names and create hard links into the MMPose layout:

```powershell
python tools/prepare_h36w_rgb.py --ann data/raw/h3wb/h3wb_train_bbox.npz --images-root E:\BaiduNetdiskDownload\H36Wdataset\extracted --output-root E:\BaiduNetdiskDownload\H36Wdataset\h36m
```

This operation does not duplicate the JPEG payloads. The current archive set
contains S1, S5, and S7, so the training configuration uses S1/S5 for train
and S7 for validation; S6 is not present in the downloaded files. The local
dataset wrapper filters the small number of H3WB samples whose source JPEG is
also absent from the archive.

Before starting a long run, validate every annotation-to-image path:

```powershell
python tools/check_h3wb_assets.py --ann data/raw/h3wb/h3wb_train_bbox.npz --h36m-root E:\BaiduNetdiskDownload\H36Wdataset\h36m --subjects S1 S5 S7 --max-missing 20
```

The current archive set should report `images: 59980/60000 present`; the
dataset wrapper filters the remaining 20 missing samples. If the number is
lower, training is intentionally blocked because the official dataset class
cannot produce valid image tensors without those RGB files.

### Training command

The configuration reads `H3WB_ROOT` and `H3WB_ANN`, so the paths can stay
outside the repository. On Windows, run the following as one command or set
the variables first:

```powershell
$env:H3WB_ROOT='C:\datasets\h36m'; $env:H3WB_ANN='C:\datasets\h36m\annotation_body3d\h3wb_train_bbox.npz'; $env:PYTHONPATH="$(Resolve-Path .);$(Resolve-Path tools\compat);$(Resolve-Path external\mmpose);$(Resolve-Path external\mmpose\projects\rtmpose3d)"; $env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'; .\.venv-rtmw3d\Scripts\python.exe external\mmpose\tools\train.py configs\rtmw3d\rtmw3d-m_68_h3wb-256x192.py --work-dir work_dirs\rtmw3d-m_68_h3wb-256x192
```

Validation runs every epoch. `EarlyStoppingHook` monitors validation MPJPE,
uses a 0.5 mm minimum improvement, and stops after 12 consecutive epochs
without improvement; the hard upper bound is 100 epochs. The best checkpoint
is saved by MPJPE. H3WB coordinates are converted from millimeters to meters
inside the official MMPose dataset class, so the early-stopping threshold is
`0.0005`, not `0.5`.

To wait for a large COCO download and start the mixed run automatically after
asset validation, use the watcher below. It records progress in
`work_dirs\coco_mixed_download_watcher.log` and writes training output under
`work_dirs\rtmw3d-m_68_h3wb-coco-body-256x192_run03_accum4_02`:

```powershell
pwsh -NoProfile -File tools\run_coco_mixed_after_download.ps1 -BitsJobId cbbf49a5-149b-45fd-a57c-8ea19b95dbd8
```
