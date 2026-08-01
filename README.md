# MyPoseLift

MyPoseLift is a small, model-agnostic workspace for real-time 3D human pose
experiments. The current quick-verification path uses the official RTMW3D-L
RGB-to-3D model from MMPose. It predicts the 133 COCO-WholeBody keypoints
directly from the current image; it does not train a 2D model or run a temporal
lifting model in this phase.

## Current Scope

- Official RTMW3D-L, 384x288 input, 133 keypoints.
- Single image, video, and webcam inference through a small local adapter.
- CPU and CUDA devices are supported when the external OpenMMLab runtime is
  installed correctly.
- Current-frame-only processing. No future frames, temporal buffer, or
  look-ahead is used.
- H3WB, COCO3D, and old training/data-processing code are not part of this
  quick verification.

The complete setup and command reference is in [docs/rtmw3d.md](docs/rtmw3d.md).

## Quick Install

Create a clean Python 3.10+ environment, install a PyTorch build appropriate
for the target CPU or CUDA device, then install this repository:

```bash
python -m pip install -e ".[test,demo]"
```

Install MMPose, MMDetection, MMCV, and MMEngine using the pinned procedure in
the RTMW3D guide. Do not copy MMPose into this repository. The guide uses an
external `external/mmpose` checkout only for official config files.

Use a clean isolated environment for Windows. The existing Anaconda base
environment is known to contain conflicting Torch/OpenMP DLLs and duplicate
OpenCV distributions; the reproducible commands are in the RTMW3D guide.

## Tests

```bash
python -m pytest -q
```

The tests cover the local contracts and latency aggregation without downloading
model weights. They do not prove model accuracy or device throughput.

## Evidence and Limits

RTMW3D is the official OpenMMLab real-time 3D whole-body model. The official
v1.3.2 model card reports COCO-WholeBody AP 0.678 and H3WB MPJPE 0.056 for
RTMW3D-L. Those are published benchmark values, not measurements from this
workspace. CPU/GPU latency must be measured on the target machine using the
benchmark output; no speed claim is made here before that measurement.

Official references:

- [MMPose v1.3.2 RTMPose3D project](https://github.com/open-mmlab/mmpose/tree/v1.3.2/projects/rtmpose3d)
- [RTMW technical report](https://arxiv.org/abs/2407.08634)
