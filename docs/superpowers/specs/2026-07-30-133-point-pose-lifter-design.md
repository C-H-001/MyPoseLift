# 133-Point 3D Whole-Body Pose Lifter Design

Date: 2026-07-30

## Goal

Build a lightweight, reproducible 2D-to-3D whole-body pose lifter for 133 keypoints. The model trains on server GPU, supports CPU and GPU inference, and predicts the current frame strictly online from the current and past 2D skeleton sequence only.

The primary target is root-relative camera-coordinate 3D pose with COCO-WholeBody 133 keypoint layout. The first implementation must produce a defensible baseline before pursuing SOTA accuracy.

## Non-Goals

- Do not port MMPose/MMEngine as a framework dependency.
- Do not directly expand VideoPose3D to 133 points as the main method.
- Do not use diffusion, multi-hypothesis sampling, large transformer backbones, or non-causal centered windows.
- Do not predict absolute global translation in the first version.

## Evidence Base

H3WB is the primary 3D supervision source. It expands Human3.6M to the COCO WholeBody 133 keypoint layout and defines 3D whole-body pose lifting tasks.

HR-GCN is the main model reference because it directly targets 2D-to-3D whole-body pose estimation on H3WB. It argues that applying conventional 3D body pose networks directly to whole-body pose is suboptimal because hands and face have much smaller spatial scale than the body, causing fine details to be under-modeled when processed uniformly. HR-GCN uses graph convolution, high-resolution feature fusion, and a fine-grained keypoint prediction module.

VideoPose3D remains useful only as evidence for causal temporal convolution and engineering discipline. It is not sufficient as a 133-point whole-body architecture.

RTMW/RTMW3D is a relevant real-time whole-body reference, especially for dataset unification and deployment constraints, but it is not selected as the core model because it is closer to an image-to-pose detector system than a pure 2D skeleton sequence lifter.

References:

- H3WB: Human3.6M 3D WholeBody Dataset and Benchmark: https://arxiv.org/abs/2211.15692
- H3WB repository: https://github.com/wholebody3d/wholebody3d
- HR-GCN: 2D-3D Whole-body Pose Estimation with High-Resolution Graph Convolutional Network From a Monocular Camera: https://doi.org/10.1109/JSEN.2025.3557770
- HR-GCN public page/PDF: https://researchportal.port.ac.uk/en/publications/hr-gcn-2d-3d-whole-body-pose-estimation-with-high-resolution-grap
- VideoPose3D: https://github.com/facebookresearch/VideoPose3D
- RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation: https://arxiv.org/abs/2407.08634

## Data Sources

Use two official 133-point data sources with distinct roles.

### H3WB

H3WB is the main 3D training and evaluation dataset. It provides 2D/3D pairs in the COCO-WholeBody 133 layout and is the source for root-relative 3D supervision.

The implementation will include an independent parser rather than copying MMPose dataset code. MMPose and H3WB scripts may be consulted for metadata, split definitions, and camera conventions, but every transform must be explicit and tested.

### COCO-WholeBody / COCO3D Official 133-Point Annotations

The official COCO 133-point annotations must be actively downloaded by project tooling. They provide the canonical 133-point keypoint layout, part grouping, visibility semantics, and 2D input distribution. They are not assumed to be a direct replacement for H3WB 3D supervision unless the selected official release contains compatible 3D labels.

The downloader must support COCO 2017 images and official whole-body annotation files. During implementation, the exact official COCO3D/COCO-WholeBody upstream URL and license must be resolved and recorded in `docs/datasets.md`.

## Sample Schema

Each training item is normalized to:

- `history_2d`: shape `(T, 133, 3)`, channels `(x, y, confidence_or_visibility)`.
- `target_3d`: shape `(133, 3)`, current-frame root-relative 3D camera coordinates.
- `target_mask`: shape `(133,)` or `(133, 1)`, valid target keypoints.
- `meta`: subject, action, camera, frame index, image size, bbox or normalization metadata, and 2D source.

Training uses GT 2D initially. Evaluation must support both GT 2D and detector 2D inputs so lifting quality and real-system quality are not conflated.

## Coordinates

2D coordinates use one fixed normalization strategy shared by training, evaluation, and inference. The strategy must be documented and tested. Candidate default: normalize image coordinates to a camera/image-centered range using image size, with confidence preserved as a separate channel.

3D targets are converted to camera coordinates and translated by the pelvis/root joint. Hands and face remain in the same root-relative coordinate system. Local aligned metrics and losses are added for hands and face to avoid hiding local errors behind body-root alignment.

Unit handling is explicit. H3WB/Human3.6M values must be converted consistently to millimeters for metrics.

## Data Validation

Dataset loading must fail fast on:

- Wrong keypoint count.
- Wrong or unknown keypoint order.
- Left/right hand swap.
- Missing camera parameters when camera transforms are requested.
- Frame alignment mismatch between 2D, 3D, subject, action, and camera.
- NaN/Inf coordinates.
- Unexpected unit scale.
- Excessive invalid target mask rate.

Small inspection tooling must render or print a few samples with part-level summary stats before training.

## Model

The selected model is an HR-GCN-inspired causal whole-body lifter.

### Framewise HR-GCN Lifter

The required baseline is a framewise graph model from one frame of 2D 133-point skeleton to one frame of 3D 133-point skeleton. It should follow HR-GCN at the level of architectural principles:

- graph convolution over the 133-point skeleton,
- high-resolution or multi-branch feature fusion,
- explicit fine-grained prediction path for hands and face,
- unified output tensor `(133, 3)`.

The implementation should not blindly copy HR-GCN code. It should be pure PyTorch, small enough to inspect, and adapted to this repository's sample schema.

### Causal Temporal Adapter

A lightweight temporal module can be enabled before or inside the framewise lifter. It must be strictly causal: prediction at frame `t` can use only frames `<= t`.

Candidate implementation:

- depthwise separable causal 1D convolution over time,
- optional dilation,
- residual connection,
- bounded receptive field,
- streaming cache for CPU inference.

The adapter is optional and must be evaluated by ablation:

- `T=1`: framewise HR-GCN-style baseline.
- `T>1`: causal temporal model.

If the temporal adapter improves whole-body MPJPE but hurts hand/face aligned metrics, it is not the default model.

### Part-Aware Heads

The output path separates body, feet, face, left hand, and right hand features before concatenating the final `(133, 3)` output. Hands and face are first-class outputs with independent losses and metrics.

## Loss

Use simple, inspectable losses:

- Whole-body MPJPE-style coordinate loss over valid target points.
- Part losses for body, feet, face, left hand, and right hand.
- Wrist-aligned local hand losses.
- Nose-aligned or head-root-aligned local face loss.
- Lightweight bone or adjacent-edge consistency regularizer.

Part loss weights are explicit config values. The default must compensate for hands and face being small, high-detail regions rather than letting body joints dominate naturally averaged loss.

Low-confidence 2D inputs remain visible to the model as confidence values. Invalid 3D targets are masked out of the loss.

## Metrics

Always report:

- `MPJPE_whole`
- `MPJPE_body`
- `MPJPE_feet`
- `MPJPE_face`
- `MPJPE_face_nose_aligned`
- `MPJPE_left_hand`
- `MPJPE_right_hand`
- `MPJPE_hands_wrist_aligned`

Optional metrics such as P-MPJPE or N-MPJPE are supplemental only. They are not primary acceptance criteria.

## Training

Training is pure PyTorch and config-driven. It supports:

- CPU/GPU device selection,
- server GPU training,
- checkpoint resume,
- best and last checkpoints,
- deterministic seed setup where feasible,
- part-level training logs,
- GT 2D and detector 2D evaluation modes.

The first training target is not SOTA. It is a reproducible baseline with metrics that expose whether hands and face are actually learned.

## Inference

Inference supports:

- batch offline inference for evaluation,
- strict online streaming inference,
- CPU and GPU execution,
- detector adapter input as `(133, 3)`.

The online API should expose:

- `reset_stream()`
- `step(frame_2d)`

The temporal cache must never contain future frames.

## Project Structure

```text
mypose/
  data/
    coco_wholebody.py
    h3wb.py
    keypoints133.py
    transforms.py
    validation.py
  models/
    hrgcn_lifter.py
    temporal_adapter.py
    losses.py
  engine/
    train.py
    evaluate.py
    infer_stream.py
    checkpoint.py
  utils/
    camera.py
    metrics.py
    seed.py
configs/
  h3wb_hrgcn_t1.yaml
  h3wb_hrgcn_causal_t27.yaml
tools/
  download_datasets.py
  prepare_h3wb.py
  prepare_coco_wholebody.py
  inspect_sample.py
tests/
  test_keypoints133.py
  test_camera.py
  test_causal_window.py
  test_losses.py
docs/
  datasets.md
```

## Dataset Download Tooling

`tools/download_datasets.py` must actively handle official downloads where licensing permits:

- `--dataset coco-wholebody`: COCO 2017 images plus official 133-point annotation files.
- `--dataset h3wb`: H3WB annotations and any required metadata; if access requires manual authorization, the script must print exact instructions and verify the expected local files after the user places them.

The repository stores downloader scripts, manifests, and checksums, not large dataset files.

## Tests

Required tests:

- 133-point keypoint order and part index tests.
- Root-relative camera coordinate tests.
- Unit conversion tests.
- Loss weighting tests proving hand/face weights affect gradients.
- Local hand/face alignment metric tests.
- Causal window tests proving frame `t` output cannot depend on `t+1`.
- Dataset parser tests on tiny fixture files.

## Acceptance Criteria

The first implementation is accepted when:

- H3WB and official COCO 133-point data download/preparation paths are documented and runnable where permissions allow.
- The framewise HR-GCN-style model trains and evaluates end to end.
- The causal temporal adapter trains and evaluates as an ablation.
- Metrics are reported by whole body and by part, including local hand/face aligned metrics.
- CPU inference runs with batch size 1.
- Tests cover keypoint layout, root-relative transforms, causal windows, and hand/face losses.

## Open Risks

- The exact official COCO3D/COCO-WholeBody release source must be verified during implementation. If multiple official-looking releases exist, choose the canonical upstream and document the decision.
- HR-GCN supports the 133-point graph/framewise design, but strict causal temporal extension still requires empirical validation.
- H3WB and COCO distributions differ. COCO should improve 2D layout/noise robustness, but it cannot replace true 3D supervision unless compatible 3D labels are confirmed.
- Hand and face quality may remain limited by 2D detector noise, annotation quality, and resolution. The project must expose this through metrics rather than hiding it behind whole-body averages.
