"""RTMW3D-M-sized 68-point RGB model fine-tuning on H3WB."""

os = __import__('os')


_base_ = [
    '../../external/mmpose/projects/rtmpose3d/configs/'
    'rtmw3d-l_8xb64_cocktail14-384x288.py'
]

custom_imports = dict(
    imports=[
        'rtmpose3d',
        'rtmw3d.h3wb_transforms',
        'rtmw3d.h36w_dataset',
        'rtmw3d.validation_visualization',
    ],
    allow_failed_imports=False,
)

num_keypoints = 68
input_size = (192, 256, 288)  # width, height, depth
auxiliary_sample_ratio = 0.05
auxiliary_2d_loss_weight = 0.25
# H3WB indices 0, 1, 2 are nose and the two eyes. The 23-point body/foot
# prefix already contains them; the explicit duplicate channels preserve the
# requested 68-channel head while ensuring no other face landmark is trained.
face_indices = [0, 1, 2]
keep_indices = list(range(23)) + face_indices + list(range(91, 133))
mapping = [(source, target) for target, source in enumerate(keep_indices)]

# Kept for layout validation. BoneLoss is disabled below because its upstream
# implementation matches batch-mean bone lengths rather than per-sample ones.
full_parents = _base_.model['head']['loss'][1]['joint_parents']
source_to_target = {}
for target, source in enumerate(keep_indices):
    source_to_target.setdefault(source, target)
joint_parents = [
    target if 23 <= target < 23 + len(face_indices)
    else source_to_target[full_parents[source]]
    for target, source in enumerate(keep_indices)
]

codec = dict(
    type='SimCC3DLabel',
    input_size=input_size,
    sigma=(4., 4.9, 6.),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
    root_index=(11, 12),
)

model = dict(
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth',
        ),
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[192, 384, 768],
        out_channels=None,
        out_indices=(1, 2),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    head=dict(
        type='RTMW3DHead',
        in_channels=768,
        out_channels=num_keypoints,
        input_size=input_size,
        in_featuremap_size=tuple(s // 32 for s in input_size),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.1,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=[
            dict(
                type='KLDiscretLossWithWeight',
                use_target_weight=True,
                beta=10.,
                label_softmax=True,
            ),
        ],
        decoder=codec,
    ),
    test_cfg=dict(flip_test=False),
)

reduce_keypoints = [
    dict(type='SetCausalTargetIndex'),
    dict(type='KeypointConverter', num_keypoints=num_keypoints, mapping=mapping),
    dict(type='SelectTransformedKeypoints', indices=keep_indices),
]
train_pipeline = [
    *_base_.train_pipeline[:-2],
    *reduce_keypoints,
    dict(type='GenerateTarget', encoder=codec),
    dict(type='Scale2DOnlyTargetWeights', weight=auxiliary_2d_loss_weight),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    *_base_.val_pipeline[:-2],
    *reduce_keypoints,
    dict(type='GenerateTarget', encoder=codec),
    dict(type='Scale2DOnlyTargetWeights', weight=auxiliary_2d_loss_weight),
    dict(type='PackPoseInputs'),
]
# The inherited pipeline stores the L-model resolution literally in its
# TopdownAffine transform; replace it so the M model really receives 256x192.
train_pipeline[5] = dict(type='TopdownAffine', input_size=input_size[:2])
val_pipeline[2] = dict(type='TopdownAffine', input_size=input_size[:2])

h3wb_root = os.environ.get('H3WB_ROOT', 'data/h36m')
h3wb_ann = os.environ.get(
    'H3WB_ANN', os.path.join(h3wb_root, 'annotation_body3d', 'h3wb_train_bbox.npz')
)

# Body-only datasets are converted to the common 133-keypoint layout first.
# The outer pipeline then selects the current 68-point target.  For 2D-only
# datasets SimCC3DLabel sets the z-axis weight to zero, so they supervise only
# the visible 2D body coordinates and never create fake 3D labels.  These
# mappings are the ones used by the official RTMW3D cocktail configuration.
coco_body_root = os.environ.get('COCO_BODY_ROOT', 'data/coco')
coco_wholebody_ann = os.environ.get(
    'COCO_WHOLEBODY_ANN',
    os.path.join(coco_body_root, 'annotations',
                 'coco_wholebody_train_v1.0.json'))

aic_coco133 = [(0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9),
               (6, 12), (7, 14), (8, 16), (9, 11), (10, 13), (11, 15)]
crowdpose_coco133 = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
                     (6, 11), (7, 12), (8, 13), (9, 14), (10, 15), (11, 16)]
mpii_coco133 = [
    (0, 16), (1, 14), (2, 12), (3, 11), (4, 13), (5, 15), (10, 10),
    (11, 8), (12, 6), (13, 5), (14, 7), (15, 9)
]
jhmdb_coco133 = [
    (3, 6), (4, 5), (5, 12), (6, 11), (7, 8), (8, 7), (9, 14),
    (10, 13), (11, 10), (12, 9), (13, 16), (14, 15)
]
posetrack_coco133 = [(0, 0)] + [(index, index) for index in range(3, 17)]
coco_body_to_coco133 = [(index, index) for index in range(17)]


def _ready_dataset(name, dataset, required_paths):
    """Include an auxiliary dataset only when all assets are present."""
    if all(os.path.exists(path) for path in required_paths):
        auxiliary_body_dataset_names.append(name)
        return dataset
    return None


def _make_body_dataset(dataset_type, data_root, ann_file, image_dir, mapping,
                       metainfo_file):
    return dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file=ann_file,
        data_prefix=dict(img=image_dir),
        metainfo=dict(from_file=metainfo_file),
        pipeline=[
            dict(
                type='KeypointConverter',
                num_keypoints=133,
                mapping=mapping,
            )
        ],
    )


auxiliary_body_dataset_names = []
auxiliary_body_datasets = []

# COCO-WholeBody already uses the common 133-point order and therefore needs
# no inner converter. It shares the COCO 2017 image directory with COCO Body.
dataset_coco_wholebody = dict(
    type='CocoWholeBodyDataset',
    data_root=coco_body_root,
    data_mode='topdown',
    ann_file=os.path.relpath(coco_wholebody_ann, coco_body_root),
    data_prefix=dict(img='train2017/'),
    metainfo=dict(
        from_file='external/mmpose/configs/_base_/datasets/coco_wholebody.py'),
    pipeline=[],
)
ready_coco_wholebody = _ready_dataset(
    'COCO-WholeBody', dataset_coco_wholebody,
    [coco_wholebody_ann, os.path.join(coco_body_root, 'train2017')])
if ready_coco_wholebody is not None:
    auxiliary_body_datasets.append(ready_coco_wholebody)

dataset_coco_body = _make_body_dataset(
    'CocoDataset', coco_body_root,
    'annotations/person_keypoints_train2017.json', 'train2017/',
    coco_body_to_coco133,
    'external/mmpose/configs/_base_/datasets/coco.py')
ready_coco_body = _ready_dataset(
    'COCO-Body', dataset_coco_body,
    [os.path.join(coco_body_root, 'annotations',
                  'person_keypoints_train2017.json'),
     os.path.join(coco_body_root, 'train2017')])
if ready_coco_body is not None:
    auxiliary_body_datasets.append(ready_coco_body)

dataset_specs = [
    ('MPII', 'MpiiDataset', os.environ.get('MPII_ROOT', 'data'),
     'mpii/annotations/mpii_train.json', 'pose/MPI/images/', mpii_coco133,
     'external/mmpose/configs/_base_/datasets/mpii.py'),
    ('CrowdPose', 'CrowdPoseDataset', os.environ.get('CROWDPOSE_ROOT', 'data'),
     'crowdpose/annotations/mmpose_crowdpose_trainval.json',
     'pose/CrowdPose/images/', crowdpose_coco133,
     'external/mmpose/configs/_base_/datasets/crowdpose.py'),
    ('PoseTrack18', 'PoseTrack18Dataset',
     os.environ.get('POSETRACK_ROOT', 'data'),
     'posetrack18/annotations/posetrack18_train.json',
     'pose/PoseChallenge2018/', posetrack_coco133,
     'external/mmpose/configs/_base_/datasets/posetrack18.py'),
    ('AIC', 'AicDataset', os.environ.get('AIC_ROOT', 'data'),
     'aic/annotations/aic_train.json',
     'pose/ai_challenge/ai_challenger_keypoint_train_20170902/'
     'keypoint_train_images_20170902/', aic_coco133,
     'external/mmpose/configs/_base_/datasets/aic.py'),
    ('JHMDB', 'JhmdbDataset', os.environ.get('JHMDB_ROOT', 'data'),
     'jhmdb/annotations/Sub1_train.json', 'pose/JHMDB/', jhmdb_coco133,
     'external/mmpose/configs/_base_/datasets/jhmdb.py'),
]
for name, dataset_type, data_root, ann_file, image_dir, source_mapping, metainfo_file in dataset_specs:
    dataset = _make_body_dataset(
        dataset_type, data_root, ann_file, image_dir, source_mapping,
        metainfo_file)
    ready = _ready_dataset(
        name, dataset, [os.path.join(data_root, ann_file),
                        os.path.join(data_root, image_dir)])
    if ready is not None:
        auxiliary_body_datasets.append(ready)

if not auxiliary_body_datasets:
    print('WARNING: no auxiliary 2D body dataset is available; '
          'training is H3WB-only. Set COCO_BODY_ROOT or the dataset-specific '
          'root environment variables after downloading the assets.')

base_lr = 5e-4
max_epochs = 100
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
h3wb_train_dataset = dict(
    type='H36WWholeBodyDataset',
    subjects=['S1', 'S5'],
    ann_file=h3wb_ann,
    seq_len=1,
    causal=True,
    data_root=h3wb_root,
    data_prefix=dict(img=''),
    metainfo=dict(from_file='external/mmpose/configs/_base_/datasets/h3wb.py'),
    test_mode=False,
    subset_frac=0.2,
    pipeline=[],
)
train_datasets = [h3wb_train_dataset, *auxiliary_body_datasets]
train_dataset = dict(
    type='RelativeRatioCombinedDataset',
    datasets=train_datasets,
    # Ratios are relative to the effective H3WB length.  Thus COCO is 5% of
    # H3WB, rather than 5% of its own much larger raw length.
    sample_ratio_factor=[1.0] + [auxiliary_sample_ratio] * len(
        auxiliary_body_datasets),
    reference_dataset=0,
    pipeline=train_pipeline,
    metainfo=dict(from_file='external/mmpose/configs/_base_/datasets/h3wb.py'),
    test_mode=False,
)
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
)
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='H36WWholeBodyDataset',
        subjects=['S7'],
        ann_file=h3wb_ann,
        seq_len=1,
        causal=True,
        data_root=h3wb_root,
        data_prefix=dict(img=''),
        metainfo=dict(from_file='external/mmpose/configs/_base_/datasets/h3wb.py'),
        test_mode=True,
        subset_frac=0.2,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader
val_evaluator = [
    dict(type='SimpleMPJPE', mode='mpjpe'),
    dict(type='SimpleMPJPE', mode='p-mpjpe'),
]
test_evaluator = val_evaluator

default_hooks = dict(
    # SimpleMPJPE returns the uppercase metric key ``MPJPE``.
    checkpoint=dict(save_best='MPJPE', rule='less', max_keep_ckpts=2),
    logger=dict(interval=50),
)
custom_hooks = [
    dict(
        type='ValidationSampleVisualizationHook',
        joint_parents=joint_parents,
        num_samples=3,
        out_dir='val_visualizations',
    ),
    dict(
        type='EarlyStoppingHook',
        monitor='MPJPE',
        rule='less',
        # H3WB 3D coordinates are converted from millimeters to meters by
        # H36MWholeBodyDataset, so 0.0005 corresponds to 0.5 mm.
        min_delta=0.0005,
        patience=12,
    )
]
work_dir = 'work_dirs/rtmw3d-m_68_h3wb-256x192'
