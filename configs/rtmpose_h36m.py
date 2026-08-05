# RTMPose-L 在 H36M 17 点数据集上训练
# 数据: S1/S5/S7 图像 + h5 part 2D 标注 (59980 帧)
_base_ = ['/tmp/mmpose_repo/configs/_base_/default_runtime.py']

max_epochs = 100
stage2_num_epochs = 10
base_lr = 1e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr*0.05, begin=max_epochs//2,
         end=max_epochs, T_max=max_epochs//2, by_epoch=True, convert_to_iter_based=True),
]
auto_scale_lr = dict(base_batch_size=256)

codec = dict(type='SimCCLabel', input_size=(288, 384), sigma=(6., 6.93),
             simcc_split_ratio=2.0, normalize=False, use_dark=False)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet', type='CSPNeXt', arch='P5', expand_ratio=0.5,
        deepen_factor=1., widen_factor=1., out_indices=(4,),
        channel_attention=True, norm_cfg=dict(type='SyncBN'), act_cfg=dict(type='SiLU'),
        init_cfg=dict(type='Pretrained', prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth')),
    head=dict(
        type='RTMCCHead', in_channels=1024, out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(hidden_dims=256, s=128, expansion_factor=2, dropout_rate=0.,
                     drop_path=0., act_fn='SiLU', use_rel_bias=False, pos_enc=False),
        loss=dict(type='KLDiscretLoss', use_target_weight=True, beta=10., label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# H36M 17 关节 metainfo (VideoPose3D 顺序)
dataset_info = dict(
    dataset_name='h36m17',
    keypoint_info={
        0: dict(name='Hip', id=0, color=[51,153,255], type='lower', swap=''),
        1: dict(name='RHip', id=1, color=[255,128,0], type='lower', swap='LHip'),
        2: dict(name='RKnee', id=2, color=[255,128,0], type='lower', swap='LKnee'),
        3: dict(name='RAnkle', id=3, color=[255,128,0], type='lower', swap='LAnkle'),
        4: dict(name='LHip', id=4, color=[0,255,0], type='lower', swap='RHip'),
        5: dict(name='LKnee', id=5, color=[0,255,0], type='lower', swap='RKnee'),
        6: dict(name='LAnkle', id=6, color=[0,255,0], type='lower', swap='RAnkle'),
        7: dict(name='Spine', id=7, color=[51,153,255], type='upper', swap=''),
        8: dict(name='Thorax', id=8, color=[51,153,255], type='upper', swap=''),
        9: dict(name='Neck', id=9, color=[51,153,255], type='upper', swap=''),
        10: dict(name='Head', id=10, color=[51,153,255], type='upper', swap=''),
        11: dict(name='RShoulder', id=11, color=[255,128,0], type='upper', swap='LShoulder'),
        12: dict(name='RElbow', id=12, color=[255,128,0], type='upper', swap='LElbow'),
        13: dict(name='RWrist', id=13, color=[255,128,0], type='upper', swap='LWrist'),
        14: dict(name='LShoulder', id=14, color=[0,255,0], type='upper', swap='RShoulder'),
        15: dict(name='LElbow', id=15, color=[0,255,0], type='upper', swap='RElbow'),
        16: dict(name='LWrist', id=16, color=[0,255,0], type='upper', swap='RWrist'),
    },
    skeleton_info={
        0: dict(link=('Hip','RHip'), id=0, color=[255,128,0]),
        1: dict(link=('RHip','RKnee'), id=1, color=[255,128,0]),
        2: dict(link=('RKnee','RAnkle'), id=2, color=[255,128,0]),
        3: dict(link=('Hip','LHip'), id=3, color=[0,255,0]),
        4: dict(link=('LHip','LKnee'), id=4, color=[0,255,0]),
        5: dict(link=('LKnee','LAnkle'), id=5, color=[0,255,0]),
        6: dict(link=('Hip','Spine'), id=6, color=[51,153,255]),
        7: dict(link=('Spine','Thorax'), id=7, color=[51,153,255]),
        8: dict(link=('Thorax','Neck'), id=8, color=[51,153,255]),
        9: dict(link=('Neck','Head'), id=9, color=[51,153,255]),
        10: dict(link=('Thorax','RShoulder'), id=10, color=[255,128,0]),
        11: dict(link=('RShoulder','RElbow'), id=11, color=[255,128,0]),
        12: dict(link=('RElbow','RWrist'), id=12, color=[255,128,0]),
        13: dict(link=('Thorax','LShoulder'), id=13, color=[0,255,0]),
        14: dict(link=('LShoulder','LElbow'), id=14, color=[0,255,0]),
        15: dict(link=('LElbow','LWrist'), id=15, color=[0,255,0]),
    },
    joint_weights=[1.0] * 17,
    sigmas=[0.05] * 17,
)

# 数据集
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/mnt/disk2/ch/H36M/images/'
backend_args = dict(backend='local')

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=128, num_workers=4, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root, data_mode=data_mode,
                 ann_file='/home/user/ch/MyPoseLift/data/cache/h36m_2d/annotations/h36m_train_03.json', metainfo=dataset_info,
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=128, num_workers=4, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(type=dataset_type, data_root=data_root, data_mode=data_mode,
                 ann_file='/home/user/ch/MyPoseLift/data/cache/h36m_2d/annotations/h36m_val.json', metainfo=dataset_info,
                 pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file='/home/user/ch/MyPoseLift/data/cache/h36m_2d/annotations/h36m_val.json')
test_evaluator = val_evaluator
