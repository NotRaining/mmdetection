_base_ = './fcos_r50_caffe_fpn_gn-head_1x_coco.py'

model = dict(
    bbox_head=dict(num_classes=1),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

img_norm_cfg = dict(
    mean=[40, 40, 40], std=[1.0, 1.0, 1.0], to_rgb=False)
img_scale = (1333, 800)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=[(900, 600), (1500, 1000)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal']),
    # dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # added for showing gt
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(900, 600), (1050, 700), (1200, 800), (1350, 900), (1500, 1000)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data_root = 'data/ssdd/'
classes = ['ship', ]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        classes=classes,
        img_prefix=data_root + 'train/',
        ann_file=data_root + 'annotations/instances_train.json',
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        img_prefix=data_root + 'val/',
        ann_file=data_root + 'annotations/instances_val.json',
        pipeline=val_pipeline),
    test=dict(
        classes=classes,
        img_prefix=data_root + 'test/',
        ann_file=data_root + 'annotations/instances_test.json',
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[30, 40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
# cudnn_benchmark = True

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])