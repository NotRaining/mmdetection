_base_ = './fcos_r50_caffe_fpn_gn-head_1x_coco.py'

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(num_classes=1))

img_norm_cfg = dict(mean=[40, 40, 40], std=[1.0, 1.0, 1.0], to_rgb=False)
img_scale = (1333, 800)
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114),
    # dict(type='RandomAffine',
    #      scaling_ratio_range=(0.1, 2),
    #      border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    # dict(type='MixUp',
    #      img_scale=img_scale,
    #      ratio_range=(0.8, 1.6),
    #      pad_val=40),
    # dict(type='PhotoMetricDistortion',
    #      brightness_delta=32,
    #      contrast_range=(0.5, 1.5),
    #      saturation_range=(0.5, 1.5),
    #      hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
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

data_root = 'data/ssdd/'
classes = ['ship', ]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        _delete_=True,    # ignore the base config
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            classes=classes,
            img_prefix=data_root + 'train/',
            ann_file=data_root + 'annotations/instances_train.json',
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)],
        ),
        dynamic_scale=img_scale,
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        img_prefix=data_root + 'val/',
        ann_file=data_root + 'annotations/instances_val.json',
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        img_prefix=data_root + 'test/',
        ann_file=data_root + 'annotations/instances_test.json',
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 22])
# runner = dict(type='EpochBasedRunner', max_epochs=24)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[30, 40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
