_base_ = './faster_rcnn_r101_fpn_2x_coco.py'

model = dict(
    backbone=dict(init_cfg=None),
    roi_head=dict(
        bbox_head=dict(num_classes=1)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
)

# img_scale = (1333, 800)
img_scale = (512, 416)
img_norm_cfg = dict(mean=[40, 40, 40], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114),
    # dict(type='RandomAffine',
    #      scaling_ratio_range=(0.1, 2),
    #      border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    # dict(type='MixUp',
    #      img_scale=img_scale,
    #      # ratio_range=(0.8, 1.6),
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
    samples_per_gpu=4,
    workers_per_gpu=8,
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

# lr=0.02, loss_rpn=NAN
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
