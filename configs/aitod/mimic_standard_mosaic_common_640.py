# YOLO standard-Mosaic protocol for SET, non-canonical variant.
custom_imports = dict(
    imports=['mmdet.core.utils.musgd', 'mmdet.core.hook.mosaic_close_hook'],
    allow_failed_imports=False,
)

_inner_train = dict(data['train'])
_inner_train['pipeline'] = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
_mosaic_pipeline = [
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114, prob=1.0),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='MultiImageMixDataset',
        dataset=_inner_train,
        pipeline=_mosaic_pipeline),
)
optimizer = dict(
    type='MuSGD', lr=0.01, momentum=0.9, nesterov=True,
    weight_decay=0.0005, muon=0.2, sgd=1.0)
optimizer_config = dict(grad_clip=None)
runner = dict(type='EpochBasedRunner', max_epochs=100)
lr_config = dict(
    policy='poly', power=1.0, min_lr=0.0001,
    warmup='linear', warmup_iters=500, warmup_ratio=0.001)
custom_hooks = [dict(type='MosaicCloseHook', num_last_epochs=10)]
seed = 42
mosaic = 1.0
close_mosaic = 10
nms_iou = 0.5
