# YOLO standard-Mosaic protocol for SR-TOD, non-canonical variant.
custom_imports = dict(
    imports=[
        'srtod_project.srtod_faster_rcnn.srtod_fasterrcnn',
        'srtod_project.srtod_faster_rcnn.srtod_datapreprocessor',
        'srtod_project.srtod_faster_rcnn.srtod_twostagedetector',
        'srtod_project.optim.musgd',
        'srtod_project.hooks.mosaic_close_hook',
    ],
    allow_failed_imports=False,
)

_inner_train_dataset = dict(train_dataloader['dataset'])
_inner_train_dataset['pipeline'] = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
_mosaic_pipeline = [
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0, prob=1.0),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=_inner_train_dataset,
        pipeline=_mosaic_pipeline,
    ),
)
val_dataloader = dict(batch_size=8, num_workers=8)
test_dataloader = dict(batch_size=8, num_workers=8)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='MuSGD', lr=0.01, momentum=0.9, nesterov=True,
        weight_decay=0.0005, muon=0.2, sgd=1.0),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=3),
    dict(type='LinearLR', start_factor=1.0, end_factor=0.01, by_epoch=True, begin=3, end=100),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
custom_hooks = [dict(type='MosaicCloseHook', num_last_epochs=10)]
seed = 42
mosaic = 1.0
close_mosaic = 10
nms_iou = 0.5
