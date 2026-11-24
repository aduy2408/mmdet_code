# YOLO baseline protocol override for SR-TOD.
# This is a variant config, not a canonical SR-TOD baseline.
custom_imports = dict(
    imports=['srtod_project.optim.musgd'],
    allow_failed_imports=False,
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='MuSGD',
        lr=0.01,
        momentum=0.9,
        nesterov=True,
        weight_decay=0.0005,
        muon=0.2,
        sgd=1.0,
    ),
)

# Ultralytics default is linear warmup followed by linear decay because
# cos_lr=False. The final LR is lr0*lrf = 0.01*0.01.
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=3),
    dict(type='LinearLR', start_factor=1.0, end_factor=0.01, by_epoch=True, begin=3, end=100),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=8, num_workers=8)
test_dataloader = dict(batch_size=8, num_workers=8)
seed = 42

# Explicit no-Mosaic protocol. The generated SR-TOD pipeline contains only
# LoadImage, LoadAnnotations, Resize, RandomFlip, and PackDetInputs.
mosaic = 0.0
close_mosaic = 0
nms_iou = 0.5
