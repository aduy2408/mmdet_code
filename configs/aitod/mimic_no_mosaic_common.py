# YOLO baseline protocol override for SET.
# Variant config only. It does not mutate the upstream SET configs.
custom_imports = dict(
    imports=['mmdet.core.utils.musgd'],
    allow_failed_imports=False,
)

optimizer = dict(
    type='MuSGD',
    lr=0.01,
    momentum=0.9,
    nesterov=True,
    weight_decay=0.0005,
    muon=0.2,
    sgd=1.0,
)
optimizer_config = dict(grad_clip=None)
runner = dict(type='EpochBasedRunner', max_epochs=100)

# MMCV 1.x uses iteration-based warmup. 500 iterations is the existing
# project convention; the post-warmup policy is linear decay to lr0*0.01.
lr_config = dict(
    policy='poly',
    power=1.0,
    min_lr=0.0001,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
)
data = dict(samples_per_gpu=8, workers_per_gpu=8)
seed = 42
mosaic = 0.0
close_mosaic = 0
nms_iou = 0.5
