_base_ = '../fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py'

custom_imports = dict(
    imports=['projects.set'],
    allow_failed_imports=False,
)

# SET keeps the upstream FCOS model and training defaults, with the only
# architecture change being the training-only HBS/API enhancement.
model = dict(
    type='FCOS_set',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        norm_cfg=None,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        loss_bbox=dict(type='DIoULoss', loss_weight=1.0)),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000),
    set_cfg=dict(
        reg_factor_range=[0, 0.01, 0.1, 0.5, 1, 2, 5],
        reg_factors=[4, 4, 4, 4, 4],
        scale=1.0),
)

# Match the YOLO matrix protocol: 100 epochs, no early stopping.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100, eta_min=0.0),
]
optim_wrapper = dict(
    optimizer=dict(type='MuSGD', lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0005, muon=0.2, sgd=1.0),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
)
