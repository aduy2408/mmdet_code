import os

custom_imports = dict(
    imports=["dynamicvis", "projects.set", "mmdet.engine.hooks"],
    allow_failed_imports=False,
)
default_scope = "mmdet"

work_dir = "work_dirs/dynamicvis_varroa"
pretrained_ckpt = os.environ.get(
    "DYNAMICVIS_PRETRAINED_CKPT",
    "/marimo/DynamicVis/checkpoints/pretrain_dynamicvis_b_bf16_mamba_epoch_200.pth",
)
img_size = 640
batch_size = 8
num_workers = 8
num_classes = 1

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
log_level = "INFO"
log_processor = dict(type="LogProcessor", window_size=1, by_epoch=True)
randomness = dict(seed=42, deterministic=False)

_data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)

model = dict(
    type="FCOS",
    data_preprocessor=_data_preprocessor,
    backbone=dict(
        type="mmpretrain.DynamicVisBackbone",
        arch="b",
        path_type="forward_reverse_mean",
        sampling_scale=dict(type="decay", val=0.1),
        global_token_cfg=dict(pos="head", num=-1),
        is_softmax_on_x=True,
        img_size=img_size,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        spatial_token_keep_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        out_type="featmap",
        init_cfg=dict(type="Pretrained", checkpoint=pretrained_ckpt, prefix="backbone."),
    ),
    neck=dict(
        type="FPN",
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained_ckpt, prefix="pre_neck."),
    ),
    bbox_head=dict(
        type="FCOSHead",
        num_classes=num_classes,
        regress_ranges=((0, 20), (16, 40), (32, 80), (64, 160), (128, 1024)),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="GIoULoss", loss_weight=1.0),
        loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=3000,
    ),
)

_backend_args = None
_train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_backend_args, to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="Resize", scale=(img_size, img_size), keep_ratio=True, interpolation="bicubic"),
    dict(type="Pad", size=(img_size, img_size), pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="PackDetInputs"),
]
_test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_backend_args, to_float32=True),
    dict(type="Resize", scale=(img_size, img_size), keep_ratio=True),
    dict(type="Pad", size=(img_size, img_size), pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "pad_shape", "scale_factor")),
]
_dataset = dict(
    type="CocoDataset",
    data_root="/marimo/Varroa",
    ann_file="annotations/train.json",
    data_prefix=dict(img="images/train/"),
    metainfo=dict(classes=("varroa",)),
    pipeline=_train_pipeline,
    backend_args=_backend_args,
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=_dataset,
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root="/marimo/Varroa",
        ann_file="annotations/val.json",
        data_prefix=dict(img="images/val/"),
        metainfo=dict(classes=("varroa",)),
        test_mode=True,
        pipeline=_test_pipeline,
        backend_args=_backend_args,
    ),
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root="/marimo/Varroa",
        ann_file="annotations/test.json",
        data_prefix=dict(img="images/test/"),
        metainfo=dict(classes=("varroa",)),
        test_mode=True,
        pipeline=_test_pipeline,
        backend_args=_backend_args,
    ),
)

val_evaluator = dict(type="CocoMetric", metric=["bbox"], format_only=False, iou_thrs=[0.5], ann_file="/marimo/Varroa/annotations/val.json")
test_evaluator = dict(type="CocoMetric", metric=["bbox"], format_only=False, iou_thrs=[0.5], ann_file="/marimo/Varroa/annotations/test.json")
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=1)
optim_wrapper = dict(
    optimizer=dict(type="MuSGD", lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0005, muon=0.2, sgd=1.0),
    paramwise_cfg=dict(bias_lr_mult=1.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=35, norm_type=2),
)
param_scheduler = [
    dict(type="ConstantLR", factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type="CosineAnnealingLR", T_max=100, by_epoch=True, begin=0, end=100, eta_min=0.0),
]
custom_hooks = [
    dict(type="EarlyStoppingHook", monitor="coco/bbox_mAP", rule="greater", patience=15, min_delta=0.001),
]
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, by_epoch=True, max_keep_ckpts=1, save_last=True, save_best="coco/bbox_mAP", rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook", draw=False),
)
