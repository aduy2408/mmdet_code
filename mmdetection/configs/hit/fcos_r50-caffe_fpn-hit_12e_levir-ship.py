_base_ = '../fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py'

data_root = ''
image_prefix = '../LevirShipData/All Images/'
annotation_root = 'data/levir_ship_coco/annotations/'
metainfo = dict(classes=('ship', ))

model = dict(
    neck=dict(
        _delete_=True,
        type='FeatureAugmentNeck',
        base_neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5,
            relu_before_extra_convs=True),
        out_channels=256,
        levels=(0, ),
        hit=dict(
            type='DualIrreducibilityHIT',
            stride=8,
            reduction=8,
            topk=4,
            max_offset=8.0,
            sigma_min=0.5,
            sigma_max=1.5,
            alpha_init=1e-3,
            alpha_max=1.0,
            loss_recon_spatial_weight=0.1,
            loss_recon_channel_weight=0.1,
            loss_offset_weight=1.0)),
    bbox_head=dict(num_classes=1))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        ann_file=annotation_root + 'train.json',
        data_prefix=dict(img=image_prefix),
        metainfo=metainfo,
        pipeline=train_pipeline))
val_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        ann_file=annotation_root + 'val.json',
        data_prefix=dict(img=image_prefix),
        metainfo=metainfo,
        pipeline=test_pipeline))
test_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        ann_file=annotation_root + 'test.json',
        data_prefix=dict(img=image_prefix),
        metainfo=metainfo,
        pipeline=test_pipeline))

val_evaluator = dict(ann_file=annotation_root + 'val.json')
test_evaluator = dict(ann_file=annotation_root + 'test.json')

train_cfg = dict(max_epochs=12, val_interval=1)
randomness = dict(seed=42)
