_base_ = '../fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py'

metainfo = dict(classes=('ship', ))
image_root = '../LevirShipData/All Images/'
annotation_root = 'data/levir_ship_coco/annotations/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(768, 768), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(768, 768), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

model = dict(bbox_head=dict(num_classes=1))
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root='',
        ann_file=annotation_root + 'train.json',
        data_prefix=dict(img=image_root),
        metainfo=metainfo,
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        data_root='',
        ann_file=annotation_root + 'val.json',
        data_prefix=dict(img=image_root),
        metainfo=metainfo,
        pipeline=test_pipeline))
test_dataloader = dict(
    dataset=dict(
        data_root='',
        ann_file=annotation_root + 'test.json',
        data_prefix=dict(img=image_root),
        metainfo=metainfo,
        pipeline=test_pipeline))
val_evaluator = dict(ann_file=annotation_root + 'val.json')
test_evaluator = dict(ann_file=annotation_root + 'test.json')

train_cfg = dict(max_epochs=30, val_interval=1)
randomness = dict(seed=42)
auto_scale_lr = dict(enable=True, base_batch_size=16)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='coco/bbox_mAP',
        rule='greater',
        save_last=True))
work_dir = 'work_dirs/morphology_p3_768/baseline'
