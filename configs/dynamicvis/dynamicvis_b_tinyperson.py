_base_ = '/marimo/DynamicVis/configs_DynamicVis/Levir-Ship/dynamicvis_b_levirship_mamba.py'

# DynamicVis-B TinyPerson run. This is a variant, not a baseline mutation.
data_root = '/marimo/TinyPerson'
work_dir = '/marimo/mmdet_code/runs/dynamicvis_tinyperson/seed_42'
pretrained_ckpt = '/marimo/DynamicVis/checkpoints/pretrain_dynamicvis_b_bf16_mamba_epoch_200.pth'
batch_size = 4
num_workers = 4
img_size = 640
crop_size = (img_size, img_size)
randomness = dict(seed=42, deterministic=False)
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
custom_imports = dict(imports=['mmdet.datasets.transforms', 'dynamicvis'], allow_failed_imports=False)

# Strict TinyPerson baseline protocol: no Mosaic/OACP. Keep only the baseline's
# resize and horizontal-flip behavior, with DynamicVis's square input size.
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Pad', size=crop_size, pad_val=dict(img=(103.53, 116.28, 123.675))),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=(103.53, 116.28, 123.675))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor')),
]

dataset_type = 'CocoDataset'
metainfo = dict(classes=('person',))
train_dataloader = dict(
    batch_size=batch_size, num_workers=num_workers, persistent_workers=True,
    dataset=dict(type=dataset_type, data_root=data_root,
        ann_file=data_root + '/annotations/tiny_set_train.json',
        data_prefix=dict(img='erase_with_uncertain_dataset/train/'),
        metainfo=metainfo, pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=num_workers, persistent_workers=True, drop_last=False,
    dataset=dict(type=dataset_type, data_root=data_root,
        ann_file=data_root + '/annotations/tiny_set_test_with_dense.json',
        data_prefix=dict(img='erase_with_uncertain_dataset/test/'),
        metainfo=metainfo, test_mode=True, pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', metric=['bbox'], iou_thrs=[0.5],
    ann_file=data_root + '/annotations/tiny_set_test_with_dense.json')
test_evaluator = val_evaluator
model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=pretrained_ckpt, prefix='backbone.')),
    neck=dict(init_cfg=dict(type='Pretrained', checkpoint=pretrained_ckpt, prefix='pre_neck.')),
    bbox_head=dict(num_classes=1),
    test_cfg=dict(nms=dict(type='nms', iou_threshold=0.5)),
)
