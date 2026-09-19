import os

_base_ = ['./fcos_r50_baseline.py']

dataset_type = 'CocoDataset'
classes = ('person',)
set_root = os.environ.get('SET_ROOT', os.getcwd())
data_root = os.environ.get(
    'SET_TINYPERSON_ROOT', os.path.abspath(os.path.join(
        set_root, '../TinyPerson/tiny_set')))
prepared_root = os.environ.get(
    'SET_TINYPERSON_PREPARED_ROOT',
    os.path.abspath(os.path.join(set_root, 'data/tinyperson_seed42')))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
corner_train_pipeline = [
    dict(type='LoadTinyPersonImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
corner_test_pipeline = [
    dict(type='LoadTinyPersonImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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

image_root = prepared_root + '/images/erase_with_uncertain_dataset/train/'
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=prepared_root + '/annotations/train_corner.json',
        img_prefix=image_root,
        classes=classes,
        pipeline=corner_train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=prepared_root + '/annotations/val_corner.json',
        img_prefix=image_root,
        classes=classes,
        pipeline=corner_test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/corner/task/'
                 'tiny_set_test_sw640_sh512_all.json',
        img_prefix=data_root + '/test/',
        classes=classes,
        pipeline=corner_test_pipeline))

model = dict(bbox_head=dict(num_classes=1))
