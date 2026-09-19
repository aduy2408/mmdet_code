_base_ = ['./mimic_standard_mosaic_common_640.py']

_mosaic_pipeline = [
    dict(type='Mosaic', img_scale=(512, 512), pad_val=114, prob=1.0),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=_mosaic_pipeline))
