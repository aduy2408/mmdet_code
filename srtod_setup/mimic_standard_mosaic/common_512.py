_base_ = ['./common_640.py']

_mosaic_pipeline = [
    dict(type='Mosaic', img_scale=(512, 512), pad_val=114.0, prob=1.0),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(
    dataset=dict(pipeline=_mosaic_pipeline),
)
