_base_ = ['./fcos_baseline.py']

_base_.model.bbox_head.type = 'LTMRFCOSHead'
_base_.model.bbox_head.tiny_max_sqrt_area = 16.0
_base_.model.bbox_head.radius = 2
_base_.model.bbox_head.topk = 5
_base_.model.bbox_head.margin = 1.0
_base_.model.bbox_head.loss_weight = 0.05

custom_hooks = [
    dict(
        type='LTMRWeightWarmupHook',
        target_weight=0.05,
        warmup_ratio=0.1,
    )
]
