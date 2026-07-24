_base_ = './fcos_r50-caffe_fpn_gn-head_1x_coco.py'

# phase_mode is one of: pixel_unshuffle, context, deviation.
# phase_stages=(1, 2) replaces C2->C3 and C3->C4 only.
model = dict(
    backbone=dict(
        _delete_=True,
        type='DeviationResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        phase_mode='deviation',
        phase_stages=(1, 2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))
