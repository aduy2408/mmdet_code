_base_ = ['../../../configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py']

custom_imports = dict(imports=['projects.pahr'])

model = dict(
    neck=dict(
        _delete_=True,
        type='HaarC2FusionFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True,
        fusion_channels=256))
