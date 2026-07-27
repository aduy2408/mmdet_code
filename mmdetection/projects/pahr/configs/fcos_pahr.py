_base_ = ['../../../configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py']

custom_imports = dict(imports=['projects.pahr'])

model = dict(
    type='PAHRFCOS',
    neck=dict(
        _delete_=True,
        type='PAHRFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True,
        locator_channels=64,
        detail_channels=64,
        gate_power=1.0,
        correction_gate_floor=0.0,
        detach_position_gate=False),
    tiny_max_sqrt_area=16.0,
    position_stride=8,
    loss_pos_weight=0.1,
    loss_offset_weight=0.1,
    use_phase_shift=False)
