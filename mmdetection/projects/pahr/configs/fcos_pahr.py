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
        detach_position_gate=False,
        guide_channels=0,
        use_output_gate=True,
        correction_gain=1.0),
    tiny_max_sqrt_area=16.0,
    position_stride=8,
    loss_pos_weight=0.1,
    loss_offset_weight=0.1,
    use_phase_shift=False,
    use_tiny_measurement=False,
    measurement_channels=16,
    measurement_center_weight=0.1,
    measurement_phase_weight=0.1,
    measurement_size_weight=0.1,
    measurement_size_blend=0.5,
    measurement_tiny_limit=24.0)
