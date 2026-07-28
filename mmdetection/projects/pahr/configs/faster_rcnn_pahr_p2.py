_base_ = [
    '../../../configs/faster_rcnn/'
    'faster-rcnn_r50-caffe_fpn_1x_coco.py'
]

custom_imports = dict(imports=['projects.pahr'])

model = dict(
    neck=dict(
        _delete_=True,
        type='PAHRFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5,
        locator_channels=64,
        detail_channels=64,
        guide_channels=0,
        use_output_gate=False,
        correction_gain=1.0,
        target_correction_ratio=0.0))
