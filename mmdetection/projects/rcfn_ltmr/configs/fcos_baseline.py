_base_ = [
    '../../../configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py'
]

custom_imports = dict(imports=['projects.rcfn_ltmr'])

randomness = dict(seed=42)
