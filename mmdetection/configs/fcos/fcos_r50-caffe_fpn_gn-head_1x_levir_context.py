_base_ = './fcos_r50-caffe_fpn_gn-head_1x_levir_deviation.py'

model = dict(backbone=dict(phase_mode='context'))
