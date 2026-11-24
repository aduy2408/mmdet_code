_base_ = './fcos_r50-caffe_fpn_morph-positive_30e_levir-ship-768.py'

model = dict(neck=dict(morphology=dict(mode='raw')))
work_dir = 'work_dirs/morphology_p3_768/raw'
