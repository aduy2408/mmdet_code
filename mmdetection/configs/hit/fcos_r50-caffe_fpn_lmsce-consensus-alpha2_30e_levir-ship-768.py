_base_ = './fcos_r50-caffe_fpn_lmsce-consensus_30e_levir-ship-768.py'

model = dict(neck=dict(lmsce=dict(residual_scale=2.)))
work_dir = 'work_dirs/lmsce_strength/train_alpha2'
