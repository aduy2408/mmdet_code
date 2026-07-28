_base_ = ['./fcos_dbss_ridge.py']

model = dict(neck=dict(projection_mode='softmax'))
