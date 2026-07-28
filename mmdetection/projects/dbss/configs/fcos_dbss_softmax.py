_base_ = ['./fcos_dbss_ridge.py']

model = dict(neck=dict(
    residual_mode='softmax',
    projection_mode='softmax'))
