_base_ = ['./fcos_dbss_ridge.py']

model = dict(neck=dict(use_haar_reliability=True))
