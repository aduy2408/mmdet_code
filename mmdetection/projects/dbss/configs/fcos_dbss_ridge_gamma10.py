_base_ = ['./fcos_dbss_ridge.py']

model = dict(neck=dict(gamma_max=1.0))
