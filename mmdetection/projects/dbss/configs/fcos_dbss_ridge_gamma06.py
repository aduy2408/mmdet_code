_base_ = ['./fcos_dbss_ridge.py']

model = dict(neck=dict(gamma_max=0.6, legacy_artifact_mode=True))
