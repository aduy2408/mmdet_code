_base_ = ['./fcos_baseline.py']

_base_.model.neck.type = 'RCFNFPN'
_base_.model.neck.eps = 1e-4
_base_.model.neck.gamma_init = 0.0
_base_.model.neck.position_channels = 0
_base_.model.neck.gate_mode = 'none'
_base_.model.neck.predict_contrast = False
