_base_ = ['./fcos_baseline.py']

_base_.model.neck.type = 'RCFNFPN'
_base_.model.neck.eps = 1e-4
_base_.model.neck.gamma_init = 0.0
