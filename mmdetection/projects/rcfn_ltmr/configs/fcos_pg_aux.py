_base_ = ['./fcos_r2.py']

_base_.model.type = 'PGRCFNFCOS'
_base_.model.neck.position_channels = 64
_base_.model.neck.gate_mode = 'none'
_base_.model.neck.predict_contrast = False
_base_.model.tiny_max_sqrt_area = 16.0
_base_.model.gaussian_alpha = 1.0
_base_.model.gaussian_sigma_min = 1.0
_base_.model.position_positive_weight = 4.0
_base_.model.loss_pos_weight = 1.0
