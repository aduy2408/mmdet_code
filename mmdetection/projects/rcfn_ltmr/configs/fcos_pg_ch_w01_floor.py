_base_ = ['./fcos_pg_ch.py']

_base_.model.loss_pos_weight = 0.1
_base_.model.neck.gate_floor = 0.1
