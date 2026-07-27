_base_ = ['./fcos_pg_aux.py']

_base_.model.neck.gate_mode = 'position'
