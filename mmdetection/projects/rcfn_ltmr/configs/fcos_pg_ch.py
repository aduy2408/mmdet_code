_base_ = ['./fcos_pg_aux.py']

_base_.model.neck.gate_mode = 'contrast_position'
_base_.model.neck.predict_contrast = True
