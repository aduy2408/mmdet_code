_base_ = './fcos_r50-caffe_fpn_lmsce-ring_30e_levir-ship-768.py'

optim_wrapper = dict(
    paramwise_cfg=dict(
        bias_lr_mult=2.,
        bias_decay_mult=0.,
        custom_keys={
            'neck.lmsce_modules.0.transform.3': dict(lr_mult=10.)
        }))
work_dir = 'work_dirs/lmsce_opt_control/ring_lr10'
