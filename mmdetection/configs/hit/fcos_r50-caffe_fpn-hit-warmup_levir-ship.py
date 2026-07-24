_base_ = './fcos_r50-caffe_fpn-hit_12e_levir-ship.py'

load_from = '../best_coco_bbox_mAP_epoch_12.pth'
work_dir = './work_dirs/levir_hit/sparse_warmup_seed42'

# One epoch updates only HIT. lr_mult=0 leaves the loaded baseline unchanged.
train_cfg = dict(max_epochs=1, val_interval=1)
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0, decay_mult=0.0),
            'neck.base_neck': dict(lr_mult=0.0, decay_mult=0.0),
            'bbox_head': dict(lr_mult=0.0, decay_mult=0.0),
        }))

