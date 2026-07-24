_base_ = './fcos_r50-caffe_fpn-hit_12e_levir-ship.py'

# Stage 3: enable offset gradients only after the detached-input run passes its
# validation gate. Keep every other transport setting fixed.
model = dict(neck=dict(hit=dict(detach_offset_input=False)))

work_dir = './work_dirs/levir_hit/joint_seed42'
