_base_ = './fcos_r50-caffe_fpn-hit_12e_levir-ship.py'

# Used by tools/analysis_tools/hit_probe.py. The detector is loaded from the
# baseline checkpoint and frozen by the script; only reconstructors are fitted.
model = dict(
    neck=dict(
        hit=dict(
            transport_enabled=False,
            background_recon_only=True,
            loss_offset_weight=0.0)))

