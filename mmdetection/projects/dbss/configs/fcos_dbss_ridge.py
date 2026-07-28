_base_ = ['../../../configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py']

custom_imports = dict(imports=['projects.dbss'])

model = dict(
    type='DBSSFCOS',
    neck=dict(
        _delete_=True,
        type='DBSSFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True,
        embed_channels=64,
        candidate_grid=(8, 8),
        shortlist_size=24,
        num_bases=8,
        diversity_beta=0.25,
        projection_mode='ridge',
        ridge_lambda=1e-3,
        temperature=0.1,
        gamma_max=0.1,
        use_haar_reliability=False),
    position_stride=8,
    separation_margin=0.2,
    loss_sep_weight=0.1)
