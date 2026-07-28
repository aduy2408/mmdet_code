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
        diversity_beta=1.0,
        basis_similarity_threshold=0.9,
        selector_mode='legacy_forced_k',
        residual_mode='ridge',
        projection_mode='ridge',
        ridge_lambda=1e-3,
        temperature=0.1,
        gamma_max=0.3,
        use_haar_reliability=False),
    position_stride=8,
    improvement_margin=0.03,
    loss_sep_weight=0.5)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'neck.direction.2': dict(lr_mult=10.0, decay_mult=0.0)
        }))
