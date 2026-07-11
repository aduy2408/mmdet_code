# FeatureAugmentNeck wrapper for FPN + DyHead ATSS-style configs.

neck = dict(
    type='FeatureAugmentNeck',
    base_neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=6)
    ],
    out_channels=256,
    levels=(0, ),
    dgfe=dict(type='FeatureDGFE'),
    api=dict(type='AdversarialPerturbationInjection'))
