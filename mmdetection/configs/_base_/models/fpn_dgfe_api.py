# FeatureAugmentNeck wrapper for FPN-based detectors.
# Use this as a model.neck replacement in ATSS, RetinaNet, Faster/Cascade R-CNN,
# TridentNet, DetectoRS, and SABL configs that use a 256-channel FPN.

neck = dict(
    type='FeatureAugmentNeck',
    base_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    out_channels=256,
    levels=(0, ),
    dgfe=dict(type='FeatureDGFE'),
    api=dict(type='AdversarialPerturbationInjection'))
