# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.structures import InstanceData
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.detectors.base import BaseDetector
from mmdet.models.necks import (FPG, FPN, FPN_CARAFE, NASFCOS_FPN, NASFPN, SSH,
                                YOLOXPAFPN, ChannelMapper, DilatedEncoder,
                                DyHead, FeatureAugmentNeck,
                                AdversarialPerturbationInjection, SSDNeck,
                                YOLOV3Neck)
from mmdet.structures import DetDataSample


class _APIOnlyDetector(BaseDetector):

    def __init__(self, api):
        super().__init__()
        self.neck = _APIOnlyNeck(api)

    def loss(self, batch_inputs, batch_data_samples):
        raise NotImplementedError

    def predict(self, batch_inputs, batch_data_samples):
        raise NotImplementedError

    def _forward(self, batch_inputs, batch_data_samples=None):
        raise NotImplementedError

    def extract_feat(self, batch_inputs):
        raise NotImplementedError


class _APIOnlyNeck:

    def __init__(self, api):
        self.api_modules = [api]

    def clear_api_state(self):
        self.api_modules[0].clear_state()

    def perturb_api(self):
        self.api_modules[0].perturb()


def test_fpn():
    """Tests fpn."""
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 8

    # end_level=-1 is equal to end_level=3
    FPN(in_channels=in_channels,
        out_channels=out_channels,
        start_level=0,
        end_level=-1,
        num_outs=5)
    FPN(in_channels=in_channels,
        out_channels=out_channels,
        start_level=0,
        end_level=3,
        num_outs=5)

    # `num_outs` is not equal to end_level - start_level + 1
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=2,
            num_outs=3)

    # `num_outs` is not equal to len(in_channels) - start_level
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            num_outs=2)

    # `end_level` is larger than len(in_channels) - 1
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=4,
            num_outs=2)

    # `num_outs` is not equal to end_level - start_level
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=3,
            num_outs=1)

    # Invalid `add_extra_convs` option
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            add_extra_convs='on_xxx',
            num_outs=5)

    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        num_outs=5)

    # FPN expects a multiple levels of features per image
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]
    outs = fpn_model(feats)
    assert fpn_model.add_extra_convs == 'on_input'
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)


def test_feature_augment_neck():
    feats = (
        torch.rand(2, 8, 32, 32, requires_grad=True),
        torch.rand(2, 16, 16, 16, requires_grad=True),
        torch.rand(2, 32, 8, 8, requires_grad=True),
        torch.rand(2, 64, 4, 4, requires_grad=True),
    )
    imgs = torch.rand(2, 3, 128, 128)
    neck = FeatureAugmentNeck(
        base_neck=dict(
            type='FPN',
            in_channels=[8, 16, 32, 64],
            out_channels=8,
            num_outs=4),
        out_channels=8,
        levels=(0, ),
        dgfe=dict(type='FeatureDGFE'),
        api=dict(type='AdversarialPerturbationInjection',
                 forward_mode='partial'))
    assert neck.api_modules[0].forward_mode == 'partial'

    neck.eval()
    eval_outs = neck(feats, batch_inputs=imgs)
    assert len(eval_outs) == 4
    assert eval_outs[0].shape == (2, 8, 32, 32)

    neck.train()
    neck.capture_api()
    outs = neck(feats, batch_inputs=imgs)
    api = neck.api_modules[0]
    assert api.captured is not None
    target = torch.zeros(2, 1, 32, 32)
    loss = api.auxiliary_loss(target) + outs[0].sum()
    grad = torch.autograd.grad(loss, api.captured, retain_graph=True)[0]
    assert api.set_perturbation_from_grad(grad)
    assert api.spatial_guidance is None
    norms = api.perturbation.float().flatten(1).norm(p=2, dim=1)
    assert torch.allclose(norms, torch.full_like(norms, api.rho))
    neck.perturb_api()
    perturbed_outs = neck(feats, batch_inputs=imgs)
    assert perturbed_outs[0].shape == outs[0].shape


def test_dgfe_guided_api_perturbation():
    api = AdversarialPerturbationInjection(
        channels=2, rho=0.2, api_weight=1.0, guidance_mode='dgfe')
    grad = torch.ones(2, 2, 2, 2)
    guidance = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]).expand(
        2, -1, -1, -1)
    api.set_spatial_guidance(guidance)

    assert api.set_perturbation_from_grad(grad)
    perturbation = api.perturbation
    assert perturbation is not None
    assert torch.count_nonzero(perturbation[..., 1::2]) == 0
    norms = perturbation.float().flatten(1).norm(p=2, dim=1)
    assert torch.allclose(norms, torch.full_like(norms, 0.2))

    feats = (
        torch.rand(1, 8, 16, 16),
        torch.rand(1, 16, 8, 8),
        torch.rand(1, 32, 4, 4),
    )
    neck = FeatureAugmentNeck(
        base_neck=dict(
            type='FPN', in_channels=[8, 16, 32], out_channels=8,
            num_outs=3),
        levels=(0, ),
        dgfe=dict(type='FeatureDGFE', upsample_steps=1),
        api=dict(
            type='AdversarialPerturbationInjection', guidance_mode='dgfe'))
    neck.train()
    neck.capture_api()
    outs = neck(feats, batch_inputs=torch.rand(1, 3, 32, 32))
    guided_api = neck.api_modules[0]
    spatial_prob = neck.dgfe_modules['0'].last_aux['spatial_prob']
    assert guided_api.captured is outs[0]
    assert torch.equal(guided_api.spatial_guidance, spatial_prob.detach())


def test_api_partial_forward_uses_captured_feature():
    api = AdversarialPerturbationInjection(
        channels=8, rho=0.1, api_weight=1.0, forward_mode='partial')
    api.train()
    detector = _APIOnlyDetector(api)
    detector.train()

    feature = torch.ones(1, 8, 4, 4, requires_grad=True)
    other_feature = torch.zeros(1, 8, 2, 2, requires_grad=True)
    clean_features = (feature, other_feature)
    api.captured = feature
    clean_losses = dict(loss_cls=feature.sum())
    batch_inputs = torch.zeros(1, 3, 16, 16)
    data_sample = DetDataSample()
    data_sample.gt_instances = InstanceData(bboxes=torch.empty(0, 4))

    def compute_losses(adv_features):
        assert adv_features[0] is not feature
        assert adv_features[1] is other_feature
        return dict(loss_cls=adv_features[0].sum())

    def compute_full_losses():
        raise AssertionError('partial forward should not re-run full losses')

    losses = detector.api_augmented_losses(
        batch_inputs, [data_sample], clean_losses, clean_features,
        compute_losses, compute_full_losses)

    assert 'api_adv_loss_cls' in losses
    assert 'loss_api_aux' in losses

    # Tests for fpn with no extra convs (pooling is used instead)
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=False,
        num_outs=5)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    assert not fpn_model.add_extra_convs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Tests for fpn with lateral bns
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        no_norm_on_lateral=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        num_outs=5)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    assert fpn_model.add_extra_convs == 'on_input'
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
    bn_exist = False
    for m in fpn_model.modules():
        if isinstance(m, _BatchNorm):
            bn_exist = True
    assert bn_exist

    # Bilinear upsample
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        upsample_cfg=dict(mode='bilinear', align_corners=True),
        num_outs=5)
    fpn_model(feats)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    assert fpn_model.add_extra_convs == 'on_input'
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Scale factor instead of fixed upsample size upsample
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        upsample_cfg=dict(scale_factor=2),
        num_outs=5)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'inputs'
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_input',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_input'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'laterals'
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_lateral',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_lateral'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'outputs'
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_output',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_output'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)


def test_channel_mapper():
    """Tests ChannelMapper."""
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 8
    kernel_size = 3
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]

    # in_channels must be a list
    with pytest.raises(AssertionError):
        channel_mapper = ChannelMapper(
            in_channels=10, out_channels=out_channels, kernel_size=kernel_size)
    # the length of channel_mapper's inputs must be equal to the length of
    # in_channels
    with pytest.raises(AssertionError):
        channel_mapper = ChannelMapper(
            in_channels=in_channels[:-1],
            out_channels=out_channels,
            kernel_size=kernel_size)
        channel_mapper(feats)

    channel_mapper = ChannelMapper(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size)

    outs = channel_mapper(feats)
    assert len(outs) == len(feats)
    for i in range(len(feats)):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)


def test_dilated_encoder():
    in_channels = 16
    out_channels = 32
    out_shape = 34
    dilated_encoder = DilatedEncoder(in_channels, out_channels, 16, 2,
                                     [2, 4, 6, 8])
    feat = [torch.rand(1, in_channels, 34, 34)]
    out_feat = dilated_encoder(feat)[0]
    assert out_feat.shape == (1, out_channels, out_shape, out_shape)


def test_yolov3_neck():
    # num_scales, in_channels, out_channels must be same length
    with pytest.raises(AssertionError):
        YOLOV3Neck(num_scales=3, in_channels=[16, 8, 4], out_channels=[8, 4])

    # len(feats) must equal to num_scales
    with pytest.raises(AssertionError):
        neck = YOLOV3Neck(
            num_scales=3, in_channels=[16, 8, 4], out_channels=[8, 4, 2])
        feats = (torch.rand(1, 4, 16, 16), torch.rand(1, 8, 16, 16))
        neck(feats)

    # test normal channels
    s = 32
    in_channels = [16, 8, 4]
    out_channels = [8, 4, 2]
    feat_sizes = [s // 2**i for i in range(len(in_channels) - 1, -1, -1)]
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels) - 1, -1, -1)
    ]
    neck = YOLOV3Neck(
        num_scales=3, in_channels=in_channels, out_channels=out_channels)
    outs = neck(feats)

    assert len(outs) == len(feats)
    for i in range(len(outs)):
        assert outs[i].shape == \
               (1, out_channels[i], feat_sizes[i], feat_sizes[i])

    # test more flexible setting
    s = 32
    in_channels = [32, 8, 16]
    out_channels = [19, 21, 5]
    feat_sizes = [s // 2**i for i in range(len(in_channels) - 1, -1, -1)]
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels) - 1, -1, -1)
    ]
    neck = YOLOV3Neck(
        num_scales=3, in_channels=in_channels, out_channels=out_channels)
    outs = neck(feats)

    assert len(outs) == len(feats)
    for i in range(len(outs)):
        assert outs[i].shape == \
               (1, out_channels[i], feat_sizes[i], feat_sizes[i])


def test_ssd_neck():
    # level_strides/level_paddings must be same length
    with pytest.raises(AssertionError):
        SSDNeck(
            in_channels=[8, 16],
            out_channels=[8, 16, 32],
            level_strides=[2],
            level_paddings=[2, 1])

    # length of out_channels must larger than in_channels
    with pytest.raises(AssertionError):
        SSDNeck(
            in_channels=[8, 16],
            out_channels=[8],
            level_strides=[2],
            level_paddings=[2])

    # len(out_channels) - len(in_channels) must equal to len(level_strides)
    with pytest.raises(AssertionError):
        SSDNeck(
            in_channels=[8, 16],
            out_channels=[4, 16, 64],
            level_strides=[2, 2],
            level_paddings=[2, 2])

    # in_channels must be same with out_channels[:len(in_channels)]
    with pytest.raises(AssertionError):
        SSDNeck(
            in_channels=[8, 16],
            out_channels=[4, 16, 64],
            level_strides=[2],
            level_paddings=[2])

    ssd_neck = SSDNeck(
        in_channels=[4],
        out_channels=[4, 8, 16],
        level_strides=[2, 1],
        level_paddings=[1, 0])
    feats = (torch.rand(1, 4, 16, 16), )
    outs = ssd_neck(feats)
    assert outs[0].shape == (1, 4, 16, 16)
    assert outs[1].shape == (1, 8, 8, 8)
    assert outs[2].shape == (1, 16, 6, 6)

    # test SSD-Lite Neck
    ssd_neck = SSDNeck(
        in_channels=[4, 8],
        out_channels=[4, 8, 16],
        level_strides=[1],
        level_paddings=[1],
        l2_norm_scale=None,
        use_depthwise=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU6'))
    assert not hasattr(ssd_neck, 'l2_norm')

    from mmcv.cnn.bricks import DepthwiseSeparableConvModule
    assert isinstance(ssd_neck.extra_layers[0][-1],
                      DepthwiseSeparableConvModule)

    feats = (torch.rand(1, 4, 8, 8), torch.rand(1, 8, 8, 8))
    outs = ssd_neck(feats)
    assert outs[0].shape == (1, 4, 8, 8)
    assert outs[1].shape == (1, 8, 8, 8)
    assert outs[2].shape == (1, 16, 8, 8)


def test_yolox_pafpn():
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 24
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]
    neck = YOLOXPAFPN(in_channels=in_channels, out_channels=out_channels)
    outs = neck(feats)
    assert len(outs) == len(feats)
    for i in range(len(feats)):
        assert outs[i].shape[1] == out_channels
        assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # test depth-wise
    neck = YOLOXPAFPN(
        in_channels=in_channels, out_channels=out_channels, use_depthwise=True)

    from mmcv.cnn.bricks import DepthwiseSeparableConvModule
    assert isinstance(neck.downsamples[0], DepthwiseSeparableConvModule)

    outs = neck(feats)
    assert len(outs) == len(feats)
    for i in range(len(feats)):
        assert outs[i].shape[1] == out_channels
        assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)


def test_dyhead():
    s = 64
    in_channels = 8
    out_channels = 16
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    feats = [
        torch.rand(1, in_channels, feat_sizes[i], feat_sizes[i])
        for i in range(len(feat_sizes))
    ]
    neck = DyHead(
        in_channels=in_channels, out_channels=out_channels, num_blocks=3)
    outs = neck(feats)
    assert len(outs) == len(feats)
    for i in range(len(outs)):
        assert outs[i].shape[1] == out_channels
        assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    feat = torch.rand(1, 8, 4, 4)
    # input feat must be tuple or list
    with pytest.raises(AssertionError):
        neck(feat)


def test_fpg():
    # end_level=-1 is equal to end_level=3
    norm_cfg = dict(type='BN', requires_grad=True)
    FPG(in_channels=[8, 16, 32, 64],
        out_channels=8,
        inter_channels=8,
        num_outs=5,
        add_extra_convs=True,
        start_level=1,
        end_level=-1,
        stack_times=9,
        paths=['bu'] * 9,
        same_down_trans=None,
        same_up_trans=dict(
            type='conv',
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        across_lateral_trans=dict(
            type='conv',
            kernel_size=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        across_down_trans=dict(
            type='interpolation_conv',
            mode='nearest',
            kernel_size=3,
            norm_cfg=norm_cfg,
            order=('act', 'conv', 'norm'),
            inplace=False),
        across_up_trans=None,
        across_skip_trans=dict(
            type='conv',
            kernel_size=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        output_trans=dict(
            type='last_conv',
            kernel_size=3,
            order=('act', 'conv', 'norm'),
            inplace=False),
        norm_cfg=norm_cfg,
        skip_inds=[(0, 1, 2, 3), (0, 1, 2), (0, 1), (0, ), ()])
    FPG(in_channels=[8, 16, 32, 64],
        out_channels=8,
        inter_channels=8,
        num_outs=5,
        add_extra_convs=True,
        start_level=1,
        end_level=3,
        stack_times=9,
        paths=['bu'] * 9,
        same_down_trans=None,
        same_up_trans=dict(
            type='conv',
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        across_lateral_trans=dict(
            type='conv',
            kernel_size=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        across_down_trans=dict(
            type='interpolation_conv',
            mode='nearest',
            kernel_size=3,
            norm_cfg=norm_cfg,
            order=('act', 'conv', 'norm'),
            inplace=False),
        across_up_trans=None,
        across_skip_trans=dict(
            type='conv',
            kernel_size=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        output_trans=dict(
            type='last_conv',
            kernel_size=3,
            order=('act', 'conv', 'norm'),
            inplace=False),
        norm_cfg=norm_cfg,
        skip_inds=[(0, 1, 2, 3), (0, 1, 2), (0, 1), (0, ), ()])

    # `end_level` is larger than len(in_channels) - 1
    with pytest.raises(AssertionError):
        FPG(in_channels=[8, 16, 32, 64],
            out_channels=8,
            stack_times=9,
            paths=['bu'] * 9,
            start_level=1,
            end_level=4,
            num_outs=2,
            skip_inds=[(0, 1, 2, 3), (0, 1, 2), (0, 1), (0, ), ()])

    # `num_outs` is not equal to end_level - start_level + 1
    with pytest.raises(AssertionError):
        FPG(in_channels=[8, 16, 32, 64],
            out_channels=8,
            stack_times=9,
            paths=['bu'] * 9,
            start_level=1,
            end_level=2,
            num_outs=3,
            skip_inds=[(0, 1, 2, 3), (0, 1, 2), (0, 1), (0, ), ()])


def test_fpn_carafe():
    # end_level=-1 is equal to end_level=3
    FPN_CARAFE(
        in_channels=[8, 16, 32, 64],
        out_channels=8,
        start_level=0,
        end_level=3,
        num_outs=4)
    FPN_CARAFE(
        in_channels=[8, 16, 32, 64],
        out_channels=8,
        start_level=0,
        end_level=-1,
        num_outs=4)
    # `end_level` is larger than len(in_channels) - 1
    with pytest.raises(AssertionError):
        FPN_CARAFE(
            in_channels=[8, 16, 32, 64],
            out_channels=8,
            start_level=1,
            end_level=4,
            num_outs=2)

    # `num_outs` is not equal to end_level - start_level + 1
    with pytest.raises(AssertionError):
        FPN_CARAFE(
            in_channels=[8, 16, 32, 64],
            out_channels=8,
            start_level=1,
            end_level=2,
            num_outs=3)


def test_nas_fpn():
    # end_level=-1 is equal to end_level=3
    NASFPN(
        in_channels=[8, 16, 32, 64],
        out_channels=8,
        stack_times=9,
        start_level=0,
        end_level=3,
        num_outs=4)
    NASFPN(
        in_channels=[8, 16, 32, 64],
        out_channels=8,
        stack_times=9,
        start_level=0,
        end_level=-1,
        num_outs=4)
    # `end_level` is larger than len(in_channels) - 1
    with pytest.raises(AssertionError):
        NASFPN(
            in_channels=[8, 16, 32, 64],
            out_channels=8,
            stack_times=9,
            start_level=1,
            end_level=4,
            num_outs=2)

    # `num_outs` is not equal to end_level - start_level + 1
    with pytest.raises(AssertionError):
        NASFPN(
            in_channels=[8, 16, 32, 64],
            out_channels=8,
            stack_times=9,
            start_level=1,
            end_level=2,
            num_outs=3)


def test_nasfcos_fpn():
    # end_level=-1 is equal to end_level=3
    NASFCOS_FPN(
        in_channels=[8, 16, 32, 64],
        out_channels=8,
        start_level=0,
        end_level=3,
        num_outs=4)
    NASFCOS_FPN(
        in_channels=[8, 16, 32, 64],
        out_channels=8,
        start_level=0,
        end_level=-1,
        num_outs=4)

    # `end_level` is larger than len(in_channels) - 1
    with pytest.raises(AssertionError):
        NASFCOS_FPN(
            in_channels=[8, 16, 32, 64],
            out_channels=8,
            start_level=1,
            end_level=4,
            num_outs=2)

    # `num_outs` is not equal to end_level - start_level + 1
    with pytest.raises(AssertionError):
        NASFCOS_FPN(
            in_channels=[8, 16, 32, 64],
            out_channels=8,
            start_level=1,
            end_level=2,
            num_outs=3)


def test_ssh_neck():
    """Tests ssh."""
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = [16, 32, 64, 128]
    ssh_model = SSH(
        num_scales=4, in_channels=in_channels, out_channels=out_channels)

    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]
    outs = ssh_model(feats)
    assert len(outs) == len(feats)
    for i in range(len(outs)):
        assert outs[i].shape == \
            (1, out_channels[i], feat_sizes[i], feat_sizes[i])
