from pathlib import Path

import pytest
import torch
from mmengine.config import Config, ConfigDict
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample
from projects.pahr import PAHRFCOS, PAHRFPN


def sample(boxes, img_shape=(32, 32), ignored=(), **metainfo):
    data_sample = DetDataSample()
    data_sample.set_metainfo(dict(img_shape=img_shape, **metainfo))
    data_sample.gt_instances = InstanceData(
        bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        labels=torch.zeros(len(boxes), dtype=torch.long))
    data_sample.ignored_instances = InstanceData(
        bboxes=torch.tensor(ignored, dtype=torch.float32).reshape(-1, 4))
    return data_sample


def neck(**kwargs):
    return PAHRFPN(
        in_channels=[1, 2, 4, 8],
        out_channels=4,
        num_outs=5,
        start_level=1,
        add_extra_convs='on_output',
        locator_channels=2,
        detail_channels=2,
        **kwargs)


def detector():
    return PAHRFCOS(
        backbone=dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            norm_cfg=dict(type='BN'),
            norm_eval=True,
            style='pytorch',
            init_cfg=None),
        neck=dict(
            type='PAHRFPN',
            in_channels=[64, 128, 256, 512],
            out_channels=32,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5,
            locator_channels=8,
            detail_channels=8),
        bbox_head=dict(
            type='FCOSHead',
            num_classes=1,
            in_channels=32,
            feat_channels=32,
            stacked_convs=1,
            norm_cfg=None,
            strides=[8, 16, 32, 64, 128]),
        test_cfg=ConfigDict(
            nms_pre=100,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=10))


def test_haar_round_trip_signed_and_odd_validation():
    feature = torch.randn(2, 3, 8, 10, requires_grad=True)
    bands = PAHRFPN.haar(feature)
    assert any((band < 0).any() for band in bands[1:])
    restored = PAHRFPN.inverse_haar(*bands)
    assert torch.allclose(restored, feature, atol=1e-6)
    restored.sum().backward()
    assert torch.isfinite(feature.grad).all()
    with pytest.raises(ValueError, match='even P3 size'):
        PAHRFPN.haar(torch.randn(1, 3, 7, 8))


def test_zero_init_is_exact_fpn_and_only_p3_changes():
    module = neck()
    inputs = tuple(
        torch.randn(1, channels, size, size)
        for channels, size in zip((1, 2, 4, 8), (32, 16, 8, 4)))
    baseline = super(PAHRFPN, module).forward(inputs)
    output, aux = module.forward_with_aux(inputs)
    assert all(torch.equal(left, right)
               for left, right in zip(output, baseline))
    assert aux['position_logits'].shape == (1, 1, 16, 16)
    assert aux['offsets'].shape == (1, 2, 16, 16)
    assert ((aux['offsets'] >= 0) & (aux['offsets'] <= 1)).all()

    module.detail_scales.data.fill_(1)
    changed, _ = module.forward_with_aux(inputs)
    assert not torch.equal(changed[0], baseline[0])
    assert all(torch.equal(left, right)
               for left, right in zip(changed[1:], baseline[1:]))


def test_scale_then_mixer_gradients_are_finite():
    module = neck()
    p3 = torch.randn(1, 4, 8, 8, requires_grad=True)
    output, aux = module.recompose(p3)
    (output.square().mean() + aux['position_logits'].square().mean()).backward()
    assert torch.isfinite(module.detail_scales.grad).all()
    assert module.locator[-1].weight.grad is not None

    module.zero_grad(set_to_none=True)
    module.detail_scales.data.fill_(0.1)
    module.recompose(p3.detach())[0].square().mean().backward()
    assert module.detail_mixer[-1].weight.grad is not None
    assert torch.isfinite(module.detail_mixer[-1].weight.grad).all()


def test_targets_resize_padding_ignore_collision_and_empty():
    model = detector()
    logits = torch.zeros(3, 1, 5, 5)
    samples = [
        sample(
            [[4, 4, 12, 12], [6, 6, 10, 10], [0, 0, 40, 40]],
            img_shape=(24, 24),
            ori_shape=(12, 12),
            scale_factor=(2.0, 2.0),
            ignored=[[16, 0, 24, 8]]),
        sample(
            [[0, 0, 12, 12]],
            img_shape=(24, 24),
            ori_shape=(48, 48),
            scale_factor=(0.5, 0.5)),
        sample([], img_shape=(16, 16)),
    ]
    target, offset_target, valid, offset_valid = model.auxiliary_targets(
        logits, samples)
    assert target[0].max() == 1
    assert offset_valid[0].sum() == 1
    assert torch.allclose(
        offset_target[0, :, 1, 1], torch.tensor([0.0, 0.0]))
    assert not valid[0, 0, 0, 2]
    assert not valid[0, 0, 3:].any()
    assert target[1].sum() == 0
    assert target[2].sum() == 0
    losses = model.auxiliary_losses(
        dict(position_logits=logits, offsets=torch.full((3, 2, 5, 5), .5)),
        samples)
    assert all(torch.isfinite(loss) for loss in losses.values())


def test_detector_loss_predict_and_config():
    model = detector()
    inputs = torch.randn(1, 3, 64, 64)
    train_sample = sample(
        [[8, 8, 16, 16]],
        img_shape=(64, 64),
        ori_shape=(64, 64),
        scale_factor=(1.0, 1.0))
    model.train()
    losses = model.loss(inputs, [train_sample])
    assert {'loss_pos', 'loss_offset'} <= losses.keys()
    assert torch.isfinite(losses['loss_pos'])
    assert torch.isfinite(losses['loss_offset'])

    model.eval()
    predict_sample = sample(
        [],
        img_shape=(64, 64),
        ori_shape=(64, 64),
        scale_factor=(1.0, 1.0))
    with torch.inference_mode():
        predictions = model.predict(inputs, [predict_sample])
    assert len(predictions) == 1
    assert 'pred_instances' in predictions[0]

    config_path = Path(__file__).parents[1] / 'configs' / 'fcos_pahr.py'
    config = Config.fromfile(config_path)
    assert config.model.type == 'PAHRFCOS'
    assert config.model.neck.type == 'PAHRFPN'
    assert config.model.bbox_head.num_classes == 80
