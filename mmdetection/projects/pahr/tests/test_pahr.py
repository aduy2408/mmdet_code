import math
from pathlib import Path

import pytest
import torch
from mmengine.config import Config, ConfigDict
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample
from projects.pahr import PAHRFCOS, PAHRFPN
from train_all_haar import VARIANTS, scale_schedule


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


def detector(
        use_phase_shift=False,
        norm_on_bbox=False,
        guide_channels=0,
        use_tiny_measurement=False):
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
            detail_channels=8,
            guide_channels=guide_channels,
            use_output_gate=not guide_channels),
        use_tiny_measurement=use_tiny_measurement,
        bbox_head=dict(
            type='FCOSHead',
            num_classes=1,
            in_channels=32,
            feat_channels=32,
            stacked_convs=1,
            norm_cfg=None,
            norm_on_bbox=norm_on_bbox,
            strides=[8, 16, 32, 64, 128]),
        use_phase_shift=use_phase_shift,
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
    module.init_weights()
    assert torch.count_nonzero(module.detail_mixer[-1].weight) == 0
    assert torch.count_nonzero(module.detail_mixer[-1].bias) == 0
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

    module.detail_mixer[-1].bias.data.fill_(0.1)
    changed, _ = module.forward_with_aux(inputs)
    assert not torch.equal(changed[0], baseline[0])
    assert all(torch.equal(left, right)
               for left, right in zip(changed[1:], baseline[1:]))


def test_output_conv_gets_gradient_at_identity_init():
    module = neck()
    p3 = torch.randn(1, 4, 8, 8, requires_grad=True)
    output, aux = module.recompose(p3)
    (output.square().mean() + aux['position_logits'].square().mean()).backward()
    output_conv = module.detail_mixer[-1]
    assert output_conv.weight.grad is not None
    assert torch.isfinite(output_conv.weight.grad).all()
    assert output_conv.weight.grad.abs().sum() > 0
    assert module.locator[-1].weight.grad is not None

    with torch.no_grad():
        output_conv.weight.add_(output_conv.weight.grad, alpha=-0.01)
    module.zero_grad(set_to_none=True)
    changed, _ = module.recompose(p3.detach())
    assert not torch.equal(changed, p3.detach())
    changed.square().mean().backward()
    assert module.detail_mixer[0].weight.grad is not None
    assert torch.isfinite(module.detail_mixer[0].weight.grad).all()


def test_offset_context_is_position_gated():
    module = neck()
    p3 = torch.randn(1, 4, 8, 8)
    captured = {}

    def capture(_, inputs):
        captured['input'] = inputs[0].detach()

    handle = module.detail_mixer.register_forward_pre_hook(capture)
    _, aux = module.recompose(p3)
    handle.remove()
    context = captured['input'][:, -12:]
    packed = torch.nn.functional.pixel_unshuffle(
        torch.cat((
            aux['correction_gate'],
            aux['correction_gate'] * aux['offsets']), dim=1), 2)
    assert torch.equal(context, packed)


def test_v3_gates_are_exact_and_detach_position_gradient():
    module = neck(
        gate_power=0.5,
        correction_gate_floor=0.05,
        detach_position_gate=True)
    with torch.no_grad():
        module.locator[-1].weight.zero_()
        module.locator[-1].bias[:4].fill_(torch.logit(torch.tensor(0.25)))
        module.detail_mixer[-1].weight.fill_(0.1)
    p3 = torch.randn(1, 4, 8, 8, requires_grad=True)
    output, aux = module.recompose(p3)
    aux['position_logits'].retain_grad()
    assert torch.allclose(aux['phase_gate'], torch.full_like(
        aux['phase_gate'], 0.5))
    assert torch.allclose(aux['correction_gate'], torch.full_like(
        aux['correction_gate'], 0.525))
    output.square().mean().backward()
    assert aux['position_logits'].grad is None
    assert module.locator[-1].weight.grad is not None


def test_v4_c2_guidance_alignment_effect_and_gradient():
    module = neck(
        guide_channels=2,
        use_output_gate=False,
        correction_gain=1.0)
    module.init_weights()
    p3 = torch.randn(1, 4, 8, 8, requires_grad=True)
    c2 = torch.randn(1, 1, 16, 16, requires_grad=True)
    identity, aux = module.recompose(p3, c2)
    assert torch.equal(identity, p3)
    assert aux['guidance_rms'] > 0

    with torch.no_grad():
        module.detail_mixer[-1].weight.fill_(0.1)
    first, aux = module.recompose(p3, c2)
    second, _ = module.recompose(p3, c2 + 1)
    assert not torch.equal(first, second)
    assert torch.allclose(
        aux['raw_correction_rms'], aux['applied_correction_rms'])
    first.square().mean().backward()
    assert c2.grad is not None and c2.grad.abs().sum() > 0
    assert module.guide_projection[0].weight.grad is not None
    assert module.detail_mixer[-1].weight.grad is not None


def test_v4_guidance_validation_and_disabled_path():
    guided = neck(guide_channels=2)
    with pytest.raises(ValueError, match='was not provided'):
        guided.recompose(torch.randn(1, 4, 8, 8))
    with pytest.raises(ValueError, match='divisible by 4'):
        guided.recompose(
            torch.randn(1, 4, 8, 8), torch.randn(1, 1, 15, 16))
    with pytest.raises(ValueError, match='align with Haar bands'):
        guided.recompose(
            torch.randn(1, 4, 8, 8), torch.randn(1, 1, 20, 20))

    unguided = neck(guide_channels=0)
    p3 = torch.randn(1, 4, 8, 8)
    without_c2, _ = unguided.recompose(p3)
    with_c2, _ = unguided.recompose(p3, torch.randn(1, 1, 16, 16))
    assert torch.equal(without_c2, with_c2)


def test_scaled_schedule():
    cfg = Config(
        dict(param_scheduler=[
            dict(type='ConstantLR', begin=0, end=500),
            dict(
                type='MultiStepLR',
                begin=0,
                end=12,
                milestones=[8, 11])
        ]))
    scale_schedule(cfg, 20)
    assert cfg.param_scheduler[1].end == 20
    assert cfg.param_scheduler[1].milestones == [13, 18]
    scale_schedule(cfg, 40)
    assert cfg.param_scheduler[1].end == 40
    assert cfg.param_scheduler[1].milestones == [27, 37]
    assert VARIANTS['haar_v3_gate_lr10_768']['detail_lr_mult'] == 10.0
    assert VARIANTS['haar_v4_c2_768']['guide_channels'] == 16
    assert not VARIANTS['haar_v4_ungated_768']['use_output_gate']
    assert VARIANTS['haar_v5_measure_768']['use_tiny_measurement']


def test_phase_shift_algebra_and_level_isolation():
    model = detector(use_phase_shift=True)
    bbox_preds = [
        torch.full((1, 4, 2, 2), 10.0),
        torch.full((1, 4, 1, 1), 20.0),
    ]
    aux = dict(
        position_logits=torch.zeros(1, 1, 2, 2),
        offsets=torch.cat((
            torch.ones(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2)), dim=1))
    adjusted = model.phase_adjust_bbox_preds(bbox_preds, aux)
    expected = torch.tensor([8.0, 12.0, 12.0, 8.0]).view(1, 4, 1, 1)
    assert torch.equal(adjusted[0], expected.expand_as(adjusted[0]))
    assert torch.equal(adjusted[1], bbox_preds[1])

    model.use_phase_shift = False
    unshifted = model.phase_adjust_bbox_preds(bbox_preds, aux)
    assert torch.equal(unshifted[0], bbox_preds[0])

    normalized = detector(use_phase_shift=True, norm_on_bbox=True)
    normalized.train()
    adjusted = normalized.phase_adjust_bbox_preds(bbox_preds, aux)
    expected = torch.tensor([9.75, 10.25, 10.25, 9.75]).view(
        1, 4, 1, 1)
    assert torch.equal(adjusted[0], expected.expand_as(adjusted[0]))


@pytest.mark.parametrize('size', [64, 96])
def test_p3_sizes_for_512_and_768_are_supported(size):
    module = neck()
    p3 = torch.randn(1, 4, size, size)
    output, aux = module.recompose(p3)
    assert output.shape == p3.shape
    assert aux['position_logits'].shape[-2:] == (size, size)


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


@pytest.mark.parametrize('use_phase_shift', [False, True])
def test_detector_loss_predict_and_config(use_phase_shift):
    model = detector(use_phase_shift=use_phase_shift)
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


def test_detector_c2_guidance_loss_and_predict():
    model = detector(use_phase_shift=True, guide_channels=4)
    inputs = torch.randn(1, 3, 64, 64)
    data_sample = sample(
        [[8, 8, 16, 16]],
        img_shape=(64, 64),
        ori_shape=(64, 64),
        scale_factor=(1.0, 1.0))
    model.train()
    losses = model.loss(inputs, [data_sample])
    assert all(torch.isfinite(loss) for loss in losses.values())
    model.eval()
    with torch.inference_mode():
        predictions = model.predict(inputs, [data_sample])
    assert len(predictions) == 1


@pytest.mark.parametrize('size', [64, 96])
def test_measurement_shapes_neutral_refinement_and_gradients(size):
    model = detector(use_tiny_measurement=True)
    inputs = torch.randn(1, 3, size, size)
    maps = model.measurement_maps(inputs)
    assert maps['measurement_center_logits'].shape[-2:] == (
        size // 2, size // 2)
    assert maps['measurement_phase_logits'].shape == (
        1, 16, size // 8, size // 8)
    assert maps['measurement_log_sizes'].shape == (
        1, 2, size // 8, size // 8)

    bbox_preds = [
        torch.full((1, 4, size // 8, size // 8), 10.0),
        torch.full((1, 4, size // 16, size // 16), 20.0),
    ]
    neutral = model.measurement_adjust_bbox_preds(bbox_preds, maps)
    assert torch.equal(neutral[0], bbox_preds[0])
    assert torch.equal(neutral[1], bbox_preds[1])
    with torch.no_grad():
        model.measurement_refine_scale.fill_(10)
        maps['measurement_center_logits'].fill_(100)
        maps['measurement_phase_logits'].fill_(-100)
        maps['measurement_phase_logits'][:, 15].fill_(100)
        maps['measurement_log_sizes'].fill_(math.log(10))
    refined = model.measurement_adjust_bbox_preds(bbox_preds, maps)
    expected_shift = float(3 * torch.sigmoid(torch.tensor(2.0)))
    expected = torch.tensor([
        10 - expected_shift,
        10 - expected_shift,
        10 + expected_shift,
        10 + expected_shift,
    ]).view(1, 4, 1, 1)
    assert torch.allclose(
        refined[0], expected.expand_as(refined[0]), atol=1e-4)
    assert torch.equal(refined[1], bbox_preds[1])

    data_sample = sample(
        [[8, 8, 16, 16]],
        img_shape=(size, size),
        ori_shape=(size, size),
        scale_factor=(1.0, 1.0))
    losses = model.measurement_losses(maps, [data_sample])
    sum(losses.values()).backward()
    assert model.measurement_stem[0].weight.grad is not None
    assert model.measurement_center.weight.grad is not None
    assert model.measurement_phase.weight.grad is not None
    assert model.measurement_size.weight.grad is not None


def test_detector_measurement_loss_predict_and_tensor_forward():
    model = detector(use_tiny_measurement=True)
    inputs = torch.randn(1, 3, 64, 64)
    data_sample = sample(
        [[8, 8, 16, 16]],
        img_shape=(64, 64),
        ori_shape=(64, 64),
        scale_factor=(1.0, 1.0))
    model.train()
    losses = model.loss(inputs, [data_sample])
    assert {
        'loss_measure_center',
        'loss_measure_phase',
        'loss_measure_size',
    } <= losses.keys()
    assert all(torch.isfinite(loss) for loss in losses.values())
    tensor_outputs = model._forward(inputs, [data_sample])
    assert len(tensor_outputs) == 3
    model.eval()
    with torch.inference_mode():
        predictions = model.predict(inputs, [data_sample])
    assert len(predictions) == 1


def test_measurement_targets_collision_ignore_empty_and_phase():
    model = detector(use_tiny_measurement=True)
    maps = model.measurement_maps(torch.randn(3, 3, 64, 64))
    samples = [
        sample(
            [[8, 8, 16, 16], [10, 10, 14, 14]],
            img_shape=(64, 64),
            ori_shape=(64, 64),
            scale_factor=(1.0, 1.0),
            ignored=[[24, 24, 32, 32]]),
        sample([[60, 60, 64, 64]], img_shape=(64, 64)),
        sample([], img_shape=(48, 48)),
    ]
    center, center_valid, phase, sizes, size_valid = (
        model.measurement_targets(maps, samples))
    assert center.max() > 0
    assert size_valid[0].sum() == 1
    assert phase[0, 1, 1] == 10
    assert not center_valid[0, 0, 12:16, 12:16].any()
    assert phase[0, 3, 3] == -1
    assert size_valid[1].sum() == 1
    assert not center_valid[2, :, 24:].any()
    losses = model.measurement_losses(maps, samples)
    assert all(torch.isfinite(loss) for loss in losses.values())
    empty_maps = model.measurement_maps(torch.randn(1, 3, 64, 64))
    empty_losses = model.measurement_losses(
        empty_maps, [sample([], img_shape=(64, 64))])
    assert all(torch.isfinite(loss) for loss in empty_losses.values())
