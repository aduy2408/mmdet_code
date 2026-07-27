from pathlib import Path

import pytest
import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import FCOSHead
from mmdet.structures import DetDataSample
from projects.rcfn_ltmr import LTMRFCOSHead, PGRCFNFCOS, RCFNFPN
from projects.rcfn_ltmr.tools.diagnose_ltmr import position_statistics


def _head():
    return LTMRFCOSHead(
        num_classes=2,
        in_channels=1,
        feat_channels=1,
        stacked_convs=1,
        norm_cfg=None,
        strides=(8, 16, 32, 64, 128),
        regress_ranges=((-1, 64), (64, 128), (128, 256),
                        (256, 512), (512, 1e8)))


def _instances(boxes, labels=None):
    result = InstanceData()
    result.bboxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if labels is not None:
        result.labels = torch.tensor(labels, dtype=torch.long)
    return result


def _sample(boxes, img_shape=(32, 32), ignored=(), **metainfo):
    sample = DetDataSample()
    sample.set_metainfo(dict(img_shape=img_shape, **metainfo))
    sample.gt_instances = _instances(boxes, [0] * len(boxes))
    sample.ignored_instances = _instances(ignored)
    return sample


def _position_detector():
    detector = PGRCFNFCOS.__new__(PGRCFNFCOS)
    nn.Module.__init__(detector)
    detector.tiny_max_sqrt_area = 16.0
    detector.gaussian_alpha = 1.0
    detector.gaussian_sigma_min = 1.0
    detector.position_positive_weight = 4.0
    detector.loss_pos_weight = 1.0
    detector.position_stride = 8
    return detector


def test_ring_stats_and_identity_initialization():
    neck = RCFNFPN(
        in_channels=[1, 2, 4, 8],
        out_channels=4,
        num_outs=5,
        start_level=1,
        add_extra_convs='on_output')
    flat = torch.ones(1, 4, 5, 5, requires_grad=True)
    mean, var = neck.ring_stats(flat)
    assert torch.equal(mean, flat)
    assert torch.all(var == neck.eps)

    inputs = tuple(
        torch.randn(1, channels, size, size)
        for channels, size in zip((1, 2, 4, 8), (32, 16, 8, 4)))
    baseline = super(RCFNFPN, neck).forward(inputs)
    output = neck(inputs)
    assert torch.equal(output[0], baseline[0])
    assert all(torch.equal(a, b) for a, b in zip(output[1:], baseline[1:]))
    output[0].sum().backward()
    assert torch.isfinite(neck.gamma.grad).all()


def test_ring_stats_are_fp32_and_finite_for_large_fp16_values():
    neck = RCFNFPN(
        in_channels=[1, 2, 4, 8],
        out_channels=4,
        num_outs=5,
        start_level=1,
        add_extra_convs='on_output')
    feature = torch.full((1, 4, 5, 5), 60000, dtype=torch.float16)
    feature[:, :, 2, 2] = 59968
    mean, var = neck.ring_stats(feature)
    deviation = neck.standardized_deviation(feature)
    assert mean.dtype == torch.float32
    assert var.dtype == torch.float32
    assert deviation.dtype == feature.dtype
    assert torch.isfinite(mean).all()
    assert torch.isfinite(var).all()
    assert torch.isfinite(deviation).all()


def test_tiny_selection_uses_original_coordinates_and_diagnostics():
    detector = _position_detector()
    upscaled = _sample(
        [[0, 0, 24, 24]], img_shape=(64, 64), ori_shape=(32, 32),
        scale_factor=(2.0, 2.0))
    downscaled = _sample(
        [[0, 0, 12, 12]], img_shape=(32, 32), ori_shape=(64, 64),
        scale_factor=(0.5, 0.5))
    anisotropic = _sample(
        [[0, 0, 24, 8]], img_shape=(16, 64), ori_shape=(32, 32),
        scale_factor=(2.0, 0.5))
    derived = _sample(
        [[0, 0, 24, 24]], img_shape=(64, 64), ori_shape=(32, 32))
    assert detector.tiny_mask(
        upscaled.gt_instances.bboxes, upscaled.metainfo).item()
    assert not detector.tiny_mask(
        downscaled.gt_instances.bboxes, downscaled.metainfo).item()
    assert detector.tiny_mask(
        anisotropic.gt_instances.bboxes, anisotropic.metainfo).item()
    assert detector.tiny_mask(
        derived.gt_instances.bboxes, derived.metainfo).item()

    position = torch.full((2, 1, 8, 8), 0.75)
    target, _ = detector.position_targets(
        position, [upscaled, downscaled])
    assert target[0].sum() > 0
    assert target[1].sum() == 0
    _, stats = position_statistics(
        detector, position, [upscaled, downscaled])
    assert len(stats['center_values']) == 1


def test_tiny_selection_rejects_invalid_scale():
    detector = _position_detector()
    boxes = torch.tensor([[0, 0, 8, 8]], dtype=torch.float32)
    with pytest.raises(ValueError, match='positive'):
        detector.tiny_mask(boxes, {'scale_factor': (0.0, 1.0)})


def test_position_targets_tiny_adaptive_max_padding_and_ignore():
    detector = _position_detector()
    position = torch.full((2, 1, 5, 5), 0.5)
    samples = [
        _sample(
            [[4, 4, 12, 12], [0, 0, 16, 8], [0, 0, 32, 32]],
            img_shape=(32, 24), ignored=[[16, 0, 24, 8]]),
        _sample([], img_shape=(16, 16)),
    ]
    target, valid = detector.position_targets(position, samples)

    assert target[0, 0, 1, 1] == 1
    assert target[0, 0, 0, 1] > target[0, 0, 1, 3]
    assert target.max() <= 1
    assert target[1].sum() == 0
    assert not valid[0, 0, 0, 2]
    assert not valid[0, 0, :, 3:].any()
    assert not valid[0, 0, 4].any()
    assert valid[1, 0, :2, :2].all()
    loss = detector.position_loss(position.requires_grad_(), samples)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert torch.isfinite(position.grad).all()


def test_rcfn_gate_modes_and_position_gradients():
    inputs = tuple(
        torch.randn(1, channels, size, size)
        for channels, size in zip((1, 2, 4, 8), (32, 16, 8, 4)))
    common = dict(
        in_channels=[1, 2, 4, 8], out_channels=4, num_outs=5,
        start_level=1, add_extra_convs='on_output', gamma_init=1.0,
        position_channels=2)
    auxiliary = RCFNFPN(**common, gate_mode='none')
    position = RCFNFPN(**common, gate_mode='position')
    contrast = RCFNFPN(
        **common, gate_mode='contrast_position', predict_contrast=True)
    position.load_state_dict(auxiliary.state_dict(), strict=False)
    contrast.load_state_dict(auxiliary.state_dict(), strict=False)

    aux_out, heatmap, _ = auxiliary.forward_with_position(inputs)
    pos_out, pos_heatmap, _ = position.forward_with_position(inputs)
    ch_out, ch_heatmap, confidence = contrast.forward_with_position(inputs)
    baseline = super(RCFNFPN, auxiliary).forward(inputs)
    enhancement = aux_out[0] - baseline[0]
    assert torch.allclose(
        pos_out[0] - baseline[0], enhancement * pos_heatmap,
        rtol=1e-4, atol=1e-5)
    assert torch.allclose(
        ch_out[0] - baseline[0], enhancement * ch_heatmap * confidence,
        rtol=1e-4, atol=1e-5)
    assert all(torch.equal(a, b) for a, b in zip(aux_out[1:], pos_out[1:]))
    heatmap.mean().backward()
    assert torch.isfinite(auxiliary.position_out_conv.weight.grad).all()


def test_auxiliary_mode_matches_r2_features():
    inputs = tuple(
        torch.randn(1, channels, size, size)
        for channels, size in zip((1, 2, 4, 8), (32, 16, 8, 4)))
    neck = RCFNFPN(
        in_channels=[1, 2, 4, 8], out_channels=4, num_outs=5,
        start_level=1, add_extra_convs='on_output', gate_mode='none')
    expected = neck(inputs)
    actual, heatmap, contrast = neck.forward_with_position(inputs)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert heatmap.shape == (1, 1, 16, 16)
    assert contrast is None


def test_pg_rcfn_detector_loss_and_inference_without_gt():
    model = PGRCFNFCOS(
        backbone=dict(
            type='ResNet', depth=18, num_stages=4,
            out_indices=(0, 1, 2, 3), norm_cfg=dict(type='BN'),
            norm_eval=True, style='pytorch', init_cfg=None),
        neck=dict(
            type='RCFNFPN', in_channels=[64, 128, 256, 512],
            out_channels=32, start_level=1, add_extra_convs='on_output',
            num_outs=5, position_channels=8, gate_mode='position'),
        bbox_head=dict(
            type='FCOSHead', num_classes=1, in_channels=32,
            feat_channels=32, stacked_convs=1, norm_cfg=None,
            strides=[8, 16, 32, 64, 128]),
        test_cfg=ConfigDict(
            nms_pre=100, score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6), max_per_img=10))
    inputs = torch.randn(1, 3, 64, 64)
    train_sample = _sample([[8, 8, 16, 16]], img_shape=(64, 64))
    train_sample.set_metainfo(dict(
        img_shape=(64, 64), ori_shape=(64, 64), scale_factor=(1.0, 1.0)))
    model.train()
    losses = model.loss(inputs, [train_sample])
    assert 'loss_pos' in losses and torch.isfinite(losses['loss_pos'])

    predict_sample = _sample([], img_shape=(64, 64))
    predict_sample.set_metainfo(dict(
        img_shape=(64, 64), ori_shape=(64, 64), scale_factor=(1.0, 1.0)))
    model.eval()
    with torch.inference_mode():
        predictions = model.predict(inputs, [predict_sample])
    assert len(predictions) == 1
    assert 'pred_instances' in predictions[0]


@pytest.mark.parametrize(
    ('name', 'detector_type', 'gate_mode'),
    [
        ('fcos_baseline.py', 'FCOS', 'FPN'),
        ('fcos_r2.py', 'FCOS', 'none'),
        ('fcos_pg_aux.py', 'PGRCFNFCOS', 'none'),
        ('fcos_pg_h.py', 'PGRCFNFCOS', 'position'),
        ('fcos_pg_ch.py', 'PGRCFNFCOS', 'contrast_position'),
    ])
def test_tod_configs_are_standalone_levir(
        name, detector_type, gate_mode):
    config_dir = Path(__file__).parents[1] / 'configs'
    cfg = Config.fromfile(config_dir / name)
    assert cfg.model.type == detector_type
    if gate_mode == 'FPN':
        assert cfg.model.neck.type == gate_mode
    else:
        assert cfg.model.neck.gate_mode == gate_mode
    assert cfg.model.bbox_head.num_classes == 1
    assert cfg.train_dataloader.dataset.ann_file == (
        'data/levir_ship_coco/annotations/train.json')
    assert cfg.val_dataloader.dataset.ann_file == (
        'data/levir_ship_coco/annotations/val.json')
    assert cfg.test_dataloader.dataset.ann_file == (
        'data/levir_ship_coco/annotations/test.json')
    assert cfg.train_dataloader.dataset.data_prefix.img == (
        '../LevirShipData/All Images/')
    resize = next(
        item for item in cfg.train_dataloader.dataset.pipeline
        if item.type == 'Resize')
    assert resize.scale == (512, 512)
    assert cfg.val_evaluator.type == 'CocoMetric'
    assert cfg.test_evaluator.type == 'CocoMetric'


def test_assignment_background_overlap_and_empty_gt():
    head = _head()
    sizes = [(4, 4), (2, 2), (1, 1), (1, 1), (1, 1)]
    points = head.prior_generator.grid_priors(sizes, device='cpu')
    counts = [len(item) for item in points]
    all_points = torch.cat(points)
    ranges = torch.cat([
        point.new_tensor(head.regress_ranges[level])[None].expand_as(point)
        for level, point in enumerate(points)
    ])

    empty = _instances([], [])
    assert (head.assigned_gt_inds(
        all_points, empty, ranges, counts) == -1).all()

    overlapping = _instances([[0, 0, 32, 32], [8, 8, 24, 24]], [0, 1])
    assigned = head.assigned_gt_inds(
        all_points, overlapping, ranges, counts)
    center = ((all_points[:, 0] == 12) & (all_points[:, 1] == 12)).nonzero()
    assert assigned[center].item() == 1
    assert (assigned >= 0).any() and (assigned < 0).any()


def test_ltmr_filters_other_gt_ignore_and_padding():
    head = _head()
    points = head.prior_generator.grid_priors(
        [(5, 5), (1, 1), (1, 1), (1, 1), (1, 1)],
        device='cpu')[0]
    logits = torch.zeros(1, 2, 5, 5, requires_grad=True)
    gt = _instances([[12, 12, 20, 20], [28, 12, 36, 20]], [0, 0])
    assignments = [torch.full((25,), -1, dtype=torch.long)]
    assignments[0][12] = 0
    assignments[0][14] = 1
    ignored = [_instances([[0, 0, 8, 40]])]
    loss = head.local_tiny_margin_loss(
        logits, points, assignments, [gt],
        [{'img_shape': (32, 32, 3)}], ignored)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_ltmr_zero_when_no_valid_tiny_pair():
    head = _head()
    points = head.prior_generator.grid_priors(
        [(2, 2), (1, 1), (1, 1), (1, 1), (1, 1)],
        device='cpu')[0]
    logits = torch.randn(1, 2, 2, 2, requires_grad=True)
    large_gt = _instances([[0, 0, 64, 64]], [0])
    loss = head.local_tiny_margin_loss(
        logits, points, [torch.zeros(4, dtype=torch.long)], [large_gt],
        [{'img_shape': (16, 16, 3)}])
    assert loss.item() == 0
    loss.backward()
    assert logits.grad is not None


def test_ltmr_inference_matches_fcos():
    kwargs = dict(
        num_classes=2,
        in_channels=1,
        feat_channels=1,
        stacked_convs=1,
        norm_cfg=None)
    baseline = FCOSHead(**kwargs).eval()
    ltmr = LTMRFCOSHead(**kwargs).eval()
    ltmr.load_state_dict(baseline.state_dict(), strict=False)
    features = tuple(
        torch.randn(1, 1, 64 // stride[0], 64 // stride[1])
        for stride in baseline.prior_generator.strides)
    with torch.inference_mode():
        expected = baseline(features)
        actual = ltmr(features)
    for expected_group, actual_group in zip(expected, actual):
        for expected_tensor, actual_tensor in zip(
                expected_group, actual_group):
            assert torch.equal(expected_tensor, actual_tensor)
