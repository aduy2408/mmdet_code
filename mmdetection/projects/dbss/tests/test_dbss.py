from pathlib import Path

import pytest
import torch
from mmengine.config import Config, ConfigDict
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample
from .. import DBSSFCOS, DBSSFPN


def neck(**kwargs) -> DBSSFPN:
    return DBSSFPN(
        in_channels=[2, 4, 8, 16],
        out_channels=4,
        num_outs=5,
        start_level=1,
        add_extra_convs='on_output',
        embed_channels=3,
        candidate_grid=(2, 2),
        shortlist_size=4,
        num_bases=2,
        hidden_channels=4,
        **kwargs)


def backbone_inputs(batch_size=2):
    return tuple(
        torch.randn(batch_size, channels, size, size)
        for channels, size in zip((2, 4, 8, 16), (32, 16, 8, 4)))


def sample(boxes, img_shape=(32, 32)):
    data_sample = DetDataSample()
    data_sample.set_metainfo(dict(img_shape=img_shape))
    data_sample.gt_instances = InstanceData(
        bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        labels=torch.zeros(len(boxes), dtype=torch.long))
    return data_sample


def detector(projection_mode='ridge') -> DBSSFCOS:
    return DBSSFCOS(
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
            type='DBSSFPN',
            in_channels=[64, 128, 256, 512],
            out_channels=16,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5,
            embed_channels=8,
            candidate_grid=(2, 2),
            shortlist_size=4,
            num_bases=2,
            hidden_channels=8,
            projection_mode=projection_mode),
        bbox_head=dict(
            type='FCOSHead',
            num_classes=1,
            in_channels=16,
            feat_channels=16,
            stacked_convs=1,
            norm_cfg=None,
            strides=[8, 16, 32, 64, 128]),
        test_cfg=ConfigDict(
            nms_pre=100,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=10))


def test_identity_level_isolation_and_bounded_displacement():
    module = neck()
    module.init_weights()
    inputs = backbone_inputs()
    baseline = super(DBSSFPN, module).forward(inputs)
    output, aux = module.forward_with_aux(inputs, [(16, 16), (12, 10)])
    assert all(torch.equal(left, right)
               for left, right in zip(output, baseline))
    assert torch.count_nonzero(aux['displacement']) == 0

    with torch.no_grad():
        module.direction[-1].weight.fill_(0.1)
    changed, aux = module.forward_with_aux(inputs, [(16, 16), (12, 10)])
    assert not torch.equal(changed[0], baseline[0])
    assert all(torch.equal(left, right)
               for left, right in zip(changed[1:], baseline[1:]))
    displacement_norm = aux['displacement'].norm(dim=1, keepdim=True)
    bound = module.gamma_max * aux['feature_scale']
    assert torch.all(displacement_norm <= bound + 1e-6)


def test_valid_crop_excludes_padding_from_candidates():
    module = neck()
    p3 = torch.randn(2, 4, 8, 8)
    padded = p3.clone()
    padded[1, :, 5:, :] = 1e4
    padded[1, :, :, 6:] = 1e4
    _, original = module.enhance(p3, [(8, 8), (5, 6)])
    _, changed = module.enhance(padded, [(8, 8), (5, 6)])
    assert torch.equal(
        original['representativeness'][1],
        changed['representativeness'][1])
    assert torch.equal(
        original['selected_candidates_p3'][1],
        changed['selected_candidates_p3'][1])


def test_diversity_selection_avoids_duplicate_top_candidates():
    module = neck(
        diversity_beta=1.0, basis_similarity_threshold=0.9)
    candidates = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]])
    scores = torch.tensor([1.0, 0.99, 0.8, 0.7])
    selected = module._select_bases(scores, candidates)
    assert int(selected[0]) == 0
    assert int(selected[1]) != 1
    assert selected.numel() == module.num_bases


def test_diversity_fallback_always_returns_requested_bases():
    module = DBSSFPN(
        in_channels=[2, 4, 8, 16],
        out_channels=4,
        num_outs=5,
        start_level=1,
        add_extra_convs='on_output',
        embed_channels=3,
        candidate_grid=(3, 3),
        shortlist_size=9,
        num_bases=8,
        hidden_channels=4,
        diversity_beta=1.0,
        basis_similarity_threshold=0.9)
    candidates = torch.tensor([[1.0, 0.0, 0.0]]).repeat(9, 1)
    selected = module._select_bases(
        torch.linspace(1, 0, 9), candidates)
    assert selected.numel() == 8
    assert selected.unique().numel() == 8


@pytest.mark.parametrize('mode', ['ridge', 'softmax'])
def test_projection_is_finite_and_differentiable(mode):
    module = neck(projection_mode=mode)
    tokens = torch.randn(7, 3, requires_grad=True)
    basis = torch.ones(2, 3, requires_grad=True)
    projected = module._project(tokens, basis)
    assert projected.shape == tokens.shape
    assert torch.isfinite(projected).all()
    projected.square().mean().backward()
    assert torch.isfinite(tokens.grad).all()
    assert torch.isfinite(basis.grad).all()


def test_two_steps_reach_dbss_branches():
    module = neck()
    module.init_weights()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    inputs = backbone_inputs(batch_size=1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output, _ = module.forward_with_aux(inputs)
        output[0].square().mean().backward()
        optimizer.step()
    assert module.direction[0].weight.grad is not None
    assert module.embedding.weight.grad is not None
    assert module.magnitude[0].weight.grad is not None
    assert torch.isfinite(module.embedding.weight.grad).all()


def test_haar_only_changes_magnitude_context():
    plain = neck()
    haar = neck(use_haar_reliability=True)
    p3 = torch.randn(1, 4, 9, 7)
    _, plain_aux = plain.enhance(p3)
    _, haar_aux = haar.enhance(p3)
    assert 'haar_reliability_rms' not in plain_aux
    assert torch.isfinite(haar_aux['haar_reliability_rms'])
    assert plain_aux['selected_indices'].shape == haar_aux[
        'selected_indices'].shape


def test_center_sampling_and_separation_empty_gt():
    model = detector()
    feature = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    sampled = model._sample_centers(
        feature, torch.tensor([[0., 0., 1., 1.], [30., 30., 32., 32.]]),
        (32, 32))
    assert sampled.shape == (2, 1)
    assert torch.isfinite(sampled).all()

    p3 = torch.randn(1, 16, 4, 4, requires_grad=True)
    objective = model.separation_objective(
        dict(
            pre_p3=p3,
            post_p3=p3,
            selected_candidates_p3=torch.randn(1, 2, 16)),
        [sample([])])
    assert objective['loss_dbss_sep'] == 0
    objective['loss_dbss_sep'].backward()
    assert p3.grad is not None


def test_improvement_loss_is_active_at_identity_initialization():
    model = detector()
    pre_p3 = torch.randn(1, 16, 4, 4)
    post_p3 = pre_p3.clone().requires_grad_()
    objective = model.separation_objective(
        dict(
            pre_p3=pre_p3,
            post_p3=post_p3,
            selected_candidates_p3=torch.randn(1, 2, 16)),
        [sample([[7, 7, 9, 9]])])
    expected = model.improvement_margin * model.loss_sep_weight
    assert torch.allclose(
        objective['loss_dbss_sep'],
        objective['loss_dbss_sep'].new_tensor(expected),
        atol=1e-6)
    assert objective['dbss_active_ratio'] == 1
    objective['loss_dbss_sep'].backward()
    assert torch.isfinite(post_p3.grad).all()
    assert post_p3.grad.abs().sum() > 0


@pytest.mark.parametrize('mode', ['ridge', 'softmax'])
def test_detector_loss_and_forward_smoke(mode):
    model = detector(mode)
    model.init_weights()
    model.train()
    images = torch.randn(1, 3, 64, 64)
    samples = [sample([[7, 7, 9, 9]], img_shape=(64, 64))]
    losses = model.loss(images, samples)
    assert 'loss_dbss_sep' in losses
    assert torch.isfinite(losses['loss_dbss_sep'])
    outputs = model._forward(images, samples)
    assert len(outputs) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA AMP')
def test_ridge_autocast_uses_fp32_solve_and_restores_dtype(monkeypatch):
    module = neck().cuda()
    seen_dtypes = []
    original_solve = torch.linalg.solve

    def checked_solve(left, right):
        seen_dtypes.append((left.dtype, right.dtype))
        return original_solve(left, right)

    monkeypatch.setattr(torch.linalg, 'solve', checked_solve)
    tokens = torch.randn(7, 3, device='cuda', dtype=torch.float16)
    bases = torch.randn(2, 3, device='cuda', dtype=torch.float16)
    with torch.autocast('cuda', dtype=torch.float16):
        output = module._project(tokens, bases)
    assert seen_dtypes == [(torch.float32, torch.float32)]
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


def test_configs_load():
    config_dir = Path(__file__).parents[1] / 'configs'
    ridge = Config.fromfile(config_dir / 'fcos_dbss_ridge.py')
    gamma06 = Config.fromfile(config_dir / 'fcos_dbss_ridge_gamma06.py')
    gamma10 = Config.fromfile(config_dir / 'fcos_dbss_ridge_gamma10.py')
    softmax = Config.fromfile(config_dir / 'fcos_dbss_softmax.py')
    haar = Config.fromfile(config_dir / 'fcos_dbss_ridge_haar.py')
    assert ridge.model.neck.projection_mode == 'ridge'
    assert ridge.model.neck.gamma_max == 0.3
    assert gamma06.model.neck.gamma_max == 0.6
    assert gamma10.model.neck.gamma_max == 1.0
    assert ridge.model.improvement_margin == 0.03
    assert ridge.model.loss_sep_weight == 0.5
    direction_options = (
        ridge.optim_wrapper.paramwise_cfg.custom_keys['neck.direction.2'])
    assert direction_options.lr_mult == 10
    assert direction_options.decay_mult == 0
    assert softmax.model.neck.projection_mode == 'softmax'
    assert haar.model.neck.use_haar_reliability
