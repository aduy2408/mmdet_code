import torch
from mmengine.structures import InstanceData

from mmdet.models.necks import (DualIrreducibilityHIT,
                                MaskedCenterConv2d)
from mmdet.structures import DetDataSample


def _sample(bboxes):
    sample = DetDataSample()
    sample.gt_instances = InstanceData(
        bboxes=torch.tensor(bboxes, dtype=torch.float32),
        labels=torch.zeros(len(bboxes), dtype=torch.long))
    return sample


def test_masked_center_conv_cannot_read_center():
    conv = MaskedCenterConv2d(2)
    first = torch.zeros(1, 2, 5, 5)
    second = first.clone()
    second[:, :, 2, 2] = 10
    assert torch.allclose(
        conv(first)[:, :, 2, 2], conv(second)[:, :, 2, 2])


def test_hit_shape_harmonic_map_and_gradients():
    module = DualIrreducibilityHIT(channels=8, alpha_init=1e-3)
    module.train()
    feature = torch.randn(2, 8, 8, 8, requires_grad=True)
    output = module(feature)
    assert output.shape == feature.shape
    assert 0 <= module.alpha.item() <= 0.002

    high = torch.ones_like(feature)
    low = torch.zeros_like(feature)
    assert module.hard_map(high, low).max() < 1e-5
    assert module.hard_map(high, high).mean() > 0.9

    losses = module.auxiliary_losses([
        _sample([[12, 12, 28, 28]]),
        _sample([]),
    ])
    total = output.mean() + sum(losses.values())
    total.backward()
    assert torch.isfinite(feature.grad).all()
    assert module.offset_head.weight.grad is not None


def test_gaussian_splat_conserves_mass_and_handles_boundaries():
    module = DualIrreducibilityHIT(channels=1)
    source = torch.ones(1, 1, 3, 3)
    offsets = torch.zeros(1, 2, 3, 3, requires_grad=True)
    sigma = torch.ones(1, 1, 3, 3, requires_grad=True)
    output = module._gaussian_splat(source, offsets, sigma)
    assert torch.allclose(output.sum(), source.sum(), atol=1e-5)
    output.square().sum().backward()
    assert offsets.grad is not None
    assert sigma.grad is not None


def test_hit_offset_targets_tiny_boundary_and_overlap():
    module = DualIrreducibilityHIT(channels=4, stride=8, topk=4)
    module.train()
    module(torch.randn(1, 4, 4, 4))
    prediction, target = module._offset_targets([
        _sample([[0, 0, 2, 2], [0, 0, 24, 24], [30, 30, 32, 32]])
    ])
    assert 0 < prediction.shape[0] <= 9
    assert prediction.shape == target.shape
    assert torch.isfinite(target).all()
