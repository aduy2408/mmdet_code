import pytest
import torch

from mmdet.models.backbones import DeviationResNet, PhaseDownsample
from mmdet.registry import MODELS


@pytest.mark.parametrize('mode', PhaseDownsample.MODES)
def test_phase_downsample_shape_and_gradient(mode):
    module = PhaseDownsample(8, 16, mode)
    x = torch.randn(2, 8, 12, 10, requires_grad=True)
    output = module(x)
    assert output.shape == (2, 16, 6, 5)
    output.mean().backward()
    assert x.grad is not None


def test_phase_downsample_rejects_odd_shapes():
    module = PhaseDownsample(8, 16, 'deviation')
    with pytest.raises(ValueError, match='even spatial dimensions'):
        module(torch.randn(1, 8, 11, 10))


def test_deviation_branch_starts_disabled():
    module = PhaseDownsample(8, 16, 'deviation')
    x = torch.randn(1, 8, 12, 10)
    assert torch.equal(module(x), module.context(
        torch.nn.functional.pixel_unshuffle(x, 2).view(
        1, 8, 4, 6, 5).mean(dim=2)))


def test_legacy_projection_weight_loading():
    module = PhaseDownsample(8, 16, 'deviation')
    weight = torch.randn_like(module.context.weight)
    module.load_state_dict({'weight': weight}, strict=False)
    assert torch.equal(module.context.weight, weight)


@pytest.mark.parametrize('style', ('pytorch', 'caffe'))
def test_deviation_resnet_forward_and_registry(style):
    model = MODELS.build(
        dict(
            type='DeviationResNet',
            depth=50,
            base_channels=8,
            style=style,
            phase_mode='deviation',
            phase_stages=(1, 2)))
    outputs = model(torch.randn(1, 3, 64, 64))
    assert [output.shape[-2:] for output in outputs] == [
        (16, 16), (8, 8), (4, 4), (2, 2)
    ]
    assert isinstance(model, DeviationResNet)
