# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from mmcv.cnn import build_norm_layer
from torch import Tensor, nn

from mmdet.registry import MODELS

from .resnet import ResNet


class PhaseDownsample(nn.Module):
    """Stride-2 projection that keeps the four sampling phases explicit."""

    MODES = ('pixel_unshuffle', 'context', 'deviation')

    def __init__(self, in_channels: int, out_channels: int, mode: str) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'Unsupported phase downsample mode: {mode}')
        self.in_channels = in_channels
        self.mode = mode
        context_in = in_channels * 4 if mode == 'pixel_unshuffle' else in_channels
        self.context = nn.Conv2d(context_in, out_channels, 1, bias=False)
        if mode == 'deviation':
            self.deviation = nn.Sequential(
                nn.Conv2d(in_channels * 4, out_channels, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
            )
            # Start as a context-only model and let optimization admit residuals.
            self.deviation_scale = nn.Parameter(torch.zeros(()))

    def _phases(self, x: Tensor) -> Tensor:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                'PhaseDownsample requires even spatial dimensions; pad model '
                f'inputs to a multiple of 32, got {tuple(x.shape[-2:])}.')
        return nn.functional.pixel_unshuffle(x, 2)

    def forward(self, x: Tensor) -> Tensor:
        phases = self._phases(x)
        if self.mode == 'pixel_unshuffle':
            return self.context(phases)

        batch, _, height, width = phases.shape
        phases = phases.view(batch, self.in_channels, 4, height, width)
        mean = phases.mean(dim=2)
        context = self.context(mean)
        if self.mode == 'context':
            return context

        residual = phases - mean.unsqueeze(2)
        residual = residual.reshape(batch, self.in_channels * 4, height, width)
        return context + self.deviation_scale * self.deviation(residual)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Caffe-style ResNet downsamples with 1x1 convolutions. Reuse those
        # weights for the mean/context path after wrapping the convolution.
        legacy_key = prefix + 'weight'
        context_key = prefix + 'context.weight'
        if (legacy_key in state_dict
                and state_dict[legacy_key].shape == self.context.weight.shape):
            state_dict[context_key] = state_dict.pop(legacy_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


@MODELS.register_module()
class DeviationResNet(ResNet):
    """ResNet with phase-aware downsampling at selected residual stages.

    ``phase_stages`` uses zero-based ResNet stage indices. The default replaces
    C2->C3 and C3->C4 while leaving the stem and C4->C5 unchanged.
    """

    def __init__(self,
                 *args,
                 phase_mode: str = 'deviation',
                 phase_stages: Sequence[int] = (1, 2),
                 **kwargs) -> None:
        self.phase_mode = phase_mode
        self.phase_stages = tuple(phase_stages)
        self._building_stage = 0
        if len(set(self.phase_stages)) != len(self.phase_stages):
            raise ValueError('phase_stages must not contain duplicates')
        if any(stage not in range(4) for stage in self.phase_stages):
            raise ValueError('phase_stages must contain indices from 0 to 3')
        super().__init__(*args, **kwargs)

    def make_res_layer(self, **kwargs):
        stage = self._building_stage
        self._building_stage += 1
        layer = super().make_res_layer(**kwargs)
        if stage not in self.phase_stages or kwargs['stride'] != 2:
            return layer

        block = layer[0]
        if block.style == 'pytorch':
            old = block.conv2
            block.conv2 = PhaseDownsample(old.in_channels, old.out_channels,
                                          self.phase_mode)
        else:
            old = block.conv1
            block.conv1 = PhaseDownsample(old.in_channels, old.out_channels,
                                          self.phase_mode)

        shortcut_conv = block.downsample[-2]
        shortcut_norm = build_norm_layer(
            kwargs['norm_cfg'], shortcut_conv.out_channels)[1]
        block.downsample = nn.Sequential(
            PhaseDownsample(shortcut_conv.in_channels,
                            shortcut_conv.out_channels, self.phase_mode),
            shortcut_norm)
        return layer
