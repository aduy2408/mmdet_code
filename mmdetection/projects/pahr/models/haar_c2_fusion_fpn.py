from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.necks import FPN
from mmdet.registry import MODELS


@MODELS.register_module()
class HaarC2FusionFPN(FPN):
    """Fuse a lossless Haar packing of C2 into the baseline P3."""

    def __init__(self, *args, fusion_channels: int = 256, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.start_level != 1:
            raise ValueError('HaarC2FusionFPN requires start_level=1')
        if fusion_channels < 1:
            raise ValueError('fusion_channels must be positive')
        self.c2_lateral = nn.Conv2d(
            self.in_channels[0], self.out_channels, 1)
        merged_channels = 5 * self.out_channels
        self.fusion_mixer = nn.Sequential(
            nn.Conv2d(merged_channels, fusion_channels, 1),
            nn.Conv2d(
                fusion_channels,
                fusion_channels,
                3,
                padding=1,
                groups=fusion_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(fusion_channels, self.out_channels, 1))
        self._zero_fusion_output()

    def _zero_fusion_output(self) -> None:
        nn.init.zeros_(self.fusion_mixer[-1].weight)
        nn.init.zeros_(self.fusion_mixer[-1].bias)

    def init_weights(self) -> None:
        super().init_weights()
        self._zero_fusion_output()

    @staticmethod
    def haar(feature: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        height, width = feature.shape[-2:]
        if height % 2 or width % 2:
            raise ValueError(
                f'Haar C2 fusion requires an even C2 size, '
                f'got {height}x{width}')
        a = feature[..., 0::2, 0::2]
        b = feature[..., 0::2, 1::2]
        c = feature[..., 1::2, 0::2]
        d = feature[..., 1::2, 1::2]
        return (
            (a + b + c + d) * 0.5,
            (a - b + c - d) * 0.5,
            (a + b - c - d) * 0.5,
            (a - b - c + d) * 0.5)

    @staticmethod
    def inverse_haar(low: Tensor, horizontal: Tensor, vertical: Tensor,
                     diagonal: Tensor) -> Tensor:
        a = (low + horizontal + vertical + diagonal) * 0.5
        b = (low - horizontal + vertical - diagonal) * 0.5
        c = (low + horizontal - vertical - diagonal) * 0.5
        d = (low - horizontal - vertical + diagonal) * 0.5
        output = low.new_empty(
            (*low.shape[:-2], low.shape[-2] * 2, low.shape[-1] * 2))
        output[..., 0::2, 0::2] = a
        output[..., 0::2, 1::2] = b
        output[..., 1::2, 0::2] = c
        output[..., 1::2, 1::2] = d
        return output

    def forward_with_aux(
            self, inputs: tuple[Tensor, ...]
    ) -> tuple[tuple[Tensor, ...], dict[str, Tensor]]:
        outputs = list(super().forward(inputs))
        bands = self.haar(self.c2_lateral(inputs[0]))
        if bands[0].shape[-2:] != outputs[0].shape[-2:]:
            raise ValueError(
                'Haar-packed C2 must align with P3, got '
                f'{bands[0].shape[-2:]} and {outputs[0].shape[-2:]}')
        correction = self.fusion_mixer(
            torch.cat((outputs[0], *bands), dim=1))
        outputs[0] = outputs[0] + correction
        baseline_rms = (outputs[0] - correction).square().mean().sqrt()
        aux = dict(
            band_rms=torch.stack(
                [band.square().mean().sqrt() for band in bands]),
            correction_rms=correction.square().mean().sqrt(),
            correction_ratio=(
                correction.norm()
                / (outputs[0] - correction).norm().clamp_min(1e-12)),
            baseline_p3_rms=baseline_rms)
        return tuple(outputs), aux

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        return self.forward_with_aux(inputs)[0]
