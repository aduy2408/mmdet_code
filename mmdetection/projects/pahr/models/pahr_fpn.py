from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.necks import FPN
from mmdet.registry import MODELS


@MODELS.register_module()
class PAHRFPN(FPN):
    """FPN with position-aware Haar recomposition on P3."""

    def __init__(self,
                 *args,
                 locator_channels: int = 64,
                 detail_channels: int = 64,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        channels = self.out_channels
        if locator_channels < 1 or detail_channels < 1:
            raise ValueError('PAHR hidden channel counts must be positive')

        self.locator = nn.Sequential(
            nn.Conv2d(4 * channels, locator_channels, 1),
            nn.Conv2d(
                locator_channels,
                locator_channels,
                3,
                padding=1,
                groups=locator_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(locator_channels, 12, 1),
        )
        self.detail_mixer = nn.Sequential(
            nn.Conv2d(4 * channels + 12, detail_channels, 1),
            nn.Conv2d(
                detail_channels,
                detail_channels,
                3,
                padding=1,
                groups=detail_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(detail_channels, 3 * channels, 1),
        )
        self.detail_scales = nn.Parameter(torch.zeros(3))
        nn.init.constant_(self.locator[-1].bias[:4], -2.19)

    @staticmethod
    def haar(feature: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply an orthonormal 2D Haar transform independently per channel."""
        height, width = feature.shape[-2:]
        if height % 2 or width % 2:
            raise ValueError(
                f'PAHR requires an even P3 size, got {height}x{width}')
        top_left = feature[..., 0::2, 0::2]
        top_right = feature[..., 0::2, 1::2]
        bottom_left = feature[..., 1::2, 0::2]
        bottom_right = feature[..., 1::2, 1::2]
        low = (top_left + top_right + bottom_left + bottom_right) * 0.5
        horizontal = (
            top_left - top_right + bottom_left - bottom_right) * 0.5
        vertical = (
            top_left + top_right - bottom_left - bottom_right) * 0.5
        diagonal = (
            top_left - top_right - bottom_left + bottom_right) * 0.5
        return low, horizontal, vertical, diagonal

    @staticmethod
    def inverse_haar(low: Tensor, horizontal: Tensor, vertical: Tensor,
                     diagonal: Tensor) -> Tensor:
        """Invert :meth:`haar` without interpolation."""
        top_left = (low + horizontal + vertical + diagonal) * 0.5
        top_right = (low - horizontal + vertical - diagonal) * 0.5
        bottom_left = (low + horizontal - vertical - diagonal) * 0.5
        bottom_right = (low - horizontal - vertical + diagonal) * 0.5
        output = low.new_empty(
            (*low.shape[:-2], low.shape[-2] * 2, low.shape[-1] * 2))
        output[..., 0::2, 0::2] = top_left
        output[..., 0::2, 1::2] = top_right
        output[..., 1::2, 0::2] = bottom_left
        output[..., 1::2, 1::2] = bottom_right
        return output

    def recompose(self, p3: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        bands = self.haar(p3)
        phase = F.pixel_shuffle(self.locator(torch.cat(bands, dim=1)), 2)
        position_logits = phase[:, :1]
        offsets = phase[:, 1:].sigmoid()
        position = position_logits.sigmoid()
        phase_context = F.pixel_unshuffle(
            torch.cat((position, offsets), dim=1), 2)

        residuals = self.detail_mixer(
            torch.cat((*bands, phase_context), dim=1)).chunk(3, dim=1)
        detail_corrections = tuple(
            scale.to(dtype=residual.dtype) * residual
            for residual, scale in zip(residuals, self.detail_scales))
        correction = self.inverse_haar(
            torch.zeros_like(bands[0]), *detail_corrections)
        output = p3 + position * correction
        aux = dict(
            position_logits=position_logits,
            offsets=offsets,
            detail_scales=self.detail_scales,
        )
        return output, aux

    def forward_with_aux(
            self, inputs: tuple[Tensor, ...]
    ) -> tuple[tuple[Tensor, ...], dict[str, Tensor]]:
        outputs = list(super().forward(inputs))
        outputs[0], aux = self.recompose(outputs[0])
        return tuple(outputs), aux

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        return self.forward_with_aux(inputs)[0]
