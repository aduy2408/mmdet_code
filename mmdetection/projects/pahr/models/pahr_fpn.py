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
                 gate_power: float = 1.0,
                 correction_gate_floor: float = 0.0,
                 detach_position_gate: bool = False,
                 guide_channels: int = 0,
                 use_output_gate: bool = True,
                 correction_gain: float = 1.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        channels = self.out_channels
        if locator_channels < 1 or detail_channels < 1:
            raise ValueError('PAHR hidden channel counts must be positive')
        if gate_power <= 0:
            raise ValueError('gate_power must be positive')
        if not 0 <= correction_gate_floor <= 1:
            raise ValueError('correction_gate_floor must be in [0, 1]')
        if guide_channels < 0:
            raise ValueError('guide_channels must be non-negative')
        if correction_gain < 0:
            raise ValueError('correction_gain must be non-negative')
        self.gate_power = float(gate_power)
        self.correction_gate_floor = float(correction_gate_floor)
        self.detach_position_gate = bool(detach_position_gate)
        self.guide_channels = int(guide_channels)
        self.use_output_gate = bool(use_output_gate)
        self.correction_gain = float(correction_gain)
        packed_guide_channels = 16 * self.guide_channels
        self.guide_projection = (
            nn.Sequential(
                nn.Conv2d(self.in_channels[0], self.guide_channels, 1),
                nn.SiLU(inplace=True))
            if self.guide_channels else None)

        self.locator = nn.Sequential(
            nn.Conv2d(
                4 * channels + packed_guide_channels,
                locator_channels,
                1),
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
            nn.Conv2d(
                4 * channels + packed_guide_channels + 12,
                detail_channels,
                1),
            nn.Conv2d(
                detail_channels,
                detail_channels,
                3,
                padding=1,
                groups=detail_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(detail_channels, 3 * channels, 1),
        )
        self._init_pahr_outputs()

    def _init_pahr_outputs(self) -> None:
        """Keep PAHR identity-initialized without blocking output gradients."""
        nn.init.zeros_(self.detail_mixer[-1].weight)
        nn.init.zeros_(self.detail_mixer[-1].bias)
        with torch.no_grad():
            self.locator[-1].bias[:4].fill_(-2.19)

    def init_weights(self) -> None:
        super().init_weights()
        # FPN's init_cfg initializes every Conv2d, including PAHR layers.
        self._init_pahr_outputs()

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

    def guide_features(self, c2: Tensor | None,
                       band_shape: tuple[int, int]) -> Tensor | None:
        if self.guide_projection is None:
            return None
        if c2 is None:
            raise ValueError('C2 guidance is enabled but C2 was not provided')
        height, width = c2.shape[-2:]
        if height % 4 or width % 4:
            raise ValueError(
                f'PAHR C2 guidance requires size divisible by 4, '
                f'got {height}x{width}')
        guidance = F.pixel_unshuffle(self.guide_projection(c2), 4)
        if guidance.shape[-2:] != band_shape:
            raise ValueError(
                'PAHR C2 guidance must align with Haar bands, got '
                f'{guidance.shape[-2:]} and {band_shape}')
        return guidance

    def recompose(self, p3: Tensor,
                  c2: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        bands = self.haar(p3)
        guidance = self.guide_features(c2, bands[0].shape[-2:])
        band_context = (*bands,) if guidance is None else (*bands, guidance)
        phase = F.pixel_shuffle(
            self.locator(torch.cat(band_context, dim=1)), 2)
        position_logits = phase[:, :1]
        offsets = phase[:, 1:].sigmoid()
        position = position_logits.sigmoid()
        phase_gate = position.pow(self.gate_power)
        if self.detach_position_gate:
            phase_gate = phase_gate.detach()
        correction_gate = self.correction_gate_floor + (
            1 - self.correction_gate_floor) * phase_gate
        phase_context = F.pixel_unshuffle(
            torch.cat((correction_gate, correction_gate * offsets), dim=1), 2)

        residuals = self.detail_mixer(
            torch.cat((*band_context, phase_context), dim=1)).chunk(3, dim=1)
        correction = self.inverse_haar(
            torch.zeros_like(bands[0]), *residuals)
        output_gate = correction_gate if self.use_output_gate else 1.0
        applied_correction = self.correction_gain * output_gate * correction
        output = p3 + applied_correction
        aux = dict(
            position_logits=position_logits,
            offsets=offsets,
            correction_gate=correction_gate,
            phase_gate=phase_gate,
            guidance_rms=(
                p3.new_zeros(()) if guidance is None
                else guidance.square().mean().sqrt()),
            raw_correction_rms=correction.square().mean().sqrt(),
            applied_correction_rms=applied_correction.square().mean().sqrt(),
        )
        return output, aux

    def forward_with_aux(
            self, inputs: tuple[Tensor, ...]
    ) -> tuple[tuple[Tensor, ...], dict[str, Tensor]]:
        outputs = list(super().forward(inputs))
        outputs[0], aux = self.recompose(outputs[0], inputs[0])
        return tuple(outputs), aux

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        return self.forward_with_aux(inputs)[0]
