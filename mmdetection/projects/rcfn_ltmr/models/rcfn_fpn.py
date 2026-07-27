import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.necks import FPN
from mmdet.registry import MODELS


@MODELS.register_module()
class RCFNFPN(FPN):
    """FPN with standardized local-background enhancement on P3."""

    def __init__(self, *args, eps: float = 1e-4,
                 gamma_init: float = 0.0, position_channels: int = 64,
                 gate_mode: str = 'none',
                 predict_contrast: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if gate_mode not in {'none', 'position', 'contrast_position'}:
            raise ValueError(f'Unsupported gate_mode: {gate_mode}')
        if gate_mode == 'contrast_position' and not predict_contrast:
            raise ValueError(
                'contrast_position gate requires predict_contrast=True')
        self.eps = eps
        self.gate_mode = gate_mode
        self.predict_contrast = predict_contrast
        channels = self.out_channels
        self.dw_conv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels)
        self.out_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.full((channels,), gamma_init))
        if position_channels > 0:
            self.p3_position_conv = nn.Conv2d(
                channels, position_channels, 1)
            self.p4_position_conv = nn.Conv2d(
                channels, position_channels, 1)
            position_in_channels = 2 * position_channels
            self.position_dw_conv = nn.Conv2d(
                position_in_channels, position_in_channels, 3, padding=1,
                groups=position_in_channels)
            self.position_out_conv = nn.Conv2d(position_in_channels, 1, 1)
        else:
            self.p3_position_conv = None
            self.p4_position_conv = None
            self.position_dw_conv = None
            self.position_out_conv = None
        self.contrast_conv = (
            nn.Conv2d(channels, 1, 1) if predict_contrast else None)
        if gate_mode != 'none' and self.position_out_conv is None:
            raise ValueError('A gated RCFNFPN requires position_channels > 0')

    def ring_stats(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        """Return FP32 mean and variance of the 8 neighboring P3 cells."""
        feature_fp32 = feature.float()
        padded = F.pad(feature_fp32, (1, 1, 1, 1), mode='replicate')
        mean_3 = F.avg_pool2d(padded, 3, stride=1)
        square_mean_3 = F.avg_pool2d(padded.square(), 3, stride=1)
        ring_mean = (9 * mean_3 - feature_fp32) / 8
        ring_square_mean = (
            9 * square_mean_3 - feature_fp32.square()) / 8
        ring_var = (ring_square_mean - ring_mean.square()).clamp_min(self.eps)
        return ring_mean, ring_var

    def standardized_deviation(self, feature: Tensor) -> Tensor:
        """Standardize in FP32, then restore the feature dtype."""
        mean, var = self.ring_stats(feature)
        deviation = (feature.float() - mean) * torch.rsqrt(var)
        return deviation.to(feature.dtype)

    def forward_with_position(
            self, inputs: tuple[Tensor]
    ) -> tuple[tuple[Tensor, ...], Tensor, Tensor | None]:
        """Return enhanced FPN features and the reliability maps."""
        if self.position_out_conv is None:
            raise RuntimeError(
                'forward_with_position requires position_channels > 0')
        outs = list(super().forward(inputs))
        deviation = self.standardized_deviation(outs[0])
        enhancement = self.out_conv(F.silu(self.dw_conv(deviation)))
        p4 = F.interpolate(
            self.p4_position_conv(outs[1]), size=outs[0].shape[-2:],
            mode='nearest')
        position_features = torch.cat(
            (self.p3_position_conv(outs[0]), p4), dim=1)
        position = torch.sigmoid(self.position_out_conv(
            F.silu(self.position_dw_conv(position_features))))
        contrast = (
            torch.sigmoid(self.contrast_conv(deviation.abs()))
            if self.contrast_conv is not None else None)
        gate = 1
        if self.gate_mode == 'position':
            gate = position
        elif self.gate_mode == 'contrast_position':
            gate = position * contrast
        outs[0] = (
            outs[0]
            + self.gamma.view(1, -1, 1, 1) * gate * enhancement
        )
        return tuple(outs), position, contrast

    def forward(self, inputs: tuple[Tensor]) -> tuple[Tensor, ...]:
        if self.gate_mode == 'none':
            outs = list(super().forward(inputs))
            deviation = self.standardized_deviation(outs[0])
            enhancement = self.out_conv(F.silu(self.dw_conv(deviation)))
            outs[0] = (
                outs[0]
                + self.gamma.view(1, -1, 1, 1) * enhancement
            )
            return tuple(outs)
        return self.forward_with_position(inputs)[0]
