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
                 gamma_init: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps
        channels = self.out_channels
        self.dw_conv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels)
        self.out_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.full((channels,), gamma_init))

    def ring_stats(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        """Return mean and variance of the 8 neighboring P3 cells."""
        padded = F.pad(feature, (1, 1, 1, 1), mode='replicate')
        mean_3 = F.avg_pool2d(padded, 3, stride=1)
        square_mean_3 = F.avg_pool2d(padded.square(), 3, stride=1)
        ring_mean = (9 * mean_3 - feature) / 8
        ring_square_mean = (9 * square_mean_3 - feature.square()) / 8
        ring_var = (ring_square_mean - ring_mean.square()).clamp_min(self.eps)
        return ring_mean, ring_var

    def forward(self, inputs: tuple[Tensor]) -> tuple[Tensor, ...]:
        outs = list(super().forward(inputs))
        mean, var = self.ring_stats(outs[0])
        deviation = (outs[0] - mean) * torch.rsqrt(var)
        enhancement = self.out_conv(F.silu(self.dw_conv(deviation)))
        outs[0] = (
            outs[0]
            + self.gamma.view(1, -1, 1, 1) * enhancement
        )
        return tuple(outs)
