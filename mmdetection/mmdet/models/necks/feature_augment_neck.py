# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType


@MODELS.register_module()
class MorphologicalEnhancement(BaseModule):
    """Enhance local extrema with a zero-initialized residual."""

    _valid_modes = {'positive', 'negative', 'both', 'conv'}

    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 mode: str = 'both',
                 gamma_init: float = 0.0,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be an odd integer greater than 1')
        if mode not in self._valid_modes:
            raise ValueError(
                f'mode must be one of {sorted(self._valid_modes)}, got {mode!r}')
        self.kernel_size = int(kernel_size)
        self.mode = mode
        size = 3 if mode == 'conv' else 1
        self.mixer = nn.Conv2d(channels, channels, size, padding=size // 2)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _dilate(self, x: Tensor) -> Tensor:
        return F.max_pool2d(
            x, self.kernel_size, stride=1, padding=self.kernel_size // 2)

    def _erode(self, x: Tensor) -> Tensor:
        return -self._dilate(-x)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == 'conv':
            residual = self.mixer(x)
        else:
            residual = None
            if self.mode in {'positive', 'both'}:
                opening = self._dilate(self._erode(x))
                residual = F.relu(x - opening)
            if self.mode in {'negative', 'both'}:
                closing = self._erode(self._dilate(x))
                negative = F.relu(closing - x)
                residual = negative if residual is None else residual + negative
            residual = self.mixer(residual)
        return x + self.gamma.to(dtype=x.dtype) * residual


class UpBlock(nn.Module):
    """Small reconstruction upsample block used by DGFE."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


@MODELS.register_module()
class FeatureDGFE(BaseModule):
    """Image-guided feature enhancement ported from the YOLO reference."""

    def __init__(self,
                 channels: int,
                 reduction: int = 8,
                 threshold_init: float = 0.0156862,
                 sharpness: float = 10.0,
                 alpha_init: float = 1e-3,
                 alpha_max: float = 1.0,
                 recon_ratio: float = 0.5,
                 upsample_steps: int = 2,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        upsample_steps = max(int(upsample_steps), 1)
        hidden_channels = max(channels // max(int(reduction), 1), 8)

        up_blocks = []
        in_channels = channels
        out_channels = max(int(channels * float(recon_ratio)), 8)
        for _ in range(upsample_steps):
            up_blocks.append(UpBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels = max(out_channels // 2, 8)

        self.upsample = nn.Sequential(*up_blocks)
        self.reconstruct = nn.Sequential(
            nn.Conv2d(in_channels, 3, 3, padding=1), nn.Sigmoid())
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1),
        )
        self.threshold = nn.Parameter(torch.tensor(float(threshold_init)))
        self.sharpness = float(sharpness)
        self.alpha_max = max(float(alpha_max), 0.0)
        p = max(min(float(alpha_init) / max(self.alpha_max, 1e-12),
                    1.0 - 1e-6), 1e-6)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(p)))
        self.last_aux: dict[str, Tensor] | None = None

    @property
    def alpha(self) -> Tensor:
        return torch.sigmoid(self.alpha_logit) * self.alpha_max

    def forward(self, x: Tensor, batch_inputs: Tensor) -> Tensor:
        recon = self.reconstruct(self.upsample(x))
        if recon.shape[-2:] != batch_inputs.shape[-2:]:
            recon = F.interpolate(
                recon,
                size=batch_inputs.shape[-2:],
                mode='bilinear',
                align_corners=False)

        img = batch_inputs
        img_min = img.amin(dim=(2, 3), keepdim=True)
        img_max = img.amax(dim=(2, 3), keepdim=True)
        img = (img - img_min) / (img_max - img_min).clamp(min=1e-6)

        diff = (recon - img).abs().mean(dim=1, keepdim=True)
        logits_img = self.sharpness * (
            diff - self.threshold.to(device=diff.device, dtype=diff.dtype))
        logits = F.interpolate(
            logits_img, size=x.shape[-2:], mode='bilinear',
            align_corners=False)
        spatial_gate = 1.0 + torch.sigmoid(logits)

        avg_gate = self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
        max_gate = self.channel_mlp(F.adaptive_max_pool2d(x, 1))
        channel_gate = torch.sigmoid(avg_gate + max_gate)
        alpha = self.alpha.to(device=x.device, dtype=x.dtype)
        out = x * (1.0 + alpha * (channel_gate * spatial_gate - 1.0))
        self.last_aux = dict(recon=recon, spatial_logits=logits,
                             spatial_gate=spatial_gate,
                             alpha=alpha.reshape(1)) if self.training else None
        return out


@MODELS.register_module()
class AdversarialPerturbationInjection(BaseModule):
    """Train-time feature perturbation module for one neck output level."""

    def __init__(self,
                 channels: int,
                 rho: float = 0.02,
                 api_weight: float = 0.25,
                 target_mode: str = 'foreground',
                 eps: float = 1e-6,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.rho = max(float(rho), 0.0)
        self.api_weight = max(float(api_weight), 0.0)
        self.target_mode = str(target_mode)
        self.eps = max(float(eps), 1e-12)
        self.aux_head = nn.Conv2d(channels, 1, 1)
        self.mode = 'off'
        self.captured: Tensor | None = None
        self.perturbation: Tensor | None = None

    @property
    def current_rho(self) -> float:
        return self.rho

    @property
    def current_api_weight(self) -> float:
        return self.api_weight

    def clear_state(self) -> None:
        self.mode = 'off'
        self.captured = None
        self.perturbation = None

    def capture(self) -> None:
        self.clear_state()
        self.mode = 'capture'

    def perturb(self) -> None:
        self.mode = 'perturb'

    def set_perturbation_from_grad(self, grad: Tensor | None) -> bool:
        if grad is None or self.current_rho == 0 or self.current_api_weight == 0:
            self.perturbation = None
            return False
        grad_f = grad.detach().float()
        norm = grad_f.flatten(1).norm(p=2, dim=1).clamp(
            min=self.eps).view(-1, 1, 1, 1)
        perturbation = grad_f / norm * self.current_rho
        if not torch.isfinite(perturbation).all():
            self.perturbation = None
            return False
        self.perturbation = perturbation.to(device=grad.device,
                                            dtype=grad.dtype)
        return True

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        if self.mode == 'capture':
            self.captured = x
            if x.requires_grad:
                x.retain_grad()
            return x
        if self.mode == 'perturb' and self.perturbation is not None:
            return x + self.perturbation.to(device=x.device, dtype=x.dtype)
        return x

    def auxiliary_loss(self,
                       target: Tensor,
                       feature: Tensor | None = None) -> Tensor:
        feature = self.captured if feature is None else feature
        if feature is None:
            raise RuntimeError('API auxiliary loss requires a captured feature.')
        logits = self.aux_head(feature)
        target = target.to(device=logits.device, dtype=logits.dtype)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(
                target, size=logits.shape[-2:], mode='nearest')
        return F.binary_cross_entropy_with_logits(logits, target)


@MODELS.register_module()
class FeatureAugmentNeck(BaseModule):
    """Wrap a normal neck and apply optional DGFE/API modules to output levels."""

    needs_batch_inputs = True

    def __init__(self,
                 base_neck: ConfigType,
                 levels: Sequence[int] = (0, ),
                 out_channels: int | Sequence[int] | None = None,
                 morphology: OptConfigType = None,
                 dgfe: OptConfigType = None,
                 api: OptConfigType = None,
        init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.base_neck = self._build_base_neck(base_neck)
        self.levels = tuple(int(level) for level in levels)
        channels = self._resolve_channels(out_channels)
        self.morphology_modules = nn.ModuleDict()
        self.dgfe_modules = nn.ModuleDict()
        self.api_modules_by_level = nn.ModuleDict()

        for level in self.levels:
            level_channels = (
                channels[level] if isinstance(channels, list) else channels)
            if morphology is not None:
                cfg = dict(morphology)
                cfg.setdefault('type', 'MorphologicalEnhancement')
                cfg.setdefault('channels', level_channels)
                self.morphology_modules[str(level)] = MODELS.build(cfg)
            if dgfe is not None:
                cfg = dict(dgfe)
                cfg.setdefault('type', 'FeatureDGFE')
                cfg.setdefault('channels', level_channels)
                self.dgfe_modules[str(level)] = MODELS.build(cfg)
            if api is not None:
                cfg = dict(api)
                cfg.setdefault('type', 'AdversarialPerturbationInjection')
                cfg.setdefault('channels', level_channels)
                self.api_modules_by_level[str(level)] = MODELS.build(cfg)

        self.out_channels = getattr(self.base_neck, 'out_channels',
                                    out_channels)
        self.num_outs = getattr(self.base_neck, 'num_outs', None)

    @staticmethod
    def _build_base_neck(base_neck: ConfigType):
        if isinstance(base_neck, (list, tuple)):
            return nn.Sequential(*(MODELS.build(cfg) for cfg in base_neck))
        return MODELS.build(base_neck)

    def _resolve_channels(self, out_channels: int | Sequence[int] | None):
        if out_channels is None:
            out_channels = getattr(self.base_neck, 'out_channels', None)
        if out_channels is None:
            raise ValueError('FeatureAugmentNeck needs out_channels when the '
                             'base neck does not expose it.')
        if isinstance(out_channels, Sequence) and not isinstance(
                out_channels, str):
            return [int(c) for c in out_channels]
        return int(out_channels)

    @property
    def api_modules(self) -> list[AdversarialPerturbationInjection]:
        return list(self.api_modules_by_level.values())

    def clear_api_state(self) -> None:
        for module in self.api_modules:
            module.clear_state()

    def capture_api(self) -> None:
        self.clear_api_state()
        for module in self.api_modules[:1]:
            module.capture()

    def perturb_api(self) -> None:
        for module in self.api_modules[:1]:
            module.perturb()

    def forward(self,
                inputs: tuple[Tensor, ...] | list[Tensor],
                batch_inputs: Tensor | None = None) -> tuple[Tensor, ...]:
        outs = list(self.base_neck(inputs))
        for level in self.levels:
            key = str(level)
            if key in self.morphology_modules:
                outs[level] = self.morphology_modules[key](outs[level])
            if key in self.dgfe_modules:
                if batch_inputs is None:
                    raise RuntimeError('DGFE requires batch_inputs.')
                outs[level] = self.dgfe_modules[key](outs[level], batch_inputs)
            if key in self.api_modules_by_level:
                outs[level] = self.api_modules_by_level[key](outs[level])
        return tuple(outs)
