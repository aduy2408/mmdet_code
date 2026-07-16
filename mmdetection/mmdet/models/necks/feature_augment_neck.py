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
        spatial_prob = torch.sigmoid(logits)
        spatial_gate = 1.0 + spatial_prob

        avg_gate = self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
        max_gate = self.channel_mlp(F.adaptive_max_pool2d(x, 1))
        channel_gate = torch.sigmoid(avg_gate + max_gate)
        alpha = self.alpha.to(device=x.device, dtype=x.dtype)
        out = x * (1.0 + alpha * (channel_gate * spatial_gate - 1.0))
        self.last_aux = dict(recon=recon, spatial_logits=logits,
                             spatial_prob=spatial_prob,
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
                 forward_mode: str = 'partial',
                 guidance_mode: str = 'none',
                 eps: float = 1e-6,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.rho = max(float(rho), 0.0)
        self.api_weight = max(float(api_weight), 0.0)
        self.target_mode = str(target_mode)
        if forward_mode not in {'partial', 'full'}:
            raise ValueError('forward_mode must be "partial" or "full".')
        self.forward_mode = forward_mode
        if guidance_mode not in {'none', 'dgfe'}:
            raise ValueError('guidance_mode must be "none" or "dgfe".')
        self.guidance_mode = guidance_mode
        self.eps = max(float(eps), 1e-12)
        self.aux_head = nn.Conv2d(channels, 1, 1)
        self.mode = 'off'
        self.captured: Tensor | None = None
        self.perturbation: Tensor | None = None
        self.spatial_guidance: Tensor | None = None

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
        self.spatial_guidance = None

    def capture(self) -> None:
        self.clear_state()
        self.mode = 'capture'

    def perturb(self) -> None:
        self.mode = 'perturb'

    def set_spatial_guidance(self, guidance: Tensor) -> None:
        self.spatial_guidance = guidance.detach()

    def set_perturbation_from_grad(self, grad: Tensor | None) -> bool:
        if grad is None or self.current_rho == 0 or self.current_api_weight == 0:
            self.perturbation = None
            return False
        grad_f = grad.detach().float()
        if self.guidance_mode == 'dgfe':
            if self.spatial_guidance is None:
                self.perturbation = None
                return False
            guidance = self.spatial_guidance.to(
                device=grad.device, dtype=torch.float32)
            if guidance.shape[0] != grad_f.shape[0]:
                self.perturbation = None
                return False
            if guidance.shape[-2:] != grad_f.shape[-2:]:
                guidance = F.interpolate(
                    guidance, size=grad_f.shape[-2:], mode='bilinear',
                    align_corners=False)
            grad_f = grad_f * guidance.clamp(0.0, 1.0)
        norm = grad_f.flatten(1).norm(p=2, dim=1)
        if not torch.isfinite(norm).all() or (norm <= self.eps).any():
            self.perturbation = None
            return False
        norm = norm.view(-1, 1, 1, 1)
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
    """Wrap a normal neck and apply optional DGFE/API modules to outputs."""

    needs_batch_inputs = True

    def __init__(self,
                 base_neck: ConfigType,
                 levels: Sequence[int] = (0, ),
                 out_channels: int | Sequence[int] | None = None,
                 dgfe: OptConfigType = None,
                 api: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.base_neck = self._build_base_neck(base_neck)
        self.levels = tuple(int(level) for level in levels)
        channels = self._resolve_channels(out_channels)
        self.dgfe_modules = nn.ModuleDict()
        self.api_modules_by_level = nn.ModuleDict()

        for level in self.levels:
            level_channels = (
                channels[level] if isinstance(channels, list) else channels)
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
            if key in self.dgfe_modules:
                if batch_inputs is None:
                    raise RuntimeError('DGFE requires batch_inputs.')
                outs[level] = self.dgfe_modules[key](outs[level], batch_inputs)
            if key in self.api_modules_by_level:
                api = self.api_modules_by_level[key]
                if api.guidance_mode == 'dgfe' and self.training:
                    dgfe = (self.dgfe_modules[key]
                            if key in self.dgfe_modules else None)
                    if dgfe is None or dgfe.last_aux is None:
                        raise RuntimeError(
                            'DGFE-guided API requires a training DGFE output.')
                    api.set_spatial_guidance(dgfe.last_aux['spatial_prob'])
                outs[level] = api(outs[level])
        return tuple(outs)
