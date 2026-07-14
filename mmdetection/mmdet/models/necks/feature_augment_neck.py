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
                 use_rho_warmup: bool = False,
                 warmup_epochs: int = 10,
                 use_per_box_norm: bool = False,
                 use_fgsm_dropout: bool = False,
                 fgsm_drop_rate: float = 0.1,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.rho = max(float(rho), 0.0)
        self.api_weight = max(float(api_weight), 0.0)
        self.target_mode = str(target_mode)
        self.eps = max(float(eps), 1e-12)
        self.use_rho_warmup = bool(use_rho_warmup)
        self.warmup_epochs = max(int(warmup_epochs), 1)
        self.use_per_box_norm = bool(use_per_box_norm)
        self.use_fgsm_dropout = bool(use_fgsm_dropout)
        self.fgsm_drop_rate = max(min(float(fgsm_drop_rate), 1.0), 0.0)
        self.aux_head = nn.Conv2d(channels, 1, 1)
        self.mode = 'off'
        self.captured: Tensor | None = None
        self.perturbation: Tensor | None = None
        self.last_perturbation_norm: Tensor | None = None
        self._epoch = 0

    @property
    def current_rho(self) -> float:
        if self.use_rho_warmup and self._epoch < self.warmup_epochs:
            t = float(self._epoch) / float(self.warmup_epochs)
            return self.rho * (0.1 + 0.9 * t)
        return self.rho

    @property
    def current_api_weight(self) -> float:
        if self.use_rho_warmup and self._epoch < self.warmup_epochs:
            t = float(self._epoch) / float(self.warmup_epochs)
            return self.api_weight * (0.1 + 0.9 * t)
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

    def set_perturbation_from_grad(self,
                                   grad: Tensor | None,
                                   batch_inputs: Tensor | None = None,
                                   batch_data_samples=None) -> bool:
        if grad is None or self.current_rho == 0 or self.current_api_weight == 0:
            self.perturbation = None
            return False
        grad_f = grad.detach().float()
        if (self.use_per_box_norm and batch_inputs is not None
                and batch_data_samples is not None):
            weight_map = torch.ones(
                grad_f.shape[0], 1, grad_f.shape[2], grad_f.shape[3],
                device=grad_f.device,
                dtype=grad_f.dtype)
            _, _, img_h, img_w = batch_inputs.shape
            feat_h, feat_w = grad_f.shape[-2:]
            for batch_idx, data_sample in enumerate(batch_data_samples):
                bboxes = data_sample.gt_instances.bboxes
                bboxes = bboxes.tensor if hasattr(bboxes, 'tensor') else bboxes
                if bboxes.numel() == 0:
                    continue
                boxes = bboxes.to(device=grad_f.device, dtype=grad_f.dtype)
                xs1 = (boxes[:, 0] / img_w * feat_w).floor().long()
                ys1 = (boxes[:, 1] / img_h * feat_h).floor().long()
                xs2 = (boxes[:, 2] / img_w * feat_w).ceil().long()
                ys2 = (boxes[:, 3] / img_h * feat_h).ceil().long()
                for x1, y1, x2, y2 in zip(xs1, ys1, xs2, ys2):
                    x1 = int(x1.clamp(0, feat_w - 1))
                    y1 = int(y1.clamp(0, feat_h - 1))
                    x2 = int(x2.clamp(x1 + 1, feat_w))
                    y2 = int(y2.clamp(y1 + 1, feat_h))
                    area = max((x2 - x1) * (y2 - y1), 1)
                    scale = float(feat_h * feat_w / area) ** 0.5
                    weight_map[batch_idx, :, y1:y2, x1:x2] *= scale
            grad_f = grad_f * weight_map
        norm = grad_f.flatten(1).norm(p=2, dim=1).clamp(
            min=self.eps).view(-1, 1, 1, 1)
        perturbation = grad_f / norm * self.current_rho
        if not torch.isfinite(perturbation).all():
            self.perturbation = None
            return False
        self.perturbation = perturbation.to(device=grad.device,
                                            dtype=grad.dtype)
        self.last_perturbation_norm = self.perturbation.detach().float(
        ).flatten(1).norm(p=2, dim=1)
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
            return self.apply_perturbation(x)
        return x

    def apply_perturbation(self, feature: Tensor) -> Tensor:
        """Apply the stored perturbation without requiring another forward."""
        if self.perturbation is None:
            return feature
        out = feature + self.perturbation.to(
            device=feature.device, dtype=feature.dtype)
        if self.use_fgsm_dropout:
            ch_mag = self.perturbation.abs().mean(dim=(2, 3))
            k = max(1, int(ch_mag.shape[1] * self.fgsm_drop_rate))
            thresh = ch_mag.topk(k, dim=1).values[:, -1].view(
                -1, 1, 1, 1)
            keep_mask = (ch_mag.unsqueeze(-1).unsqueeze(-1) <
                         thresh).to(dtype=out.dtype)
            keep_frac = max(1.0 - self.fgsm_drop_rate, 1e-3)
            out = out * keep_mask / keep_frac
        return out

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
                 dgfe: OptConfigType = None,
                 api: OptConfigType = None,
                 dgfe_rec_gain: float = 0.0,
                 dgfe_spatial_gain: float = 0.0,
                 dgfe_spatial_warmup_epochs: int = 0,
                 dgfe_spatial_warmup_start: float = 0.1,
                 dgfe_boundary_ring: float = 1.0,
                 dgfe_inner_value: float = 0.3,
                 dgfe_tiny_area: float = 4.0,
                 dgfe_neg_pos_ratio: int = 3,
                 dgfe_neg_gain: float = 0.25,
                 dgfe_spatial_target_mode: str = 'iou',
                 dgfe_edge_error_norm: float = 0.25,
        init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.base_neck = self._build_base_neck(base_neck)
        self.levels = tuple(int(level) for level in levels)
        self.dgfe_rec_gain = float(dgfe_rec_gain)
        self.dgfe_spatial_gain = float(dgfe_spatial_gain)
        self.dgfe_spatial_warmup_epochs = max(
            int(dgfe_spatial_warmup_epochs), 0)
        self.dgfe_spatial_warmup_start = max(
            min(float(dgfe_spatial_warmup_start), 1.0), 0.0)
        self.dgfe_boundary_ring = float(dgfe_boundary_ring)
        self.dgfe_inner_value = float(dgfe_inner_value)
        self.dgfe_tiny_area = float(dgfe_tiny_area)
        self.dgfe_neg_pos_ratio = int(dgfe_neg_pos_ratio)
        self.dgfe_neg_gain = float(dgfe_neg_gain)
        self.dgfe_spatial_target_mode = str(dgfe_spatial_target_mode).lower()
        self.dgfe_edge_error_norm = max(float(dgfe_edge_error_norm), 1e-9)
        self.dgfe_epoch = 0
        self._last_dgfe_aux: list[dict[str, Tensor]] = []
        self._api_clean_features: tuple[Tensor, ...] | None = None
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

    def set_epoch(self, epoch: int) -> None:
        self.dgfe_epoch = int(epoch)
        for module in self.api_modules:
            module._epoch = int(epoch)

    def dgfe_aux_list(self) -> list[dict[str, Tensor]]:
        return list(self._last_dgfe_aux)

    def clear_api_state(self) -> None:
        for module in self.api_modules:
            module.clear_state()
        self._api_clean_features = None

    def capture_api(self) -> None:
        self.clear_api_state()
        for module in self.api_modules[:1]:
            module.capture()

    def perturb_api(self) -> None:
        for module in self.api_modules[:1]:
            module.perturb()

    def perturb_features(self,
                         features: tuple[Tensor, ...] | None = None
                         ) -> tuple[Tensor, ...] | None:
        """Return a perturbed copy of cached post-neck features."""
        features = self._api_clean_features if features is None else features
        if features is None:
            return None
        outs = list(features)
        for level in self.levels:
            key = str(level)
            if key in self.api_modules_by_level:
                module = self.api_modules_by_level[key]
                outs[level] = module.apply_perturbation(outs[level])
                break
        return tuple(outs)

    def forward(self,
                inputs: tuple[Tensor, ...] | list[Tensor],
                batch_inputs: Tensor | None = None) -> tuple[Tensor, ...]:
        outs = list(self.base_neck(inputs))
        self._last_dgfe_aux = []
        for level in self.levels:
            key = str(level)
            if key in self.dgfe_modules:
                if batch_inputs is None:
                    raise RuntimeError('DGFE requires batch_inputs.')
                module = self.dgfe_modules[key]
                module.last_aux = None
                outs[level] = module(outs[level], batch_inputs)
                if module.last_aux is not None:
                    aux = dict(module.last_aux)
                    aux['level'] = outs[level].new_tensor(level)
                    self._last_dgfe_aux.append(aux)
            if key in self.api_modules_by_level:
                outs[level] = self.api_modules_by_level[key](outs[level])
        result = tuple(outs)
        if any(module.mode == 'capture' for module in self.api_modules[:1]):
            self._api_clean_features = result
        return result
