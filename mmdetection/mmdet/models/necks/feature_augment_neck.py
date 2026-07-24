# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import ConfigType, OptConfigType


class MaskedCenterConv2d(nn.Conv2d):
    """A 3x3 convolution that cannot observe the predicted center."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels, channels, 3, padding=1)
        mask = torch.ones_like(self.weight)
        mask[:, :, 1, 1] = 0
        self.register_buffer('mask', mask)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


@MODELS.register_module()
class DualIrreducibilityHIT(BaseModule):
    """Locate jointly irreducible residuals and splat them toward objects."""

    def __init__(self,
                 channels: int,
                 stride: int = 8,
                 reduction: int = 8,
                 topk: int = 4,
                 max_offset: float = 8.0,
                 source_topq: float = 0.01,
                 fixed_sigma: float = 1.0,
                 detach_offset_input: bool = True,
                 offset_target_margin: int = 1,
                 transport_enabled: bool = True,
                 background_recon_only: bool = True,
                 hard_clip: float = 5.0,
                 loss_recon_spatial_weight: float = 0.1,
                 loss_recon_channel_weight: float = 0.1,
                 loss_offset_weight: float = 1.0,
                 eps: float = 1e-6,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden = max(channels // max(int(reduction), 1), 1)
        self.channels = int(channels)
        self.stride = int(stride)
        self.topk = max(int(topk), 1)
        self.max_offset = float(max_offset)
        if not 0 < source_topq <= 1:
            raise ValueError('source_topq must be in (0, 1].')
        self.source_topq = float(source_topq)
        self.fixed_sigma = float(fixed_sigma)
        self.detach_offset_input = bool(detach_offset_input)
        self.offset_target_margin = max(int(offset_target_margin), 0)
        self.transport_enabled = bool(transport_enabled)
        self.background_recon_only = bool(background_recon_only)
        self.hard_clip = float(hard_clip)
        self.loss_recon_spatial_weight = float(
            loss_recon_spatial_weight)
        self.loss_recon_channel_weight = float(
            loss_recon_channel_weight)
        self.loss_offset_weight = float(loss_offset_weight)
        self.eps = float(eps)

        self.spatial_reconstruct = MaskedCenterConv2d(channels)
        self.channel_reconstruct = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.residual_fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.SiLU(inplace=True),
        )
        self.offset_head = nn.Conv2d(channels + 1, 2, 3, padding=1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        self.transport_projection = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.transport_projection.weight)
        nn.init.zeros_(self.transport_projection.bias)
        self.last_aux: dict[str, Tensor] | None = None

    def hard_map(self, spatial_residual: Tensor,
                 channel_residual: Tensor) -> Tensor:
        spatial_energy = spatial_residual.abs().mean(dim=1, keepdim=True)
        channel_energy = channel_residual.abs().mean(dim=1, keepdim=True)
        return (2 * spatial_energy * channel_energy /
                (spatial_energy + channel_energy + self.eps))

    def sparse_gate(self, hard: Tensor) -> Tensor:
        """Select the global top-q hard locations independently per image."""
        batch, _, height, width = hard.shape
        count = max(1, math.ceil(height * width * self.source_topq))
        indices = hard.detach().flatten(2).topk(count, dim=2).indices
        gate = torch.zeros_like(hard).flatten(2)
        gate.scatter_(2, indices, 1)
        return gate.reshape(batch, 1, height, width)

    def _gaussian_splat(self, source: Tensor, offsets: Tensor,
                        sigma: Tensor) -> Tensor:
        batch, channels, height, width = source.shape
        dtype, device = source.dtype, source.device
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij')
        dest_x = xx.reshape(1, -1) + offsets[:, 0].reshape(batch, -1)
        dest_y = yy.reshape(1, -1) + offsets[:, 1].reshape(batch, -1)
        base_x, base_y = dest_x.floor(), dest_y.floor()
        sigma_flat = sigma.reshape(batch, -1).clamp(min=self.eps)

        indices, weights = [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                target_x = base_x + dx
                target_y = base_y + dy
                valid = ((target_x >= 0) & (target_x < width) &
                         (target_y >= 0) & (target_y < height))
                distance_sq = ((target_x - dest_x).square() +
                               (target_y - dest_y).square())
                weight = torch.exp(
                    -0.5 * distance_sq / sigma_flat.square())
                weights.append(weight * valid)
                indices.append(
                    (target_y.clamp(0, height - 1) * width +
                     target_x.clamp(0, width - 1)).long())

        weight_stack = torch.stack(weights, dim=1)
        weight_stack = weight_stack / weight_stack.sum(
            dim=1, keepdim=True).clamp(min=self.eps)
        source_flat = source.reshape(batch, channels, -1)
        transported = source.new_zeros(batch, channels, height * width)
        for index, weight in zip(indices, weight_stack.unbind(dim=1)):
            transported.scatter_add_(
                2, index.unsqueeze(1).expand(-1, channels, -1),
                source_flat * weight.unsqueeze(1))
        return transported.reshape(batch, channels, height, width)

    def forward(self, x: Tensor) -> Tensor:
        spatial_reconstruction = self.spatial_reconstruct(x)
        channel_reconstruction = self.channel_reconstruct(x)
        spatial_residual = x - spatial_reconstruction
        channel_residual = x - channel_reconstruction
        hard_raw = self.hard_map(spatial_residual, channel_residual)
        hard = hard_raw / hard_raw.mean(
            dim=(2, 3), keepdim=True).detach().clamp(min=self.eps)
        hard = hard.clamp(max=self.hard_clip)

        gate = self.sparse_gate(hard_raw)
        offset_x = x.detach() if self.detach_offset_input else x
        offset_hard = hard.detach() if self.detach_offset_input else hard
        offset_params = self.offset_head(torch.cat([offset_x, offset_hard],
                                                   dim=1))
        offsets = torch.tanh(offset_params) * self.max_offset
        sigma = offsets.new_full(
            (offsets.shape[0], 1, offsets.shape[2], offsets.shape[3]),
            self.fixed_sigma)
        residual = self.residual_fuse(
            torch.cat([spatial_residual, channel_residual], dim=1))
        source = residual * hard * gate
        if self.transport_enabled:
            transported = self._gaussian_splat(source, offsets, sigma)
            update = self.transport_projection(transported)
        else:
            transported = torch.zeros_like(x)
            update = torch.zeros_like(x)

        self.last_aux = dict(
            feature=x,
            spatial_reconstruction=spatial_reconstruction,
            channel_reconstruction=channel_reconstruction,
            hard_raw=hard_raw,
            hard=hard,
            gate=gate,
            source=source,
            offsets=offsets,
            sigma=sigma,
            transported=transported,
        ) if self.training else None
        return x + update

    def _offset_targets(self, batch_data_samples) -> tuple[Tensor, Tensor]:
        assert self.last_aux is not None
        hard = self.last_aux['hard_raw'].detach()
        offsets = self.last_aux['offsets']
        gate = self.last_aux['gate'].bool()
        _, _, height, width = hard.shape
        predictions, targets = [], []
        target_count = clamped_count = 0

        yy, xx = torch.meshgrid(
            torch.arange(height, device=hard.device, dtype=hard.dtype),
            torch.arange(width, device=hard.device, dtype=hard.dtype),
            indexing='ij')
        cell_x, cell_y = xx + 0.5, yy + 0.5
        flat_x, flat_y = cell_x.flatten(), cell_y.flatten()

        for batch_index, data_sample in enumerate(batch_data_samples):
            gt_instances = data_sample.gt_instances
            bboxes = get_box_tensor(gt_instances.bboxes).to(
                device=hard.device, dtype=hard.dtype)
            assignments: dict[int, tuple[Tensor, Tensor]] = {}
            for box in bboxes:
                box_cells = box / self.stride
                x1, y1, x2, y2 = box_cells
                margin = self.offset_target_margin
                inside = ((flat_x >= x1 - margin) &
                          (flat_x <= x2 + margin) &
                          (flat_y >= y1 - margin) &
                          (flat_y <= y2 + margin))
                inside &= gate[batch_index, 0].flatten()
                candidates = inside.nonzero(as_tuple=False).flatten()
                center = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2))
                if candidates.numel() == 0:
                    continue
                scores = hard[batch_index, 0].flatten()[candidates]
                count = min(self.topk, candidates.numel())
                selected = candidates[scores.topk(count).indices]
                area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
                for index in selected:
                    key = int(index)
                    raw_target = center - torch.stack(
                        (flat_x[index], flat_y[index]))
                    target = raw_target.clamp(-self.max_offset,
                                              self.max_offset)
                    target_count += 1
                    clamped_count += int(not torch.equal(raw_target, target))
                    if key not in assignments or area < assignments[key][0]:
                        assignments[key] = (area, target)

            for index, (_, target) in assignments.items():
                y, x_coord = divmod(index, width)
                predictions.append(offsets[batch_index, :, y, x_coord])
                targets.append(target)

        if not predictions:
            self.last_aux['offset_clamp_rate'] = offsets.new_tensor(0.)
            return offsets.new_empty((0, 2)), offsets.new_empty((0, 2))
        self.last_aux['offset_clamp_rate'] = offsets.new_tensor(
            clamped_count / max(target_count, 1))
        return torch.stack(predictions), torch.stack(targets)

    def _background_mask(self, batch_data_samples, feature: Tensor) -> Tensor:
        batch, _, height, width = feature.shape
        mask = torch.ones(
            batch, 1, height, width, dtype=torch.bool, device=feature.device)
        margin = self.offset_target_margin
        for batch_index, data_sample in enumerate(batch_data_samples):
            bboxes = get_box_tensor(data_sample.gt_instances.bboxes).to(
                device=feature.device, dtype=feature.dtype) / self.stride
            for x1, y1, x2, y2 in bboxes:
                left = max(math.floor(float(x1)) - margin, 0)
                top = max(math.floor(float(y1)) - margin, 0)
                right = min(math.ceil(float(x2)) + margin + 1, width)
                bottom = min(math.ceil(float(y2)) + margin + 1, height)
                mask[batch_index, :, top:bottom, left:right] = False
        return mask

    def _reconstruction_loss(self, prediction: Tensor, target: Tensor,
                             background: Tensor) -> Tensor:
        error = (prediction - target).abs()
        if self.background_recon_only and background.any():
            return error.masked_select(background.expand_as(error)).mean()
        return error.mean()

    def auxiliary_losses(self, batch_data_samples) -> dict[str, Tensor]:
        if self.last_aux is None:
            return {}
        feature = self.last_aux['feature'].detach()
        background = self._background_mask(batch_data_samples, feature)
        loss_spatial = self._reconstruction_loss(
            self.last_aux['spatial_reconstruction'], feature, background)
        loss_channel = self._reconstruction_loss(
            self.last_aux['channel_reconstruction'], feature, background)
        if self.transport_enabled and self.loss_offset_weight:
            prediction, target = self._offset_targets(batch_data_samples)
        else:
            prediction = target = feature.new_empty((0, 2))
            self.last_aux['offset_clamp_rate'] = feature.new_tensor(0.)
        if prediction.numel():
            loss_offset = F.smooth_l1_loss(
                prediction, target, beta=1.0, reduction='mean')
        else:
            loss_offset = self.last_aux['offsets'].sum() * 0
        return dict(
            loss_hit_recon_spatial=(
                loss_spatial * self.loss_recon_spatial_weight),
            loss_hit_recon_channel=(
                loss_channel * self.loss_recon_channel_weight),
            loss_hit_offset=loss_offset * self.loss_offset_weight,
            hit_offset_clamp_rate=self.last_aux['offset_clamp_rate'].detach(),
        )


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
                 dgfe: OptConfigType = None,
                 api: OptConfigType = None,
                 hit: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.base_neck = self._build_base_neck(base_neck)
        self.levels = tuple(int(level) for level in levels)
        channels = self._resolve_channels(out_channels)
        self.dgfe_modules = nn.ModuleDict()
        self.api_modules_by_level = nn.ModuleDict()
        self.hit_modules = nn.ModuleDict()

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
            if hit is not None:
                cfg = dict(hit)
                cfg.setdefault('type', 'DualIrreducibilityHIT')
                cfg.setdefault('channels', level_channels)
                self.hit_modules[str(level)] = MODELS.build(cfg)

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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Accept checkpoints produced by the unwrapped base neck."""
        base_keys = set(self.base_neck.state_dict())
        for key in list(state_dict):
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if suffix in base_keys:
                state_dict[prefix + 'base_neck.' + suffix] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

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

    def auxiliary_losses(self, batch_data_samples) -> dict[str, Tensor]:
        losses = {}
        for level, module in self.hit_modules.items():
            for name, loss in module.auxiliary_losses(
                    batch_data_samples).items():
                key = name if len(self.hit_modules) == 1 else f'{name}_p{level}'
                losses[key] = loss
        return losses

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
                outs[level] = self.api_modules_by_level[key](outs[level])
            if key in self.hit_modules:
                outs[level] = self.hit_modules[key](outs[level])
        return tuple(outs)
