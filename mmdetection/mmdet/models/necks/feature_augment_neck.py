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
    """Compare positive top-hat features with a matched raw-P3 control."""

    _valid_modes = {'positive', 'raw'}

    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 mode: str = 'positive',
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be an odd integer greater than 1')
        if mode not in self._valid_modes:
            raise ValueError(
                f'mode must be one of {sorted(self._valid_modes)}, got {mode!r}')
        self.kernel_size = int(kernel_size)
        self.mode = mode
        self.mixer = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.mixer.weight)

    def _dilate(self, x: Tensor) -> Tensor:
        return F.max_pool2d(
            x, self.kernel_size, stride=1, padding=self.kernel_size // 2)

    def _erode(self, x: Tensor) -> Tensor:
        return -self._dilate(-x)

    def forward(self, x: Tensor) -> Tensor:
        mixer_input = x
        if self.mode == 'positive':
            opening = self._dilate(self._erode(x))
            mixer_input = F.relu(x - opening)
        return x + self.mixer(mixer_input)


@MODELS.register_module()
class PhaseCongruencyFPN(BaseModule):
    """Phase Congruency & Dynamic Fourier Filtering (Phase-FPN)"""

    def __init__(self,
                 channels: int = 256,
                 num_masks: int = 4,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_masks = num_masks
        
        # Learnable Frequency Masks for Phase Congruency
        self.mask_weights = nn.Parameter(torch.randn(num_masks, 1, 1, 1))
        # Center frequencies and bandwidths parameters
        self.f_c = nn.Parameter(torch.rand(num_masks, 1, 1, 1) * 0.5)
        self.sigma = nn.Parameter(torch.rand(num_masks, 1, 1, 1) * 0.25 + 0.05)
        
        self.conv_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1)
        )
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        # 1. Real 2D FFT
        x_fft = torch.fft.rfft2(x, norm='backward')
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # 2. Estimate Phase Congruency Map in Frequency Domain
        # Phase alignment measure across channels
        phase_mean = phase.mean(dim=1, keepdim=True)
        phase_dev = torch.cos(phase - phase_mean)
        
        # Generate normalized frequency coordinates for learnable masks
        # u: vertical frequency [-0.5, 0.5]
        # v: horizontal frequency [0, 0.5]
        W_fft = amplitude.shape[-1]
        u = torch.fft.fftfreq(H, device=x.device).view(H, 1)
        v = torch.fft.rfftfreq(W, device=x.device).view(1, W_fft)
        r = torch.sqrt(u**2 + v**2) # (H, W_fft)
        
        # M_k of shape (num_masks, 1, H, W_fft)
        M_k = torch.exp(-((r.unsqueeze(0) - self.f_c) ** 2) / (2 * (self.sigma ** 2) + 1e-6))
        
        # Weight masks
        weights = torch.softmax(self.mask_weights, dim=0) # (num_masks, 1, 1, 1)
        M_k_w = (M_k * weights).unsqueeze(2) # (num_masks, 1, 1, H, W_fft)
        
        # Expand amplitude and phase_dev to shape (1, B, C, H, W_fft)
        amp_exp = amplitude.unsqueeze(0)
        pdev_exp = phase_dev.unsqueeze(0)
        
        # Weighted gating sum across masks and channels
        pc_num = (M_k_w * amp_exp * pdev_exp).sum(dim=0).sum(dim=1, keepdim=True)
        pc_den = (M_k_w * amp_exp).sum(dim=0).sum(dim=1, keepdim=True)
        
        pc_map = pc_num / (pc_den + 1e-6)
        pc_gate = torch.sigmoid(pc_map)

        # 3. Filter Frequency Representation
        x_fft_filtered = (amplitude * pc_gate) * torch.exp(1j * phase)

        # 4. Inverse 2D FFT back to Spatial Domain
        delta_x = torch.fft.irfft2(x_fft_filtered, s=(H, W), norm='backward')
        delta_x = self.conv_refine(delta_x)

        # 5. Residual connection with zero-init
        return x + self.gamma * delta_x


@MODELS.register_module()
class SubPixelImplicitRefiner(BaseModule):
    """Implicit Sub-Pixel Coordinate Field (SubPixel-INR)"""

    def __init__(self,
                 channels: int = 256,
                 embed_dim: int = 16,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_bands = 4 # Since enc_dim is 16, 2 coords * 2 (sin/cos) * num_bands = 16
        
        # 1. Define sub-pixel offsets for 2x2 local implicit grid
        offsets = torch.tensor([
            [-0.25, -0.25], [-0.25, 0.25],
            [ 0.25, -0.25], [ 0.25, 0.25]
        ], dtype=torch.float32) # (4, 2)
        self.register_buffer('offsets', offsets)
        
        # 3. Lightweight Implicit Neural Network (INR)
        self.inr_mlp = nn.Sequential(
            nn.Conv2d(channels + embed_dim, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
        
        self.attn_heads = nn.Conv2d(channels, 4, kernel_size=1) # Attention weights for 4 sub-pixels
        self.zero_conv = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def _fourier_encode(self, pos: Tensor) -> Tensor:
        # pos: (4, 2)
        scales = (2.0 ** torch.arange(self.num_bands, device=pos.device)).view(1, 1, -1) # (1, 1, num_bands)
        scaled_pos = pos.unsqueeze(-1) * scales * torch.pi # (4, 2, num_bands)
        encoded = torch.cat([torch.sin(scaled_pos), torch.cos(scaled_pos)], dim=-1) # (4, 2, 2 * num_bands)
        return encoded.view(4, -1) # (4, 16)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        
        # Encode coordinates once
        pos_enc = self._fourier_encode(self.offsets) # (4, 16)
        
        # Query implicit features at 4 sub-pixel locations using vectorized operations
        enc = pos_enc.view(1, 4, -1, 1, 1).expand(B, -1, -1, H, W)
        x_expanded = x.unsqueeze(1).expand(-1, 4, -1, -1, -1)
        x_k = torch.cat([x_expanded, enc], dim=2)
        x_k = x_k.reshape(B * 4, -1, H, W)
        
        # Pass through INR MLP
        f_k = self.inr_mlp(x_k) # (B * 4, C, H, W)
        stacked_f = f_k.view(B, 4, C, H, W)
        
        # Compute dynamic aggregation weights across sub-pixels
        attn_logits = self.attn_heads(x) # (B, 4, H, W)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(2) # (B, 4, 1, H, W)
        
        # Aggregate continuous sub-pixel details back to grid cell
        delta_x = (stacked_f * attn_weights).sum(dim=1) # (B, C, H, W)
        
        return x + self.zero_conv(delta_x)


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
                 phase_fpn: OptConfigType = None,
                 subpixel_inr: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.base_neck = self._build_base_neck(base_neck)
        self.levels = tuple(int(level) for level in levels)
        channels = self._resolve_channels(out_channels)
        self.morphology_modules = nn.ModuleDict()
        self.phase_fpn_modules = nn.ModuleDict()
        self.subpixel_inr_modules = nn.ModuleDict()
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
            if phase_fpn is not None:
                cfg = dict(phase_fpn)
                cfg.setdefault('type', 'PhaseCongruencyFPN')
                cfg.setdefault('channels', level_channels)
                self.phase_fpn_modules[str(level)] = MODELS.build(cfg)
            if subpixel_inr is not None:
                cfg = dict(subpixel_inr)
                cfg.setdefault('type', 'SubPixelImplicitRefiner')
                cfg.setdefault('channels', level_channels)
                self.subpixel_inr_modules[str(level)] = MODELS.build(cfg)
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
            if key in self.phase_fpn_modules:
                outs[level] = self.phase_fpn_modules[key](outs[level])
            if key in self.subpixel_inr_modules:
                outs[level] = self.subpixel_inr_modules[key](outs[level])
            if key in self.dgfe_modules:
                if batch_inputs is None:
                    raise RuntimeError('DGFE requires batch_inputs.')
                outs[level] = self.dgfe_modules[key](outs[level], batch_inputs)
            if key in self.api_modules_by_level:
                outs[level] = self.api_modules_by_level[key](outs[level])
        return tuple(outs)
