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

    def forward(self, x: Tensor, batch_inputs: Tensor | None = None) -> Tensor:
        B, C, H, W = x.shape
        
        if batch_inputs is None:
            # Fallback for unit testing where batch_inputs is not provided
            img = x.mean(dim=1, keepdim=True)
        else:
            # Convert RGB image (B, 3, H_img, W_img) to grayscale (B, 1, H_img, W_img)
            img = 0.299 * batch_inputs[:, 0:1] + 0.587 * batch_inputs[:, 1:2] + 0.114 * batch_inputs[:, 2:3]
            
        # 3x3 Morphological Positive Top-hat:
        # Erosion is max pooling of negative image
        erosion = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
        # Dilation of erosion:
        dilation_of_erosion = F.max_pool2d(erosion, kernel_size=3, stride=1, padding=1)
        # Positive top-hat filter to highlight bright anomalies (ships) and suppress background
        img = torch.relu(img - dilation_of_erosion)
            
        B_img, _, H_img, W_img = img.shape
        
        # 1. 2D FFT of top-hat filtered grayscale input image
        img_fft = torch.fft.rfft2(img, norm='ortho')
        amplitude = torch.abs(img_fft)
        
        # 2. Quadrature filter pairs over scales
        W_fft = amplitude.shape[-1]
        u = torch.fft.fftfreq(H_img, device=img.device).view(H_img, 1)
        v = torch.fft.rfftfreq(W_img, device=img.device).view(1, W_fft)
        r = torch.sqrt(u**2 + v**2).clamp(min=1e-6) # (H_img, W_fft)
        
        sign_u = torch.sign(u).unsqueeze(0).unsqueeze(1) # (1, 1, H_img, 1)
        sign_v = torch.sign(v).unsqueeze(0).unsqueeze(1) # (1, 1, 1, W_fft)
        
        # Gaussian masks
        f_c = self.f_c.view(self.num_masks, 1, 1, 1)
        sigma = self.sigma.view(self.num_masks, 1, 1, 1)
        r_exp = r.unsqueeze(0)
        M_k = torch.exp(-((r_exp - f_c) ** 2) / (2 * (sigma ** 2) + 1e-6))
        
        weights = torch.softmax(self.mask_weights, dim=0).view(self.num_masks, 1, 1, 1)
        M_k_w = M_k * weights # (num_masks, 1, H_img, W_fft)
        
        # Filtered representations in frequency domain
        filt_fft = img_fft.unsqueeze(0) * M_k_w.unsqueeze(1) # (num_masks, B, 1, H_img, W_fft)
        
        # Inverse transform to get even and odd components
        even_k = torch.fft.irfft2(filt_fft, s=(H_img, W_img), norm='ortho')
        odd_k_u = torch.fft.irfft2(filt_fft * (-1j * sign_u), s=(H_img, W_img), norm='ortho')
        odd_k_v = torch.fft.irfft2(filt_fft * (-1j * sign_v), s=(H_img, W_img), norm='ortho')
        
        # Compute Phase Congruency along vertical and horizontal orientations
        sum_E = even_k.sum(dim=0)
        sum_O_u = odd_k_u.sum(dim=0)
        sum_O_v = odd_k_v.sum(dim=0)
        
        energy_u = torch.sqrt(sum_E ** 2 + sum_O_u ** 2 + 1e-12)
        energy_v = torch.sqrt(sum_E ** 2 + sum_O_v ** 2 + 1e-12)
        
        amplitude_u = torch.sqrt(even_k ** 2 + odd_k_u ** 2 + 1e-12).sum(dim=0)
        amplitude_v = torch.sqrt(even_k ** 2 + odd_k_v ** 2 + 1e-12).sum(dim=0)
        
        pc_u = energy_u / (amplitude_u + 1e-4)
        pc_v = energy_v / (amplitude_v + 1e-4)
        
        pc_map = torch.max(pc_u, pc_v) # (B, 1, H_img, W_img)
        
        # 3. Downsample Phase Congruency map to match FPN levels
        if (H, W) != (H_img, W_img):
            pc_map_down = F.interpolate(pc_map, size=(H, W), mode='bilinear', align_corners=False)
        else:
            pc_map_down = pc_map
            
        pc_gate = torch.sigmoid(pc_map_down)
        
        # 4. Refine features
        delta_x = x * pc_gate
        delta_x = self.conv_refine(delta_x)
        
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
        
        # 3. Lightweight Implicit Neural Network (INR) with 2 * channels input
        # (for raw feature x + standardized local contrast x_std)
        self.inr_mlp = nn.Sequential(
            nn.Conv2d(2 * channels + embed_dim, channels, kernel_size=1),
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

    def _bilinear_shift(self, x: Tensor, dy: float, dx: float) -> Tensor:
        # x: (B, C, H, W)
        # Shift and interpolate features at (y + dy, x + dx) using replicate padding
        padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        
        ay = abs(dy)
        ax = abs(dx)
        
        h_slice = slice(1, -1)
        w_slice = slice(1, -1)
        h_neighbor = slice(0, -2) if dy < 0 else slice(2, None)
        w_neighbor = slice(0, -2) if dx < 0 else slice(2, None)
        
        f_00 = padded[:, :, h_slice, w_slice]
        f_01 = padded[:, :, h_slice, w_neighbor]
        f_10 = padded[:, :, h_neighbor, w_slice]
        f_11 = padded[:, :, h_neighbor, w_neighbor]
        
        w_00 = (1.0 - ay) * (1.0 - ax)
        w_01 = (1.0 - ay) * ax
        w_10 = ay * (1.0 - ax)
        w_11 = ay * ax
        
        return w_00 * f_00 + w_01 * f_01 + w_10 * f_10 + w_11 * f_11

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        
        # 1. Local background standardization (RCFN-like local standardized deviation Z) on x
        # 3x3 uniform sum kernel:
        kernel = torch.ones((C, 1, 3, 3), device=x.device)
        sum_3x3 = F.conv2d(x, kernel, padding=1, groups=C)
        sum_sq_3x3 = F.conv2d(x**2, kernel, padding=1, groups=C)
        
        # 8-neighbor ring sum and mean:
        sum_ring = sum_3x3 - x
        mean_ring = sum_ring / 8.0
        
        # 8-neighbor ring variance:
        var_ring = (sum_sq_3x3 - x**2) / 8.0 - mean_ring**2
        std_ring = torch.sqrt(var_ring.clamp(min=1e-6))
        
        # Local standardized features to suppress ocean background noise
        x_std = (x - mean_ring) / (std_ring + 1e-4)
        
        # Concatenate raw semantics and local standardized features
        x_feat = torch.cat([x, x_std], dim=1) # (B, 2*C, H, W)
        
        # 2. Encode coordinates once
        pos_enc = self._fourier_encode(self.offsets) # (4, 16)
        
        # Query spatially interpolated implicit features at 4 sub-pixel locations
        x_k_list = []
        for i, (dy, dx) in enumerate(self.offsets.tolist()):
            x_k_feat = self._bilinear_shift(x_feat, dy, dx) # (B, 2*C, H, W)
            enc = pos_enc[i].view(1, -1, 1, 1).expand(B, -1, H, W)
            x_k_list.append(torch.cat([x_k_feat, enc], dim=1))
            
        x_k = torch.stack(x_k_list, dim=1) # (B, 4, 2*C + embed_dim, H, W)
        x_k = x_k.view(B * 4, -1, H, W)
        
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
                outs[level] = self.phase_fpn_modules[key](outs[level], batch_inputs)
            if key in self.subpixel_inr_modules:
                outs[level] = self.subpixel_inr_modules[key](outs[level])
            if key in self.dgfe_modules:
                if batch_inputs is None:
                    raise RuntimeError('DGFE requires batch_inputs.')
                outs[level] = self.dgfe_modules[key](outs[level], batch_inputs)
            if key in self.api_modules_by_level:
                outs[level] = self.api_modules_by_level[key](outs[level])
        return tuple(outs)
