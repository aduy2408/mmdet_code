# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import CBAM, Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import LayerNorm2d, TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "AdversarialPerturbationInjection",
    "ASFAttention",
    "Attention",
    "BiLevelRoutingAttention",
    "BNContrastiveHead",
    "BoundaryFeatureBlock",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fCBAM",
    "C2fAttn",
    "C2fCIB",
    "C2fKV",
    "C2fNAT",
    "C2fPSA",
    "EnSimAM",
    "EnSimAMEdgeRepC2f",
    "FeatureDGFE",
    "C3CBAM",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "KVCompressedAttention",
    "KVCompressedAttentionPartial",
    "KVCompressedTransformerEncoder",
    "M3NATFuse",
    "Proto",
    "RegionRoutingAttentionLite",
    "TopKAdaptiveGroupKVAttention",
    "TopKGlobalGroupKVAttention",
    "RepC3",
    "RepC2f",
    "PoolingEdgeRepC2f",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "ScalSeq",
    "SCDown",
    "TorchVision",
    "WeightedAdd",
    "clear_boundary_context",
    "set_boundary_context",
    "set_boundary_enabled",
)


@dataclass
class BoundaryContext:
    """Train-time labels used by BoundaryFeatureBlock."""

    batch_idx: torch.Tensor
    bboxes: torch.Tensor
    image_shape: tuple[int, int, int, int]


_BOUNDARY_CONTEXT: BoundaryContext | None = None
_BOUNDARY_ENABLED = True


def set_boundary_enabled(enabled: bool) -> None:
    """Enable or disable train-time boundary feature refinement."""

    global _BOUNDARY_ENABLED
    _BOUNDARY_ENABLED = enabled


def set_boundary_context(
    batch_idx: torch.Tensor | None,
    bboxes: torch.Tensor | None,
    image_shape: tuple[int, int, int, int],
) -> None:
    """Store normalized YOLO xywh labels for train-time boundary masks."""

    global _BOUNDARY_CONTEXT
    if batch_idx is None or bboxes is None:
        _BOUNDARY_CONTEXT = None
        return

    _BOUNDARY_CONTEXT = BoundaryContext(
        batch_idx=batch_idx.detach(),
        bboxes=bboxes.detach(),
        image_shape=image_shape,
    )


def clear_boundary_context() -> None:
    """Clear train-time boundary labels after a forward pass."""

    global _BOUNDARY_CONTEXT
    _BOUNDARY_CONTEXT = None


class BoundaryFeatureBlock(nn.Module):
    """Train-only GT-guided boundary/background feature refinement for one FPN level."""

    def __init__(
        self,
        c1: int,
        ring: float = 1.0,
        shrinkage: float = 0.25,
        reduction: int = 4,
        alpha_init: float = 0.1,
        alpha_max: float = 0.1,
    ) -> None:
        """Initialize a residual transform applied only to boundary and near-background cells."""

        super().__init__()
        hidden_channels = max(c1 // max(int(reduction), 1), 8)
        self.ring = max(float(ring), 0.0)
        self.shrinkage = max(float(shrinkage), 0.0)
        self.alpha_max = max(float(alpha_max), 0.0)
        limit = self.alpha_max * 0.999
        init = max(min(float(alpha_init), limit), -limit)
        init = 0.0 if self.alpha_max == 0 else torch.atanh(torch.tensor(init / self.alpha_max)).item()
        self.alpha = nn.Parameter(torch.tensor(init))
        self.transform = nn.Sequential(
            nn.Conv2d(c1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, c1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine boundary/background cells during training and stay identity otherwise."""

        if not _BOUNDARY_ENABLED or not self.training or _BOUNDARY_CONTEXT is None:
            return x

        mask = _build_boundary_background_mask(
            batch_idx=_BOUNDARY_CONTEXT.batch_idx,
            bboxes=_BOUNDARY_CONTEXT.bboxes,
            batch_size=x.shape[0],
            feature_height=x.shape[2],
            feature_width=x.shape[3],
            device=x.device,
            dtype=x.dtype,
            ring=self.ring,
            shrinkage=self.shrinkage,
        )
        alpha = torch.tanh(self.alpha).to(dtype=x.dtype) * self.alpha_max
        return x + alpha * self.transform(x) * mask


def _clamp_interval(start: float, end: float, limit: int) -> tuple[int, int]:
    """Convert a continuous feature interval to a non-empty clamped integer slice."""

    if limit <= 0:
        return 0, 0
    start_i = max(0, min(limit - 1, int(start)))
    end_i = max(start_i + 1, min(limit, int(end) + 1))
    return start_i, end_i


def _build_boundary_background_mask(
    batch_idx: torch.Tensor,
    bboxes: torch.Tensor,
    batch_size: int,
    feature_height: int,
    feature_width: int,
    device: torch.device,
    dtype: torch.dtype,
    ring: float,
    shrinkage: float,
) -> torch.Tensor:
    """Build a mask for dilated-boundary and near-background cells from normalized xywh GT boxes."""

    mask = torch.zeros((batch_size, 1, feature_height, feature_width), device=device, dtype=dtype)
    if bboxes.numel() == 0:
        return mask

    batch_idx = batch_idx.to(device=device, dtype=torch.long).view(-1)
    bboxes = bboxes.to(device=device, dtype=dtype)
    ring = max(float(ring), 0.0)
    near_ring = ring * 3.0

    for idx, box in zip(batch_idx.tolist(), bboxes, strict=False):
        if idx < 0 or idx >= batch_size:
            continue

        x_center, y_center, width, height = [float(v) for v in box]
        x1 = (x_center - width / 2.0) * feature_width
        y1 = (y_center - height / 2.0) * feature_height
        x2 = (x_center + width / 2.0) * feature_width
        y2 = (y_center + height / 2.0) * feature_height

        dx1, dx2 = _clamp_interval(x1 - ring, x2 + ring, feature_width)
        dy1, dy2 = _clamp_interval(y1 - ring, y2 + ring, feature_height)
        nx1, nx2 = _clamp_interval(x1 - near_ring, x2 + near_ring, feature_width)
        ny1, ny2 = _clamp_interval(y1 - near_ring, y2 + near_ring, feature_height)
        pad_x = min(max(x2 - x1, 0.0) * shrinkage, 0.5)
        pad_y = min(max(y2 - y1, 0.0) * shrinkage, 0.5)
        ix1, ix2 = _clamp_interval(x1 + pad_x, x2 - pad_x, feature_width)
        iy1, iy2 = _clamp_interval(y1 + pad_y, y2 - pad_y, feature_height)

        mask[idx, :, ny1:ny2, nx1:nx2] = 1.0
        mask[idx, :, dy1:dy2, dx1:dx2] = 1.0
        mask[idx, :, iy1:iy2, ix1:ix2] = 0.0

    return mask


class UpBlock(nn.Module):
    """SR-TOD reconstruction upsample block."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FeatureDGFE(nn.Module):
    """Image-space SR-DGFE for P2 feature enhancement."""

    def __init__(
        self,
        c1: int,
        reduction: int = 8,
        threshold_init: float = 0.0156862,
        sharpness: float = 10.0,
        alpha_init: float = 1e-3,
        alpha_max: float = 1.0,
        recon_ratio: float = 0.5,
        upsample_steps: int = 2,
    ) -> None:
        super().__init__()
        upsample_steps = max(int(upsample_steps), 1)
        channel_hidden = max(c1 // max(int(reduction), 1), 8)

        up_blocks = []
        in_channels = c1
        out_channels = max(int(c1 * float(recon_ratio)), 8)
        for _ in range(upsample_steps):
            up_blocks.append(UpBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels = max(out_channels // 2, 8)
        self.upsample = nn.Sequential(*up_blocks)
        self.reconstruct = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=3, padding=1), nn.Sigmoid())
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(c1, channel_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_hidden, c1, kernel_size=1),
        )
        self.threshold = nn.Parameter(torch.tensor(float(threshold_init)))
        self.sharpness = float(sharpness)
        self.alpha_max = max(float(alpha_max), 0.0)
        p = max(min(float(alpha_init) / max(self.alpha_max, 1e-12), 1.0 - 1e-6), 1e-6)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(p)))
        self.last_aux: dict[str, torch.Tensor] | None = None

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit) * self.alpha_max

    def forward(self, x: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        recon = self.reconstruct(self.upsample(x))
        if recon.shape[-2:] != img.shape[-2:]:
            recon = F.interpolate(recon, size=img.shape[-2:], mode="bilinear", align_corners=False)

        diff = (recon - img).abs().mean(dim=1, keepdim=True)
        spatial_logits_img = self.sharpness * (diff - self.threshold.to(dtype=diff.dtype, device=diff.device))
        spatial_logits = F.interpolate(spatial_logits_img, size=x.shape[-2:], mode="bilinear", align_corners=False)
        spatial_gate = 1.0 + torch.sigmoid(spatial_logits)

        avg_gate = self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
        max_gate = self.channel_mlp(F.adaptive_max_pool2d(x, 1))
        channel_gate = torch.sigmoid(avg_gate + max_gate)

        attention = channel_gate * spatial_gate
        alpha = self.alpha.to(dtype=x.dtype, device=x.device)
        out = x * (1.0 + alpha * (attention - 1.0))
        self.last_aux = (
            {
                "recon": recon,
                "spatial_logits": spatial_logits,
                "spatial_gate": spatial_gate,
                "alpha": alpha.reshape(1),
            }
            if self.training
            else None
        )
        return out


class MS_Scharr_EnSimAM(nn.Module):
    """Multi-scale local-variance EnSimAM with Scharr edge attention."""

    def __init__(self, lambd: float = 1e-4, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-6):
        """Initialize learnable local and branch fusion weights with fixed Scharr kernels."""
        super().__init__()
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.local_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.branch_weights = nn.Parameter(torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32))
        scharr_x = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]).view(1, 1, 3, 3)
        scharr_y = torch.tensor([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]]).view(1, 1, 3, 3)
        self.register_buffer("scharr_x", scharr_x, persistent=False)
        self.register_buffer("scharr_y", scharr_y, persistent=False)

    @staticmethod
    def local_variance(x: torch.Tensor, kernel_size: int, padding: int) -> torch.Tensor:
        """Compute local variance while preserving [B, C, H, W]."""
        local_mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
        return F.avg_pool2d((x - local_mean).pow(2), kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global, multi-kernel local, and Scharr edge attention."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        energy = (x - mean).pow(2) / (4 * (var + self.lambd)) + 0.5
        a_global = torch.sigmoid(1.0 / energy)

        local_var_3 = self.local_variance(x, kernel_size=3, padding=1)
        local_var_5 = self.local_variance(x, kernel_size=5, padding=2)
        local_var_7 = self.local_variance(x, kernel_size=7, padding=3)
        local_weights = torch.softmax(self.local_weights, dim=0)
        local_var = local_weights[0] * local_var_3 + local_weights[1] * local_var_5 + local_weights[2] * local_var_7
        a_local = torch.sigmoid(self.alpha * local_var)

        c = x.shape[1]
        scharr_x = self.scharr_x.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        scharr_y = self.scharr_y.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        gx = F.conv2d(x, scharr_x, padding=1, groups=c)
        gy = F.conv2d(x, scharr_y, padding=1, groups=c)
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + self.eps)
        a_edge = torch.sigmoid(self.beta * edge)

        branch_weights = torch.softmax(self.branch_weights, dim=0)
        attention = branch_weights[0] * a_global + branch_weights[1] * a_local + branch_weights[2] * a_edge
        return x * attention


class AdversarialPerturbationInjection(nn.Module):
    """Train-only gradient-based feature perturbation injection for one FPN level."""

    def __init__(
        self,
        c1: int,
        rho: float = 0.02,
        api_weight: float = 0.25,
        target_mode: str = "foreground",
        eps: float = 1e-6,
        use_partial_forward: bool = False,
        use_rho_warmup: bool = False,
        warmup_epochs: int = 10,
        use_per_box_norm: bool = False,
        use_fgsm_dropout: bool = False,
        fgsm_drop_rate: float = 0.1,
    ) -> None:
        """Initialize SET-style API parameters.

        The perturbation is supplied by the training loss path from the
        auxiliary-loss gradient w.r.t. the captured feature. A small auxiliary
        head scores the perturbed P2 feature without a second full detector forward.

        Flags (all default False = original behavior):
            use_partial_forward: Re-run only layers from this layer onward for perturbed
                forward, skipping the backbone. Requires tasks.py to cache clean y-list
                and set ``_cached_clean_y`` before the perturbed pass.
            use_rho_warmup: Linearly warm-up ``rho`` and ``api_weight`` from 10 % of their
                target values over ``warmup_epochs`` epochs. Set ``_epoch`` each epoch from
                the trainer.
            warmup_epochs: Number of epochs for the rho/weight warm-up schedule.
            use_per_box_norm: Normalize the adversarial gradient per GT bounding-box region
                rather than globally, amplifying the perturbation for small objects.
            use_fgsm_dropout: In perturb mode, zero out the ``fgsm_drop_rate`` fraction of
                channels with the highest perturbation magnitude, forcing the model to learn
                redundant representations for robust detection.
            fgsm_drop_rate: Fraction of channels to drop (0.0-1.0). Active only when
                ``use_fgsm_dropout=True`` and in training perturb mode.
        """

        super().__init__()
        self.rho = max(float(rho), 0.0)
        self.api_weight = max(float(api_weight), 0.0)
        self.target_mode = str(target_mode)
        self.eps = max(float(eps), 1e-12)

        # --- Feature flags (all off = original behavior) ---
        self.use_partial_forward = bool(use_partial_forward)
        self.use_rho_warmup = bool(use_rho_warmup)
        self.warmup_epochs = max(1, int(warmup_epochs))
        self.use_per_box_norm = bool(use_per_box_norm)
        self.use_fgsm_dropout = bool(use_fgsm_dropout)
        self.fgsm_drop_rate = float(max(0.0, min(1.0, fgsm_drop_rate)))

        # --- Aux head (used only in foreground/boundary target_mode) ---
        self.aux_head = nn.Conv2d(c1, 1, kernel_size=1)

        # --- Runtime state ---
        self.mode = "off"
        self.captured: torch.Tensor | None = None
        self.perturbation: torch.Tensor | None = None
        self.last_perturbation_norm: torch.Tensor | None = None

        # Partial forward state (populated from tasks.py)
        self.layer_idx: int | None = None          # position of this module in self.model
        self._clean_input: torch.Tensor | None = None     # input tensor to this layer in clean pass
        self._cached_clean_y: list | None = None   # full y-list from clean _predict_once

        # Rho warmup state (epoch synced from trainer)
        self._epoch: int = 0

    # ------------------------------------------------------------------
    # Scheduled hyper-parameters
    # ------------------------------------------------------------------

    @property
    def current_rho(self) -> float:
        """Return rho after applying linear warm-up if enabled."""
        if not self.use_rho_warmup or self._epoch >= self.warmup_epochs:
            return self.rho
        t = self._epoch / self.warmup_epochs  # 0 -> 1 over warmup_epochs
        return self.rho * (0.1 + 0.9 * t)

    @property
    def current_api_weight(self) -> float:
        """Return api_weight after applying linear warm-up if enabled."""
        if not self.use_rho_warmup or self._epoch >= self.warmup_epochs:
            return self.api_weight
        t = self._epoch / self.warmup_epochs
        return self.api_weight * (0.1 + 0.9 * t)

    # ------------------------------------------------------------------
    # State control
    # ------------------------------------------------------------------

    def clear_state(self) -> None:
        """Clear captured feature, perturbation, and partial-forward cache."""

        self.mode = "off"
        self.captured = None
        self.perturbation = None
        self._clean_input = None
        self._cached_clean_y = None

    def capture(self) -> None:
        """Capture the next forward feature without perturbing it."""

        self.clear_state()
        self.mode = "capture"

    def perturb(self) -> None:
        """Apply the stored perturbation on the next forward."""

        self.mode = "perturb"

    # ------------------------------------------------------------------
    # Perturbation construction
    # ------------------------------------------------------------------

    def set_perturbation_from_grad(
        self,
        grad: torch.Tensor | None,
        bboxes: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
    ) -> bool:
        """Create an adversarial perturbation from a feature gradient.

        When ``use_per_box_norm`` is enabled and valid bboxes are provided, the
        gradient is weighted per GT box region before L2 normalisation.  Small
        boxes receive a higher weight (proportional to 1/sqrt(area)) so that the
        perturbation is not diluted by large background regions.

        Args:
            grad: Gradient tensor [B, C, H, W] w.r.t. the captured feature.
            bboxes: Normalised xywh GT boxes [N, 4] (optional, for per-box norm).
            batch_idx: Integer sample indices [N] matching each box to a batch item.

        Returns:
            True if the perturbation is finite and was stored; False otherwise.
        """

        if grad is None or self.current_rho == 0 or self.current_api_weight == 0:
            self.perturbation = None
            return False

        if self.use_per_box_norm and bboxes is not None and bboxes.numel() > 0 and batch_idx is not None:
            B, C, H, W = grad.shape
            weight_map = torch.ones(B, 1, H, W, device=grad.device, dtype=torch.float32)
            _batch_idx = batch_idx.to(device=grad.device, dtype=torch.long).view(-1)
            _bboxes = bboxes.to(device=grad.device, dtype=torch.float32)
            for b_idx, box in zip(_batch_idx.tolist(), _bboxes.tolist()):
                if b_idx < 0 or b_idx >= B:
                    continue
                xc, yc, bw, bh = box
                x1 = int((xc - bw / 2) * W)
                x2 = max(x1 + 1, int((xc + bw / 2) * W))
                y1 = int((yc - bh / 2) * H)
                y2 = max(y1 + 1, int((yc + bh / 2) * H))
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)
                box_area = max(1, (x2 - x1) * (y2 - y1))
                # Small objects get amplified (inversely proportional to sqrt(area))
                scale = float(H * W / box_area) ** 0.5
                weight_map[b_idx, :, y1:y2, x1:x2] *= scale
            weighted_grad = grad.detach().float() * weight_map
            flat = weighted_grad.flatten(1)
            norm = flat.norm(p=2, dim=1).clamp(min=self.eps).view(-1, 1, 1, 1)
            perturbation = weighted_grad / norm * self.current_rho
        else:
            # Default: global L2 norm (original behavior)
            flat = grad.detach().float().flatten(1)
            norm = flat.norm(p=2, dim=1).clamp(min=self.eps).view(-1, 1, 1, 1)
            perturbation = grad.detach().float() / norm * self.current_rho

        self.perturbation = perturbation.to(dtype=grad.dtype, device=grad.device)
        self.last_perturbation_norm = self.perturbation.detach().float().flatten(1).norm(p=2, dim=1)
        return bool(torch.isfinite(self.perturbation).all())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Capture or perturb a feature during training; stay identity otherwise."""

        if not self.training:
            return x
        if self.mode == "capture":
            self._clean_input = x   # cache for partial forward
            self.captured = x
            if x.requires_grad:
                x.retain_grad()
            return x
        if self.mode == "perturb" and self.perturbation is not None:
            out = x + self.perturbation.to(device=x.device, dtype=x.dtype)
            if self.use_fgsm_dropout:
                # Channel-wise structured dropout: suppress top-fgsm_drop_rate channels
                # that have the highest perturbation magnitude.  Forces the model to
                # learn redundant, attack-resistant feature representations.
                ch_mag = self.perturbation.abs().mean(dim=(2, 3))  # [B, C]
                k = max(1, int(ch_mag.shape[1] * self.fgsm_drop_rate))
                # Per-sample threshold: top-k channel magnitude
                thresh = ch_mag.topk(k, dim=1).values[:, -1].view(-1, 1, 1, 1)  # [B,1,1,1]
                keep_mask = (ch_mag.unsqueeze(-1).unsqueeze(-1) < thresh).to(dtype=out.dtype)  # [B,C,1,1]
                keep_frac = max(1.0 - self.fgsm_drop_rate, 1e-3)
                out = out * keep_mask / keep_frac
            return out
        return x

    # ------------------------------------------------------------------
    # Auxiliary head (used only in foreground / boundary target_mode)
    # ------------------------------------------------------------------

    def auxiliary_logits(self, feature: torch.Tensor | None = None) -> torch.Tensor:
        """Return auxiliary objectness logits for a captured or supplied feature."""

        if feature is None:
            if self.captured is None:
                raise RuntimeError("API auxiliary_logits requires captured feature.")
            feature = self.captured
        return self.aux_head(feature)

    def auxiliary_loss(self, target: torch.Tensor, feature: torch.Tensor | None = None) -> torch.Tensor:
        """Compute BCE objectness loss for the auxiliary P2 API branch."""

        logits = self.auxiliary_logits(feature)
        target = target.to(device=logits.device, dtype=logits.dtype)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
        return F.binary_cross_entropy_with_logits(logits, target)

    def adversarial_auxiliary_loss(self, target: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss on the captured feature plus stored perturbation."""

        if self.captured is None or self.perturbation is None:
            raise RuntimeError("API adversarial_auxiliary_loss requires captured feature and perturbation.")
        feature = self.captured + self.perturbation.to(device=self.captured.device, dtype=self.captured.dtype)
        return self.auxiliary_loss(target, feature=feature)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class WeightedAdd(nn.Module):
    """BiFPN-style weighted sum for already aligned feature maps."""

    def __init__(self, n_inputs: int = 2, eps: float = 1e-4):
        """Initialize learnable non-negative fusion weights."""
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Fuse tensors with normalized positive weights."""
        weights = torch.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        out = inputs[0] * weights[0]
        for i, x in enumerate(inputs[1:], start=1):
            out = out + x * weights[i]
        return out


class ScalSeq(nn.Module):
    """Attentional Scale Fusion sequence block for multi-scale feature fusion."""

    def __init__(self, ch: list[int] | tuple[int, ...], c2: int, kernel_size: int = 3):
        """Initialize lateral projections and a 3D convolution over the scale dimension."""
        super().__init__()
        self.lateral = nn.ModuleList(Conv(c, c2, 1, 1) for c in ch)
        padding = kernel_size // 2
        self.conv3d = nn.Conv3d(c2, c2, kernel_size=(kernel_size, 1, 1), padding=(padding, 0, 0), bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Fuse multi-scale tensors into a single [B, C, H, W] feature map."""
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        target_size = inputs[0].shape[2:]
        features = []
        for x, lateral in zip(inputs, self.lateral):
            x = lateral(x)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="nearest")
            features.append(x)
        x = torch.stack(features, dim=2)
        x = self.act(self.bn(self.conv3d(x)))
        return F.max_pool3d(x, kernel_size=(x.shape[2], 1, 1)).squeeze(2)


class ASFAttention(nn.Module):
    """ASF refinement block with channel attention and local spatial attention."""

    def __init__(self, c1: int, reduction: int = 16, local_kernel: int = 7):
        """Initialize channel and local attention branches."""
        super().__init__()
        hidden = max(c1 // reduction, 8)
        padding = local_kernel // 2
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=False),
        )
        self.local = nn.Conv2d(2, 1, local_kernel, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine important channels and local regions without changing shape."""
        avg_attn = self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
        max_attn = self.channel_mlp(F.adaptive_max_pool2d(x, 1))
        x = x * self.sigmoid(avg_attn + max_attn)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.local(torch.cat((avg_map, max_map), dim=1)))


class EnSimAM(nn.Module):
    """Enhanced SimAM attention with global, local-variance, and edge-response branches."""

    def __init__(self, lambd: float = 1e-4, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-6):
        """Initialize parameter-free attention coefficients and fixed Sobel kernels."""
        super().__init__()
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-free attention without changing tensor shape."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        energy = (x - mean).pow(2) / (4 * (var + self.lambd)) + 0.5
        a_global = torch.sigmoid(1.0 / energy)

        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((x - local_mean).pow(2), kernel_size=3, stride=1, padding=1)
        a_local = torch.sigmoid(self.alpha * local_var)

        c = x.shape[1]
        sobel_x = self.sobel_x.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        gx = F.conv2d(x, sobel_x, padding=1, groups=c)
        gy = F.conv2d(x, sobel_y, padding=1, groups=c)
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + self.eps)
        a_edge = torch.sigmoid(self.beta * edge)

        attention = 0.5 * a_global + 0.25 * a_local + 0.25 * a_edge
        return x * attention



def _choose_attention_heads(channels: int, requested_heads: int) -> int:
    """Pick a valid attention head count that divides channels."""
    requested_heads = max(1, min(int(requested_heads), int(channels)))
    for heads in range(requested_heads, 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class KVCompressedAttention(nn.Module):
    """Self-attention with full-resolution queries and spatially compressed keys/values.

    Supports multiple K/V spatial compression modes:
    - avg: AvgPool + GroupNorm for K and V.
    - avg_dwk/dwconv: AvgPool + DWConv + GroupNorm for K, AvgPool + GroupNorm for V.
    - dw_stride: stride depthwise Conv + GroupNorm for K and V.
    - group_weight: learned softmax weighting inside each sr_ratio x sr_ratio group.
    Attention is computed via ``F.scaled_dot_product_attention`` which automatically
    dispatches to FlashAttention v2 on supported CUDA hardware.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
        attn_drop: float = 0.0,
        residual: bool = True,
    ):
        """Initialize KV-compressed attention.

        Args:
            c1: Input channels.
            c2: Output channels.
            num_heads: Requested attention heads. Reduced if it does not divide c2.
            sr_ratio: Spatial compression ratio for K/V tokens.
            mode: ``avg``, ``avg_dwk``, ``dw_stride``, ``dwconv``, or ``group_weight`` compression.
            attn_drop: Dropout probability applied to attention weights during training.
            residual: Whether to add the projected attention output back to the input projection.
        """
        super().__init__()
        if mode == "dwconv":
            mode = "avg_dwk"
        if mode not in {"avg", "avg_dwk", "dw_stride", "group_weight"}:
            raise ValueError(f"Unsupported KV compression mode: {mode}")

        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = max(1, int(sr_ratio))
        self.mode = mode
        self.attn_drop_p = attn_drop
        self.residual = residual

        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.q = nn.Conv2d(c2, c2, 1, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)  # stabilize Q logits before SDPA

        if self.sr_ratio > 1 and self.mode == "avg":
            self.k_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.GroupNorm(min(32, c2), c2),
            )
            self.v_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.GroupNorm(min(32, c2), c2),
            )
        elif self.sr_ratio > 1 and self.mode == "avg_dwk":
            # No activation - K must stay linear for well-formed attention logits.
            self.k_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
                nn.GroupNorm(min(32, c2), c2),
            )
            self.v_compress = nn.Sequential(
                nn.AvgPool2d(self.sr_ratio, self.sr_ratio),
                nn.GroupNorm(min(32, c2), c2),
            )
        elif self.sr_ratio > 1 and self.mode == "dw_stride":
            self.k_compress = nn.Sequential(
                nn.Conv2d(c2, c2, 3, self.sr_ratio, 1, groups=c2, bias=False),
                nn.GroupNorm(min(32, c2), c2),
            )
            self.v_compress = nn.Sequential(
                nn.Conv2d(c2, c2, 3, self.sr_ratio, 1, groups=c2, bias=False),
                nn.GroupNorm(min(32, c2), c2),
            )
        else:
            self.k_compress = nn.Identity()
            self.v_compress = nn.Identity()

        # Separate linear projections for K and V (no shared kv conv, no activation)
        self.k_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.v_proj = nn.Conv2d(c2, c2, 1, bias=False)

        # group_weight path keeps its own scorer (unchanged)
        self.group_score = nn.Linear(c2, 1) if self.mode == "group_weight" and self.sr_ratio > 1 else None
        # Shared kv conv kept for group_weight compatibility
        self.kv = nn.Conv2d(c2, c2 * 2, 1, bias=False) if self.mode == "group_weight" else None

        self.proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c2)

    def _compress_group_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Compress each sr_ratio x sr_ratio token group with learned softmax weights."""
        if self.sr_ratio <= 1:
            return x
        b, c, h, w = x.shape
        pad_h = (self.sr_ratio - h % self.sr_ratio) % self.sr_ratio
        pad_w = (self.sr_ratio - w % self.sr_ratio) % self.sr_ratio
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        gh, gw = hp // self.sr_ratio, wp // self.sr_ratio
        tokens = x.view(b, c, gh, self.sr_ratio, gw, self.sr_ratio).permute(0, 2, 4, 3, 5, 1).contiguous()
        tokens = tokens.view(b, gh, gw, self.sr_ratio * self.sr_ratio, c)
        weights = self.group_score(tokens).softmax(dim=3)
        compressed = (tokens * weights).sum(dim=3)
        return compressed.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply KV-compressed attention and return a BCHW tensor."""
        x = self.input_proj(x)
        b, c, h, w = x.shape

        # Q: full-resolution, normalized per head
        q = self.q(x).flatten(2).transpose(1, 2)  # [B, H*W, C]
        q = q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, H*W, hd]
        q = self.q_norm(q)

        if self.mode == "group_weight" and self.group_score is not None:
            # group_weight path: unchanged (shared kv compress)
            kv_source = self._compress_group_weight(x)
            kv = self.kv(kv_source).flatten(2).transpose(1, 2)
            kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            # Non-group-weight paths use separate K and V compression and projection.
            k_src = self.k_compress(x)  # [B, C, H/sr, W/sr]
            v_src = self.v_compress(x)  # [B, C, H/sr, W/sr]
            k = self.k_proj(k_src)  # no activation
            v = self.v_proj(v_src)  # no activation
            # reshape to [B, nh, tokens, hd]
            def _to_heads(t):
                n = t.shape[2] * t.shape[3]
                return t.flatten(2).transpose(1, 2).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = _to_heads(k)
            v = _to_heads(v)

        # Flash Attention via PyTorch SDPA (dispatches to FlashAttn v2 on CUDA)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            scale=self.scale,
        )

        out = out.transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        out = self.proj_bn(self.proj(out))
        return x + out if self.residual else out


class KVCompressedTransformerEncoder(nn.Module):
    """Pre-norm transformer encoder block using KV-compressed self-attention and a DW-PW convolutional FFN.

    FFN structure: PW-expand → DW-spatial-mix → PW-project, giving each token access to its
    8-connected spatial neighborhood while keeping the channel mixing role of the outer PWs.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
        attn_drop: float = 0.0,
        mlp_ratio: float = 2.0,
    ):
        """Initialize LayerNorm-KVCA and LayerNorm-DW-FFN residual branches.

        Args:
            c1: Input channels.
            c2: Output channels.
            num_heads: Requested attention heads.
            sr_ratio: Spatial compression ratio passed to KVCompressedAttention.
            mode: KV compression mode (``dwconv`` or ``group_weight``).
            attn_drop: Attention dropout probability (training only).
            mlp_ratio: Hidden dim multiplier for the FFN.
        """
        super().__init__()
        hidden = max(c2, int(c2 * mlp_ratio))
        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.norm1 = LayerNorm2d(c2)
        self.attn = KVCompressedAttention(c2, c2, num_heads, sr_ratio, mode, attn_drop=attn_drop, residual=False)
        self.norm2 = LayerNorm2d(c2)
        # DW-PW FFN: channel expand → spatial mix → channel project
        self.ffn = nn.Sequential(
            Conv(c2, hidden, 1, 1),                    # PW: channel expand
            Conv(hidden, hidden, 3, 1, g=hidden),      # DW: spatial mix in 3×3 neighborhood
            Conv(hidden, c2, 1, 1, act=False),         # PW: channel project
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LN -> KVCA -> residual, then LN -> DW-FFN -> residual."""
        x = self.input_proj(x)
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class KVCompressedAttentionPartial(nn.Module):
    """PSA-style partial KV-compressed attention.

    Splits input channels in half: one half passes through ``KVCompressedAttention``
    (with a residual connection), the other half bypasses unchanged.  The two halves
    are concatenated and projected back to ``c2`` channels with a pointwise Conv.

    Benefits at high-resolution feature maps (P2/P3):
    - Attention head_dim is halved → compute reduced ~50 %.
    - Bypass half retains fine-grained local texture untouched by attention.
    - Parameter overhead is small: only an extra ``Conv(c2, c2, 1)`` output projection.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
        attn_drop: float = 0.0,
    ):
        """Initialize partial KVCA.

        Args:
            c1: Input channels.
            c2: Output channels (must be even).
            num_heads: Attention heads for the *attention* half; clipped to c2//2.
            sr_ratio: Spatial compression ratio passed to ``KVCompressedAttention``.
            mode: KV compression mode (``dwconv`` or ``group_weight``).
            attn_drop: Attention dropout probability (training only).
        """
        super().__init__()
        if c2 % 2 != 0:
            raise ValueError(f"KVCompressedAttentionPartial requires even c2, got {c2}")
        c_attn = c2 // 2
        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.attn = KVCompressedAttention(
            c_attn,
            c_attn,
            num_heads=max(1, num_heads // 2),  # half heads for half channels
            sr_ratio=sr_ratio,
            mode=mode,
            attn_drop=attn_drop,
            residual=True,
        )
        # Pointwise mix after concat – no activation to stay linear
        self.out_proj = Conv(c2, c2, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply KVCA on first half of channels, bypass second half, mix outputs."""
        x = self.input_proj(x)
        x_attn, x_bypass = x.chunk(2, dim=1)
        x_attn = self.attn(x_attn)
        return self.out_proj(torch.cat([x_attn, x_bypass], dim=1))


class _TopKGroupKVAttentionBase(nn.Module):
    """Shared utilities for top-k grouped K/V attention blocks."""

    def __init__(self, c1: int, c2: int, num_heads: int, group_size: int):
        super().__init__()
        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = self.head_dim**-0.5
        self.group_size = max(1, int(group_size))

        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.q = nn.Conv2d(c2, c2, 1, bias=False)
        self.kv = nn.Conv2d(c2, c2 * 2, 1, bias=False)
        self.score = nn.Conv2d(c2, 1, 3, padding=1, bias=True)
        self.proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c2)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
        """Pad H/W to a multiple and return the original H/W."""
        h, w = x.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, h, w

    def _to_groups(self, x: torch.Tensor, group_h: int, group_w: int) -> torch.Tensor:
        """Convert BCHW into B x groups x group_tokens x C."""
        b, c, _, _ = x.shape
        g = self.group_size
        return (
            x.view(b, c, group_h, g, group_w, g)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(b, group_h * group_w, g * g, c)
        )

    @staticmethod
    def _to_regions(x: torch.Tensor, region_size: int, region_h: int, region_w: int) -> torch.Tensor:
        """Convert BCHW into B x regions x region_tokens x C."""
        b, c, _, _ = x.shape
        r = region_size
        return (
            x.view(b, c, region_h, r, region_w, r)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(b, region_h * region_w, r * r, c)
        )

    def _compress_kv_groups(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress projected K/V maps into one weighted token per spatial group."""
        padded, _, _ = self._pad_to_multiple(x, self.group_size)
        _, _, hp, wp = padded.shape
        group_h, group_w = hp // self.group_size, wp // self.group_size

        k_map, v_map = self.kv(padded).chunk(2, dim=1)
        score_map = self.score(padded)
        k_tokens = self._to_groups(k_map, group_h, group_w)
        v_tokens = self._to_groups(v_map, group_h, group_w)
        score_tokens = self._to_groups(score_map, group_h, group_w).squeeze(-1)
        weights = score_tokens.softmax(dim=2).unsqueeze(-1)

        k_groups = (k_tokens * weights).sum(dim=2)
        v_groups = (v_tokens * weights).sum(dim=2)
        group_scores = score_tokens.mean(dim=2)
        return k_groups, v_groups, group_scores

    def _format_full_q(self, q_map: torch.Tensor) -> torch.Tensor:
        """Convert full-resolution Q map to B x heads x tokens x head_dim."""
        b, c, h, w = q_map.shape
        q = q_map.flatten(2).transpose(1, 2)
        return q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _format_selected_kv(self, selected: torch.Tensor) -> torch.Tensor:
        """Convert B x tokens x C selected K/V tokens to B x heads x tokens x head_dim."""
        b, n, _ = selected.shape
        return selected.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)


class TopKGlobalGroupKVAttention(_TopKGroupKVAttentionBase):
    """Full-query attention over a global top-k set of compressed K/V groups."""

    def __init__(self, c1: int, c2: int, num_heads: int = 4, group_size: int = 4, topk: int = 100):
        """Initialize global top-k grouped K/V attention."""
        super().__init__(c1, c2, num_heads, group_size)
        self.topk = max(1, int(topk))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select one top-k group set per image and attend all query tokens to it."""
        x = self.input_proj(x)
        b, c, h, w = x.shape
        q = self._format_full_q(self.q(x))
        k_groups, v_groups, group_scores = self._compress_kv_groups(x)

        route_count = min(self.topk, k_groups.shape[1])
        route_idx = group_scores.topk(route_count, dim=-1).indices
        batch_idx = torch.arange(b, device=x.device)[:, None]
        k = self._format_selected_kv(k_groups[batch_idx, route_idx])
        v = self._format_selected_kv(v_groups[batch_idx, route_idx])

        with torch.cuda.amp.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
        out = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj_bn(self.proj(out))


class TopKAdaptiveGroupKVAttention(_TopKGroupKVAttentionBase):
    """Region-wise query attention over adaptive top-k compressed K/V groups."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        group_size: int = 4,
        query_region_size: int = 10,
        topk: int = 8,
    ):
        """Initialize adaptive top-k grouped K/V attention."""
        super().__init__(c1, c2, num_heads, group_size)
        self.query_region_size = max(1, int(query_region_size))
        self.topk = max(1, int(topk))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select top-k K/V groups per query region, then attend region tokens to them."""
        x = self.input_proj(x)
        b, c, _, _ = x.shape
        q_map = self.q(x)
        q_padded, orig_h, orig_w = self._pad_to_multiple(q_map, self.query_region_size)
        _, _, hp, wp = q_padded.shape
        region_h, region_w = hp // self.query_region_size, wp // self.query_region_size
        num_regions = region_h * region_w
        tokens_per_region = self.query_region_size * self.query_region_size

        q_regions = self._to_regions(q_padded, self.query_region_size, region_h, region_w)
        q_repr = q_regions.mean(dim=2)
        k_groups, v_groups, _ = self._compress_kv_groups(x)
        affinity = (q_repr @ k_groups.transpose(-2, -1)) * (c**-0.5)

        route_count = min(self.topk, k_groups.shape[1])
        route_idx = affinity.topk(route_count, dim=-1).indices
        batch_idx = torch.arange(b, device=x.device)[:, None]

        outputs = []
        for region_idx in range(num_regions):
            selected = route_idx[:, region_idx]
            q_tokens = q_regions[:, region_idx].reshape(b, tokens_per_region, self.num_heads, self.head_dim)
            q_tokens = q_tokens.permute(0, 2, 1, 3)
            k_tokens = self._format_selected_kv(k_groups[batch_idx, selected])
            v_tokens = self._format_selected_kv(v_groups[batch_idx, selected])
            with torch.cuda.amp.autocast(enabled=False):
                attn = (q_tokens.float() @ k_tokens.float().transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
            out = (attn.to(v_tokens.dtype) @ v_tokens).transpose(1, 2).reshape(b, tokens_per_region, c)
            outputs.append(out)

        out_regions = torch.stack(outputs, dim=1)
        r = self.query_region_size
        out = out_regions.view(b, region_h, region_w, r, r, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(b, c, hp, wp)[:, :, :orig_h, :orig_w]
        return x + self.proj_bn(self.proj(out))


class BiLevelRoutingAttention(nn.Module):
    """NCHW bi-level routing attention adapted for YOLO feature maps."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_heads: int = 4,
        n_win: int = 7,
        topk: int = 4,
        side_dwconv: int = 3,
    ):
        """Initialize bi-level routing attention."""
        super().__init__()
        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = c2**-0.5
        self.n_win = max(1, int(n_win))
        self.topk = max(1, int(topk))

        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.qkv_linear = nn.Conv2d(c2, c2 * 3, 1)
        self.lepe = (
            nn.Conv2d(c2, c2, side_dwconv, stride=1, padding=side_dwconv // 2, groups=c2)
            if side_dwconv > 0
            else None
        )
        self.output_linear = nn.Conv2d(c2, c2, 1, bias=False)
        self.output_bn = nn.BatchNorm2d(c2)

    @staticmethod
    def _pad_to_region_size(x: torch.Tensor, region_size: tuple[int, int]) -> tuple[torch.Tensor, int, int]:
        """Pad H/W to multiples of region_size and return original H/W."""
        h, w = x.shape[-2:]
        pad_h = (region_size[0] - h % region_size[0]) % region_size[0]
        pad_w = (region_size[1] - w % region_size[1]) % region_size[1]
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, h, w

    @staticmethod
    def _grid2seq(x: torch.Tensor, region_size: tuple[int, int], num_heads: int) -> tuple[torch.Tensor, int, int]:
        """Convert BCHW to B x heads x regions x region_tokens x head_dim."""
        b, c, h, w = x.shape
        rh, rw = region_size
        region_h, region_w = h // rh, w // rw
        x = x.view(b, num_heads, c // num_heads, region_h, rh, region_w, rw)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
        return x.view(b, num_heads, region_h * region_w, rh * rw, c // num_heads), region_h, region_w

    @staticmethod
    def _seq2grid(x: torch.Tensor, region_h: int, region_w: int, region_size: tuple[int, int]) -> torch.Tensor:
        """Convert B x heads x regions x region_tokens x head_dim to BCHW."""
        b, num_heads, _, _, head_dim = x.shape
        rh, rw = region_size
        x = x.view(b, num_heads, region_h, region_w, rh, rw, head_dim)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        return x.view(b, num_heads * head_dim, region_h * rh, region_w * rw)

    def _regional_routing_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        region_graph: torch.Tensor,
        region_size: tuple[int, int],
    ) -> torch.Tensor:
        """Apply token attention from each query region to selected key/value regions."""
        query, orig_h, orig_w = self._pad_to_region_size(query, region_size)
        key, _, _ = self._pad_to_region_size(key, region_size)
        value, _, _ = self._pad_to_region_size(value, region_size)

        query, q_region_h, q_region_w = self._grid2seq(query, region_size, self.num_heads)
        key, _, _ = self._grid2seq(key, region_size, self.num_heads)
        value, _, _ = self._grid2seq(value, region_size, self.num_heads)

        b, num_heads, q_regions, topk = region_graph.shape
        _, _, kv_regions, kv_region_tokens, head_dim = key.shape
        index = region_graph.view(b, num_heads, q_regions, topk, 1, 1).expand(
            -1, -1, -1, -1, kv_region_tokens, head_dim
        )
        key_g = torch.gather(
            key.view(b, num_heads, 1, kv_regions, kv_region_tokens, head_dim).expand(
                -1, -1, q_regions, -1, -1, -1
            ),
            dim=3,
            index=index,
        )
        value_g = torch.gather(
            value.view(b, num_heads, 1, kv_regions, kv_region_tokens, head_dim).expand(
                -1, -1, q_regions, -1, -1, -1
            ),
            dim=3,
            index=index,
        )

        with torch.cuda.amp.autocast(enabled=False):
            attn = (query.float() * self.scale) @ key_g.flatten(-3, -2).float().transpose(-1, -2)
            attn = attn.softmax(dim=-1)
        out = attn.to(value_g.dtype) @ value_g.flatten(-3, -2)
        out = self._seq2grid(out, q_region_h, q_region_w, region_size)
        return out[:, :, :orig_h, :orig_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route query regions to top-k key/value regions and apply token attention."""
        x = self.input_proj(x)
        _, c, h, w = x.shape
        region_size = (max(1, h // self.n_win), max(1, w // self.n_win))

        qkv = self.qkv_linear(x)
        q, k, v = qkv.chunk(3, dim=1)

        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        q_r = q_r.permute(0, 2, 3, 1).flatten(1, 2)
        k_r = k_r.flatten(2, 3)
        affinity = q_r @ k_r
        route_count = min(self.topk, k_r.shape[-1])
        route_idx = affinity.topk(route_count, dim=-1).indices
        route_idx = route_idx.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        out = self._regional_routing_attention(q, k, v, route_idx, region_size)
        lepe = self.lepe(v) if self.lepe is not None else torch.zeros_like(v)
        out = out + lepe
        return x + self.output_bn(self.output_linear(out.reshape(-1, c, h, w)))


class RegionRoutingAttentionLite(nn.Module):
    """Bi-level routing attention for YOLO feature maps using pure PyTorch ops."""

    def __init__(self, c1: int, c2: int, num_heads: int = 4, region_size: int = 10, topk: int = 4):
        """Initialize region-routed token attention."""
        super().__init__()
        self.c2 = c2
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.head_dim = c2 // self.num_heads
        self.scale = self.head_dim**-0.5
        self.region_size = max(1, int(region_size))
        self.topk = max(1, int(topk))

        self.input_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.qkv = nn.Conv2d(c2, c2 * 3, 1, bias=False)
        self.proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c2)

    def _pad_to_regions(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Pad H/W so both are divisible by region_size."""
        h, w = x.shape[-2:]
        pad_h = (self.region_size - h % self.region_size) % self.region_size
        pad_w = (self.region_size - w % self.region_size) % self.region_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, h, w

    def _to_regions(self, x: torch.Tensor, region_h: int, region_w: int) -> torch.Tensor:
        """Convert BCHW into B x regions x tokens x C."""
        b, c, _, _ = x.shape
        r = self.region_size
        return (
            x.view(b, c, region_h, r, region_w, r)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(b, region_h * region_w, r * r, c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route each query region to top-k key/value regions, then attend locally."""
        x = self.input_proj(x)
        b, c, h, w = x.shape
        padded, orig_h, orig_w = self._pad_to_regions(x)
        _, _, hp, wp = padded.shape
        region_h, region_w = hp // self.region_size, wp // self.region_size
        num_regions = region_h * region_w
        tokens_per_region = self.region_size * self.region_size

        qkv = self.qkv(padded)
        q, k, v = qkv.chunk(3, dim=1)
        q_regions = self._to_regions(q, region_h, region_w)
        k_regions = self._to_regions(k, region_h, region_w)
        v_regions = self._to_regions(v, region_h, region_w)

        q_repr = q_regions.mean(dim=2)
        k_repr = k_regions.mean(dim=2)
        affinity = (q_repr @ k_repr.transpose(-2, -1)) * (c**-0.5)
        route_count = min(self.topk, num_regions)
        route_idx = affinity.topk(route_count, dim=-1).indices

        outputs = []
        batch_idx = torch.arange(b, device=x.device)[:, None]
        for region_idx in range(num_regions):
            selected = route_idx[:, region_idx]
            q_tokens = q_regions[:, region_idx].reshape(b, tokens_per_region, self.num_heads, self.head_dim)
            q_tokens = q_tokens.permute(0, 2, 1, 3)
            k_tokens = k_regions[batch_idx, selected].reshape(
                b, route_count * tokens_per_region, self.num_heads, self.head_dim
            )
            v_tokens = v_regions[batch_idx, selected].reshape(
                b, route_count * tokens_per_region, self.num_heads, self.head_dim
            )
            k_tokens = k_tokens.permute(0, 2, 1, 3)
            v_tokens = v_tokens.permute(0, 2, 1, 3)
            with torch.cuda.amp.autocast(enabled=False):
                attn = (q_tokens.float() @ k_tokens.float().transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
            out = (attn.to(v_tokens.dtype) @ v_tokens).transpose(1, 2).reshape(b, tokens_per_region, c)
            outputs.append(out)

        out_regions = torch.stack(outputs, dim=1)
        r = self.region_size
        out = out_regions.view(b, region_h, region_w, r, r, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(b, c, hp, wp)[:, :, :orig_h, :orig_w]
        return x + self.proj_bn(self.proj(out))


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fCBAM(C2f):
    """C2f block followed by CBAM channel and spatial attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        kernel_size: int = 7,
    ):
        """Initialize C2f with CBAM refinement on the output feature map."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attn = CBAM(c2, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C2f and refine its output with CBAM attention."""
        return self.attn(super().forward(x))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Apply split-based C2f and refine its output with CBAM attention."""
        return self.attn(super().forward_split(x))


class RepC2f(C2f):
    """C2f module that replaces internal bottlenecks with RepBottleneck blocks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize RepC2f with the same interface and output shape as C2f."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))


class EnSimAMEdgeRepC2f(RepC2f):
    """RepC2f with EnSimAM applied only to the second split branch."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize branch-local EnSimAM refinement."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.edge = EnSimAM()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply edge refinement only on y[1], preserving the RepC2f topology."""
        y = list(self.cv1(x).chunk(2, 1))
        y[1] = self.edge(y[1])
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class PoolingEdgeRepC2f(RepC2f):
    """RepC2f with pooling edge enhancement applied only to the second split branch."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize branch-local pooling edge refinement."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.edge_conv = Conv(self.c, self.c, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply edge refinement only on y[1], preserving the RepC2f topology."""
        y = list(self.cv1(x).chunk(2, 1))
        edge = y[1] - self.pool(y[1])
        edge = self.edge_conv(edge)
        y[1] = y[1] + edge
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fKV(nn.Module):
    """C2f block with PSA-style split and gated KV-compressed attention on the final hidden feature."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        num_heads: int = 4,
        sr_ratio: int = 2,
        mode: str = "dwconv",
    ):
        """Initialize a C2f-style block with a lightweight PSA-style KV attention branch."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.kv_c = max(1, self.c // 2)
        self.bypass_c = self.c - self.kv_c
        self.kv = KVCompressedAttention(self.kv_c, self.kv_c, num_heads, sr_ratio, mode)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _refine_last(self, x: torch.Tensor) -> torch.Tensor:
        """Refine one split of the final hidden feature with gated KV attention."""
        if self.bypass_c == 0:
            return x + self.gamma * (self.kv(x) - x)
        a, b = x.split((self.bypass_c, self.kv_c), dim=1)
        b = b + self.gamma * (self.kv(b) - b)
        return torch.cat((a, b), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f with PSA-style KV refinement on the final hidden feature."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y[-1] = self._refine_last(y[-1])
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        y[-1] = self._refine_last(y[-1])
        return self.cv2(torch.cat(y, 1))


class C2fNAT(nn.Module):
    """C2f block with gated Neighborhood Attention on the final hidden feature."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        num_heads: int = 4,
        kernel_size: int = 3,
    ):
        """Initialize a C2f-style block with a lightweight NAT refinement branch."""
        super().__init__()
        try:
            from natten import NeighborhoodAttention2D
        except ImportError as exc:
            raise ImportError(
                "C2fNAT requires the 'natten' package. Install natten in the training environment before using "
                "YAMLs that reference C2fNAT."
            ) from exc

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.num_heads = _choose_attention_heads(self.c, num_heads)
        self.norm1 = nn.LayerNorm(self.c)
        self.attn = NeighborhoodAttention2D(embed_dim=self.c, num_heads=self.num_heads, kernel_size=int(kernel_size))
        self.norm2 = nn.LayerNorm(self.c)
        self.mlp = nn.Sequential(
            nn.Linear(self.c, 2 * self.c),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * self.c, self.c),
            nn.Dropout(0.1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def _refine_last(self, x: torch.Tensor) -> torch.Tensor:
        """Refine one hidden split with NHWC Neighborhood Attention."""
        if x.device.type == "cpu" and x.requires_grad:
            return x
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        att = self.attn(self.norm1(x_nhwc)) + x_nhwc
        refined = self.mlp(self.norm2(att)) + att
        refined = refined.permute(0, 3, 1, 2).contiguous()
        return x + self.gamma * (refined - x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f with NAT refinement on the final hidden feature."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y[-1] = self._refine_last(y[-1])
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        y[-1] = self._refine_last(y[-1])
        return self.cv2(torch.cat(y, 1))


class NATBlock(nn.Module):
    """Neighborhood Attention Transformer block for YOLO."""

    def __init__(self, c1: int, c2: int, num_heads: int = 4, kernel_size: int = 7):
        """Initialize a Neighborhood Attention block."""
        super().__init__()
        try:
            from natten import NeighborhoodAttention2D
        except ImportError as exc:
            raise ImportError(
                "NATBlock requires the 'natten' package. Install natten in the training environment before using "
                "YAMLs that reference NATBlock."
            ) from exc

        self.c = c1
        self.num_heads = _choose_attention_heads(c1, num_heads)
        self.norm1 = nn.LayerNorm(c1)
        self.attn = NeighborhoodAttention2D(embed_dim=c1, num_heads=self.num_heads, kernel_size=int(kernel_size))
        self.norm2 = nn.LayerNorm(c1)
        self.mlp = nn.Sequential(
            nn.Linear(c1, 2 * c1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * c1, c1),
            nn.Dropout(0.1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through NATBlock."""
        if x.device.type == "cpu" and x.requires_grad:
            return self.proj(x)
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        att = self.attn(self.norm1(x_nhwc)) + x_nhwc
        refined = self.mlp(self.norm2(att)) + att
        refined = refined.permute(0, 3, 1, 2).contiguous()
        out = x + self.gamma * (refined - x)
        return self.proj(out)


class M3NATFuse(nn.Module):
    """Fuse P2/P3/P4 features into a P3-resolution feature and refine it with NAT."""

    def __init__(self, c1: list[int], c2: int, num_heads: int = 4, kernel_size: int = 3):
        """Initialize three-scale fusion.

        Args:
            c1: Input channels for [P2, P3, P4].
            c2: Output channels at P3 resolution.
            num_heads: Requested NAT heads.
            kernel_size: NAT neighborhood kernel size.
        """
        super().__init__()
        if len(c1) != 3:
            raise ValueError(f"M3NATFuse expects 3 input channel values for [P2, P3, P4], got {c1}.")
        try:
            from natten import NeighborhoodAttention2D
        except ImportError as exc:
            raise ImportError(
                "M3NATFuse requires the 'natten' package. Install natten in the training environment before using "
                "YAMLs that reference M3NATFuse."
            ) from exc

        self.p2_down = Conv(c1[0], c2, 3, 2)
        self.p3_proj = Conv(c1[1], c2, 3, 1)
        self.p4_proj = Conv(c1[2], c2, 3, 1)
        self.fuse = Conv(3 * c2, c2, 3, 1)
        self.num_heads = _choose_attention_heads(c2, num_heads)
        self.norm1 = nn.LayerNorm(c2)
        self.attn = NeighborhoodAttention2D(embed_dim=c2, num_heads=self.num_heads, kernel_size=int(kernel_size))
        self.norm2 = nn.LayerNorm(c2)
        self.mlp = nn.Sequential(
            nn.Linear(c2, 2 * c2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * c2, c2),
            nn.Dropout(0.1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse [P2, P3, P4] into a P3-resolution feature map."""
        if len(x) != 3:
            raise ValueError(f"M3NATFuse expects [P2, P3, P4], got {len(x)} inputs.")
        p2, p3, p4 = x
        target_size = p3.shape[-2:]
        p2 = self.p2_down(p2)
        if p2.shape[-2:] != target_size:
            p2 = F.interpolate(p2, size=target_size, mode="nearest")
        p3 = self.p3_proj(p3)
        p4 = F.interpolate(p4, size=target_size, mode="nearest")
        p4 = self.p4_proj(p4)
        fused = self.fuse(torch.cat((p2, p3, p4), dim=1))
        if fused.device.type == "cpu" and fused.requires_grad:
            return fused

        fused_nhwc = fused.permute(0, 2, 3, 1).contiguous()
        att = self.attn(self.norm1(fused_nhwc)) + fused_nhwc
        refined = self.mlp(self.norm2(att)) + att
        refined = refined.permute(0, 3, 1, 2).contiguous()
        return fused + self.gamma * (refined - fused)


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3CBAM(C3):
    """C3 block followed by CBAM channel and spatial attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        kernel_size: int = 7,
    ):
        """Initialize C3 with CBAM refinement on the output feature map."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attn = CBAM(c2, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C3 and refine its output with CBAM attention."""
        return self.attn(super().forward(x))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize RepC3 module with RepConv blocks.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and addition to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 4 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes image features through unchanged after fusing."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with three convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the fused RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        if not hasattr(self, "conv1"):
            return  # already fused
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
        m (nn.ModuleList): List of PSABlock modules for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided into.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, all_head_dim, 7, 1, 3, g=all_head_dim, act=False)

    def __setstate__(self, state):
        """Add missing all_head_dim attribute to old checkpoints."""
        super().__setstate__(state)
        if not hasattr(self, "all_head_dim"):
            self.all_head_dim = self.head_dim * self.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, _, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
            v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided into.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semantic: bool = True) -> torch.Tensor:
        """Perform a forward pass by fusing multi-scale feature maps and generating proto masks."""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semantic:
            semantic = self.semseg(feat)
            return (p, semantic)
        return p

    def fuse(self):
        """Fuse the model for inference by removing the semantic segmentation head."""
        self.semseg = None


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian
        matrix.
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det
