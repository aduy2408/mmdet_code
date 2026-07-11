# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import CITYSCAPES_WEIGHT, OKS_SIGMA, RLE_WEIGHT, box_iou
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


@dataclass(frozen=True)
class BoundaryContrastiveLossConfig:
    """YOLO.train kwargs for the boundary-aware contrastive localization loss."""

    gain: float = 0.05
    levels: int = 2
    ring: float = 1.0
    samples: int = 16
    tau: float = 0.2
    shrinkage: float = 0.25

    def as_train_kwargs(self) -> dict[str, float | int]:
        """Return kwargs accepted by YOLO.train(...)."""

        return {
            "boundary_contrast": float(self.gain),
            "boundary_levels": int(self.levels),
            "boundary_ring": float(self.ring),
            "boundary_samples": int(self.samples),
            "boundary_tau": float(self.tau),
            "boundary_shrinkage": float(self.shrinkage),
        }


def boundary_contrastive_loss_kwargs(
    gain: float = 0.05,
    levels: int = 2,
    ring: float = 1.0,
    samples: int = 16,
    tau: float = 0.2,
    shrinkage: float = 0.25,
) -> dict[str, float | int]:
    """Build YOLO.train kwargs that enable the boundary contrastive loss."""

    return BoundaryContrastiveLossConfig(
        gain=gain,
        levels=levels,
        ring=ring,
        samples=samples,
        tau=tau,
        shrinkage=shrinkage,
    ).as_train_kwargs()


def add_boundary_contrastive_loss(
    train_kwargs: dict | None = None,
    *,
    gain: float = 0.05,
    levels: int = 2,
    ring: float = 1.0,
    samples: int = 16,
    tau: float = 0.2,
    shrinkage: float = 0.25,
) -> dict:
    """Return train kwargs with boundary contrastive localization loss enabled."""

    kwargs = dict(train_kwargs or {})
    kwargs.update(
        boundary_contrastive_loss_kwargs(
            gain=gain,
            levels=levels,
            ring=ring,
            samples=samples,
            tau=tau,
            shrinkage=shrinkage,
        )
    )
    return kwargs


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, pos_q_weight: bool = True):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_q_weight = pos_q_weight

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        pos_weight = gt_score if self.pos_q_weight else label
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + pos_weight * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class WiseIouLoss(nn.Module):
    """Wise-IoU family losses for xyxy boxes with optional dynamic focusing."""

    momentum = 1e-2
    alpha = 1.7
    delta = 2.7

    def __init__(self, ltype: str = "WIoU", monotonous: bool | None = False, eps: float = 1e-7):
        """Initialize Wise-IoU loss.

        Args:
            ltype: IoU loss type, e.g. ``WIoU``, ``IoU``, ``GIoU``, ``DIoU``, ``CIoU``, ``EIoU`` or ``SIoU``.
            monotonous: ``True`` for monotonic focusing, ``False`` for non-monotonic focusing, ``None`` to disable it.
            eps: Small value for numerical stability.
        """
        super().__init__()
        self.ltype = ltype.upper()
        if self.ltype == "WIOU":
            self.ltype = "WIoU"
        elif self.ltype.endswith("IOU"):
            self.ltype = self.ltype[:-3] + "IoU"
        if getattr(self, f"_{self.ltype}", None) is None:
            raise ValueError(f"Unsupported bbox_iou_loss='{ltype}'.")
        self.monotonous = monotonous
        self.eps = eps
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def __getitem__(self, item: str) -> torch.Tensor:
        """Evaluate cached geometry terms lazily."""
        if callable(self._fget[item]):
            self._fget[item] = self._fget[item]()
        return self._fget[item]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-box IoU-family loss and plain IoU for xyxy boxes."""
        self._fget = {
            "pred": pred,
            "target": target,
            "pred_xy": lambda: (self["pred"][..., :2] + self["pred"][..., 2:4]) / 2,
            "pred_wh": lambda: (self["pred"][..., 2:4] - self["pred"][..., :2]).clamp(min=self.eps),
            "target_xy": lambda: (self["target"][..., :2] + self["target"][..., 2:4]) / 2,
            "target_wh": lambda: (self["target"][..., 2:4] - self["target"][..., :2]).clamp(min=self.eps),
            "min_coord": lambda: torch.minimum(self["pred"][..., :4], self["target"][..., :4]),
            "max_coord": lambda: torch.maximum(self["pred"][..., :4], self["target"][..., :4]),
            "wh_inter": lambda: torch.relu(self["min_coord"][..., 2:4] - self["max_coord"][..., :2]),
            "s_inter": lambda: torch.prod(self["wh_inter"], dim=-1),
            "s_union": lambda: torch.prod(self["pred_wh"], dim=-1)
            + torch.prod(self["target_wh"], dim=-1)
            - self["s_inter"]
            + self.eps,
            "wh_box": lambda: (self["max_coord"][..., 2:4] - self["min_coord"][..., :2]).clamp(min=self.eps),
            "s_box": lambda: torch.prod(self["wh_box"], dim=-1) + self.eps,
            "l2_box": lambda: torch.square(self["wh_box"]).sum(dim=-1) + self.eps,
            "d_center": lambda: self["pred_xy"] - self["target_xy"],
            "l2_center": lambda: torch.square(self["d_center"]).sum(dim=-1),
            "iou": lambda: self["s_inter"] / self["s_union"],
            "iou_loss": lambda: 1.0 - self["iou"],
        }

        if self.training:
            self.iou_mean.mul_(1 - self.momentum)
            self.iou_mean.add_(self.momentum * self["iou_loss"].detach().mean())

        loss = self._scaled_loss(getattr(self, f"_{self.ltype}")())
        iou = self["iou"]
        delattr(self, "_fget")
        return loss, iou

    def _scaled_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply Wise-IoU dynamic focusing when configured."""
        if isinstance(self.monotonous, bool):
            beta = self["iou_loss"].detach() / self.iou_mean.clamp(min=self.eps)
            if self.monotonous:
                loss = loss * beta.sqrt()
            else:
                divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
                loss = loss * beta / divisor
        return loss

    def _IoU(self) -> torch.Tensor:
        return self["iou_loss"]

    def _WIoU(self) -> torch.Tensor:
        dist = torch.exp(self["l2_center"] / self["l2_box"].detach())
        return dist * self["iou_loss"]

    def _EIoU(self) -> torch.Tensor:
        penalty = self["l2_center"] / self["l2_box"] + torch.square(self["d_center"] / self["wh_box"]).sum(dim=-1)
        return self["iou_loss"] + penalty

    def _GIoU(self) -> torch.Tensor:
        return self["iou_loss"] + (self["s_box"] - self["s_union"]) / self["s_box"]

    def _DIoU(self) -> torch.Tensor:
        return self["iou_loss"] + self["l2_center"] / self["l2_box"]

    def _CIoU(self) -> torch.Tensor:
        v = (
            4
            / math.pi**2
            * (
                torch.atan(self["pred_wh"][..., 0] / self["pred_wh"][..., 1])
                - torch.atan(self["target_wh"][..., 0] / self["target_wh"][..., 1])
            ).pow(2)
        )
        alpha = v / (self["iou_loss"] + v + self.eps)
        return self["iou_loss"] + self["l2_center"] / self["l2_box"] + alpha.detach() * v

    def _SIoU(self, theta: int = 4) -> torch.Tensor:
        angle = torch.arcsin(torch.abs(self["d_center"]).min(dim=-1)[0] / (self["l2_center"].sqrt() + self.eps))
        angle = torch.sin(2 * angle) - 2
        dist = angle[..., None] * torch.square(self["d_center"] / self["wh_box"])
        dist = 2 - torch.exp(dist[..., 0]) - torch.exp(dist[..., 1])
        d_shape = torch.abs(self["pred_wh"] - self["target_wh"])
        big_shape = torch.maximum(self["pred_wh"], self["target_wh"])
        w_shape = 1 - torch.exp(-d_shape[..., 0] / big_shape[..., 0])
        h_shape = 1 - torch.exp(-d_shape[..., 1] / big_shape[..., 1])
        shape = w_shape**theta + h_shape**theta
        return self["iou_loss"] + (dist + shape) / 2


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16, iou_loss: str = "ciou", wiou_monotonous: bool | None = False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.iou_loss_type = iou_loss.lower()
        if self.iou_loss_type not in {"ciou", "wiou", "iou", "giou", "diou", "eiou", "siou"}:
            raise ValueError(
                f"Unsupported bbox_iou_loss='{iou_loss}'. Expected one of: ciou, wiou, iou, giou, diou, eiou, siou."
            )
        self.wise_iou = (
            WiseIouLoss(ltype=iou_loss, monotonous=wiou_monotonous)
            if self.iou_loss_type != "ciou"
            else None
        )

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
        quality_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if quality_weights is not None:
            weight = weight * quality_weights.to(device=weight.device, dtype=weight.dtype).view(-1, 1)
        if self.wise_iou is None:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        else:
            iou_loss, _ = self.wise_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_iou = (iou_loss.unsqueeze(-1) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


# EXPERIMENTAL: Boundary-aware contrastive loss for tiny-object localization.
# This auxiliary objective samples Detect feature maps around each GT box so
# object-interior features stay close while boundary/background features are
# pushed away. It is opt-in through boundary_contrast and has no inference path.
class BoundaryContrastiveLoss(nn.Module):
    """Boundary-aware InfoNCE loss over Detect feature maps."""

    def __init__(self, levels: int = 2, ring: float = 1.0, samples: int = 16, tau: float = 0.2, shrinkage: float = 0.25):
        """Initialize boundary contrast settings."""
        super().__init__()
        self.levels = max(int(levels), 0)
        self.ring = max(float(ring), 0.0)
        self.samples = max(int(samples), 1)
        self.tau = max(float(tau), 1e-6)
        self.shrinkage = max(float(shrinkage), 0.0)

    @staticmethod
    def _sample_indices(mask: torch.Tensor, limit: int) -> torch.Tensor:
        """Sample up to limit flattened indices from a boolean mask."""
        idx = mask.flatten().nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > limit:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:limit]]
        return idx

    def forward(
        self,
        feats: list[torch.Tensor],
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        strides: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary-aware contrastive loss from feature maps and image-space GT boxes."""
        losses = []
        num_levels = min(self.levels, len(feats), len(strides))
        for level, feat in enumerate(feats[:num_levels]):
            bs, _, h, w = feat.shape
            stride = strides[level].to(device=feat.device, dtype=feat.dtype).clamp(min=1)
            y, x = torch.meshgrid(
                torch.arange(h, device=feat.device, dtype=feat.dtype) + 0.5,
                torch.arange(w, device=feat.device, dtype=feat.dtype) + 0.5,
                indexing="ij",
            )

            for bi in range(bs):
                boxes = gt_bboxes[bi][mask_gt[bi, :, 0].bool()]
                if boxes.numel() == 0:
                    continue

                fmap = feat[bi].flatten(1).transpose(0, 1)
                fmap = F.normalize(fmap, dim=1)

                for box in boxes:
                    x1, y1, x2, y2 = box / stride
                    x1 = x1.clamp(0, w)
                    x2 = x2.clamp(0, w)
                    y1 = y1.clamp(0, h)
                    y2 = y2.clamp(0, h)
                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue

                    # EXPERIMENTAL: prefer a tighter interior region when it
                    # exists, so positives represent object texture, not edges.
                    pad_x = torch.minimum((x2 - x1) * self.shrinkage, torch.tensor(0.5, device=feat.device, dtype=feat.dtype))
                    pad_y = torch.minimum((y2 - y1) * self.shrinkage, torch.tensor(0.5, device=feat.device, dtype=feat.dtype))
                    inner = (x >= x1 + pad_x) & (x < x2 - pad_x) & (y >= y1 + pad_y) & (y < y2 - pad_y)
                    obj = inner if inner.any() else ((x >= x1) & (x < x2) & (y >= y1) & (y < y2))

                    dx1 = (x1 - self.ring).clamp(0, w)
                    dy1 = (y1 - self.ring).clamp(0, h)
                    dx2 = (x2 + self.ring).clamp(0, w)
                    dy2 = (y2 + self.ring).clamp(0, h)
                    dilated = (x >= dx1) & (x < dx2) & (y >= dy1) & (y < dy2)
                    original = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
                    boundary = dilated & ~original

                    # EXPERIMENTAL: nearby background is a cheap hard-negative
                    # proxy without using prediction confidence.
                    near_x1 = (x1 - self.ring * 3).clamp(0, w)
                    near_y1 = (y1 - self.ring * 3).clamp(0, h)
                    near_x2 = (x2 + self.ring * 3).clamp(0, w)
                    near_y2 = (y2 + self.ring * 3).clamp(0, h)
                    near = (x >= near_x1) & (x < near_x2) & (y >= near_y1) & (y < near_y2)
                    background = near & ~dilated
                    if not background.any():
                        background = ~dilated

                    obj_idx = self._sample_indices(obj, self.samples)
                    bnd_idx = self._sample_indices(boundary, self.samples)
                    bg_idx = self._sample_indices(background, self.samples)
                    if obj_idx.numel() < 2 or bnd_idx.numel() == 0 or bg_idx.numel() == 0:
                        continue

                    obj_feat = fmap[obj_idx]
                    pos = obj_feat.roll(1, dims=0)
                    neg = fmap[torch.cat((bnd_idx, bg_idx), 0)]

                    pos_logits = (obj_feat * pos).sum(1, keepdim=True) / self.tau
                    neg_logits = obj_feat @ neg.T / self.tau
                    logits = torch.cat((pos_logits, neg_logits), 1)
                    labels = torch.zeros(logits.shape[0], device=feat.device, dtype=torch.long)
                    losses.append(F.cross_entropy(logits, labels))

        return torch.stack(losses).mean() if losses else gt_bboxes.sum() * 0.0


# EXPERIMENTAL: Localization Quality Map loss for tiny-object localization.
# This auxiliary objective teaches train-only 1x1 heads to predict a smooth
# center-high localization target for each GT box. It has no inference path.
class LocalizationQualityLoss(nn.Module):
    """Gaussian localization quality map supervision over Detect feature maps."""

    def __init__(self, levels: int = 2, sigma: float = 0.45, loss_type: str = "mse"):
        """Initialize LQM settings."""
        super().__init__()
        self.levels = max(int(levels), 0)
        self.sigma = max(float(sigma), 1e-3)
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"mse", "smoothl1"}:
            raise ValueError("loc_quality_loss must be 'mse' or 'smoothl1'.")

    def forward(
        self,
        loc_maps: list[torch.Tensor],
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        strides: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LQM loss from predicted maps and image-space GT boxes."""
        losses = []
        num_levels = min(self.levels, len(loc_maps), len(strides))
        for level, loc_map in enumerate(loc_maps[:num_levels]):
            bs, _, h, w = loc_map.shape
            stride = strides[level].to(device=loc_map.device, dtype=loc_map.dtype).clamp(min=1)
            y, x = torch.meshgrid(
                torch.arange(h, device=loc_map.device, dtype=loc_map.dtype) + 0.5,
                torch.arange(w, device=loc_map.device, dtype=loc_map.dtype) + 0.5,
                indexing="ij",
            )
            target = torch.zeros((bs, 1, h, w), device=loc_map.device, dtype=loc_map.dtype)

            for bi in range(bs):
                boxes = gt_bboxes[bi][mask_gt[bi, :, 0].bool()]
                if boxes.numel() == 0:
                    continue
                image_target = target[bi, 0]
                for box in boxes:
                    x1, y1, x2, y2 = box / stride
                    x1 = x1.clamp(0, w)
                    x2 = x2.clamp(0, w)
                    y1 = y1.clamp(0, h)
                    y2 = y2.clamp(0, h)
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < 1 or bh < 1:
                        continue

                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    sigma_x = (bw * self.sigma).clamp(min=1e-3)
                    sigma_y = (bh * self.sigma).clamp(min=1e-3)
                    inside = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
                    quality = torch.exp(
                        -0.5 * (((x - cx) / sigma_x).pow(2) + ((y - cy) / sigma_y).pow(2))
                    ) * inside.to(loc_map.dtype)
                    image_target.copy_(torch.maximum(image_target, quality))

            pred = loc_map.sigmoid()
            if self.loss_type == "smoothl1":
                losses.append(F.smooth_l1_loss(pred, target))
            else:
                losses.append(F.mse_loss(pred, target))

        return torch.stack(losses).mean() if losses else gt_bboxes.sum() * 0.0


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Attributes:
        size_average (bool): Option to average the loss by the batch_size.
        use_target_weight (bool): Option to use weighted loss.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.vfl = (
            VarifocalLoss(
                gamma=float(getattr(h, "vfl_gamma", 2.0)),
                alpha=float(getattr(h, "vfl_alpha", 0.75)),
                pos_q_weight=bool(getattr(h, "vfl_pos_q_weight", True)),
            )
            if bool(getattr(h, "vfl", False))
            else None
        )
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        # Class weights for handling imbalanced datasets
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        self.assigner = TaskAlignedAssigner(
            topk=int(getattr(h, "tal_topk", tal_topk)),
            num_classes=self.nc,
            alpha=float(getattr(h, "tal_alpha", 0.5)),
            beta=float(getattr(h, "tal_beta", 6.0)),
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(
            m.reg_max,
            iou_loss=getattr(h, "bbox_iou_loss", "ciou"),
            wiou_monotonous=getattr(h, "wiou_monotonous", False),
        ).to(device)
        # EXPERIMENTAL: optional boundary-aware contrastive objective for
        # Varroa-scale localization. Disabled by default via boundary_contrast=0.
        self.boundary_gain = float(getattr(h, "boundary_contrast", 0.0))
        self.boundary_loss = (
            BoundaryContrastiveLoss(
                levels=getattr(h, "boundary_levels", 2),
                ring=getattr(h, "boundary_ring", 1.0),
                samples=getattr(h, "boundary_samples", 16),
                tau=getattr(h, "boundary_tau", 0.2),
                shrinkage=getattr(h, "boundary_shrinkage", 0.25),
            ).to(device)
            if self.boundary_gain > 0
            else None
        )
        # EXPERIMENTAL: optional train-only Localization Quality Map supervision.
        # The Detect module owns the 1x1 heads so optimizer updates them.
        self.loc_quality_gain = float(getattr(h, "loc_quality", 0.0))
        m.loc_quality_enabled = self.loc_quality_gain > 0
        self.loc_quality_heads = getattr(m, "loc_cv", None)
        self.loc_quality_loss = (
            LocalizationQualityLoss(
                levels=getattr(h, "loc_quality_levels", 2),
                sigma=getattr(h, "loc_quality_sigma", 0.45),
                loss_type=getattr(h, "loc_quality_loss", "mse"),
            ).to(device)
            if self.loc_quality_gain > 0
            else None
        )
        self.quality_head = bool(getattr(m, "quality_head", getattr(h, "quality_head", False)))
        self.quality_loss_type = str(getattr(h, "quality_loss", "bce")).lower()
        if self.quality_loss_type not in {"bce", "bce_balanced", "l1"}:
            raise ValueError("quality_loss must be 'bce', 'bce_balanced', or 'l1'.")
        self.quality_gain = float(getattr(h, "quality_gain", 0.5))
        self.quality_neg_gain = float(getattr(h, "quality_neg_gain", 0.10))
        self.quality_pos_iou_thr = float(getattr(h, "quality_pos_iou_thr", 0.5))
        self.quality_hard_neg_iou_thr = float(getattr(h, "quality_hard_neg_iou_thr", 0.3))
        self.quality_hard_neg_score_thr = float(getattr(h, "quality_hard_neg_score_thr", 0.05))
        self.quality_target_mode = str(getattr(h, "quality_target_mode", "iou_power")).lower()
        if self.quality_target_mode not in {"iou_power", "ap75_ramp"}:
            raise ValueError("quality_target_mode must be 'iou_power' or 'ap75_ramp'.")
        self.quality_target_power = float(getattr(h, "quality_target_power", 1.0))
        self.quality_ramp_low = float(getattr(h, "quality_ramp_low", 0.50))
        self.quality_ramp_high = float(getattr(h, "quality_ramp_high", 0.75))
        if self.quality_ramp_high <= self.quality_ramp_low:
            raise ValueError("quality_ramp_high must be greater than quality_ramp_low.")
        self.quality_neg_mode = str(getattr(h, "quality_neg_mode", "hard")).lower()
        if self.quality_neg_mode not in {"hard", "all"}:
            raise ValueError("quality_neg_mode must be 'hard' or 'all'.")
        self.quality_detach_target = bool(getattr(h, "quality_detach_target", True))
        self.quality_debug = bool(getattr(h, "quality_debug", False))
        self.quality_debug_batches = int(getattr(h, "quality_debug_batches", 1))
        self.quality_debug_seen = 0
        self.dgfe_rec_gain = float(getattr(h, "dgfe_rec_gain", 0.0))
        self.dgfe_spatial_gain = float(getattr(h, "dgfe_spatial_gain", 0.0))
        self.dgfe_boundary_ring = float(getattr(h, "dgfe_boundary_ring", 1.0))
        self.dgfe_inner_value = float(getattr(h, "dgfe_inner_value", 0.3))
        self.dgfe_tiny_area = float(getattr(h, "dgfe_tiny_area", 4.0))
        self.dgfe_neg_pos_ratio = int(getattr(h, "dgfe_neg_pos_ratio", 3))
        self.dgfe_neg_gain = float(getattr(h, "dgfe_neg_gain", 0.25))
        self.dgfe_spatial_target_mode = str(getattr(h, "dgfe_spatial_target_mode", "iou")).lower()
        if self.dgfe_spatial_target_mode not in {"iou", "edge_error"}:
            raise ValueError("dgfe_spatial_target_mode must be 'iou' or 'edge_error'.")
        self.dgfe_edge_error_norm = max(float(getattr(h, "dgfe_edge_error_norm", 0.25)), 1e-9)
        self.rank_gain = float(getattr(h, "rank_loss", 0.0))
        self.rank_tau = float(getattr(h, "rank_tau", 0.25))
        self.rank_iou_margin = float(getattr(h, "rank_iou_margin", 0.10))
        self.rank_topk = int(getattr(h, "rank_topk", 10))
        self.loc_assign = bool(getattr(h, "loc_assign", False))
        self.loc_assign_topk = int(getattr(h, "loc_assign_topk", 3))
        self.loc_assign_max_stride = float(getattr(h, "loc_assign_max_stride", 8.0))
        self.loc_assign_center_radius = float(getattr(h, "loc_assign_center_radius", 2.5))
        self.loc_assign_weight = float(getattr(h, "loc_assign_weight", 0.5))
        self.dfl_residual = bool(getattr(m, "dfl_residual", False))
        self.dfl_residual_scale = float(getattr(m, "dfl_residual_scale", getattr(h, "dfl_residual_scale", 0.25)))
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(
        self,
        anchor_points: torch.Tensor,
        pred_dist: torch.Tensor,
        pred_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
            if pred_residual is not None:
                pred_dist = (pred_dist + pred_residual.tanh() * self.dfl_residual_scale).clamp(
                    min=0, max=max(self.reg_max - 1, 0)
                )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def pairwise_ranking_loss(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Rank TAL foreground candidates so higher-IoU boxes receive higher assigned-class logits."""
        if not fg_mask.any() or self.rank_topk < 2:
            return pred_scores.sum() * 0.0

        fg_indices = fg_mask.nonzero(as_tuple=False)
        batch_indices, anchor_indices = fg_indices[:, 0], fg_indices[:, 1]
        rank_pred_bboxes = pred_bboxes.detach()[fg_mask]
        rank_target_bboxes = target_bboxes.detach()[fg_mask]
        rank_iou = bbox_iou(rank_pred_bboxes, rank_target_bboxes, xywh=False, CIoU=False).squeeze(-1).clamp(0)

        class_indices = target_scores[fg_mask].argmax(-1)
        assigned_logits = pred_scores[batch_indices, anchor_indices, class_indices]
        group_ids = batch_indices * (target_gt_idx.max() + 1).clamp_min(1) + target_gt_idx[fg_mask]

        rank_losses = []
        rank_weights = []
        tau = max(self.rank_tau, 1e-6)
        for group_id in group_ids.unique():
            group_mask = group_ids == group_id
            if group_mask.sum() < 2:
                continue

            group_iou = rank_iou[group_mask]
            group_logits = assigned_logits[group_mask]
            topk = min(self.rank_topk, int(group_iou.numel()))
            order = group_iou.argsort(descending=True)[:topk]
            group_iou = group_iou[order]
            group_logits = group_logits[order]

            iou_gap = group_iou[:, None] - group_iou[None, :]
            pair_mask = iou_gap > self.rank_iou_margin
            if not pair_mask.any():
                continue

            score_gap = group_logits[:, None] - group_logits[None, :]
            weights = iou_gap[pair_mask].detach()
            rank_losses.append(F.softplus(-score_gap[pair_mask] / tau) * weights)
            rank_weights.append(weights)

        if not rank_losses:
            return pred_scores.sum() * 0.0
        rank_losses = torch.cat(rank_losses)
        rank_weights = torch.cat(rank_weights)
        return rank_losses.sum() / (rank_weights.sum() + 1e-9)

    def build_localization_targets(
        self,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand box/DFL positives with GT-center local anchors without changing cls targets."""
        if not self.loc_assign or self.loc_assign_topk <= 0:
            return target_bboxes, target_scores, fg_mask

        loc_bboxes = target_bboxes.clone()
        loc_scores = target_scores.clone()
        loc_mask = fg_mask.clone()
        anchor_pixel = anchor_points * stride_tensor
        stride_flat = stride_tensor.squeeze(-1)
        stride_ok = stride_flat <= self.loc_assign_max_stride if self.loc_assign_max_stride > 0 else torch.ones_like(
            stride_flat, dtype=torch.bool
        )
        if not stride_ok.any():
            return loc_bboxes, loc_scores, loc_mask

        nc = loc_scores.shape[-1]
        for bi in range(target_bboxes.shape[0]):
            valid = mask_gt[bi].squeeze(-1).bool()
            if not valid.any():
                continue
            valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
            for gi in valid_indices:
                gt_box = gt_bboxes[bi, gi]
                gt_c = (gt_box[:2] + gt_box[2:]) * 0.5
                dist = (anchor_pixel - gt_c).pow(2).sum(-1).sqrt()
                radius = self.loc_assign_center_radius * stride_flat
                cand = stride_ok & (dist <= radius)
                if not cand.any():
                    continue
                cand_idx = cand.nonzero(as_tuple=False).squeeze(1)
                norm_dist = dist[cand_idx] / stride_flat[cand_idx].clamp_min(1e-6)
                topk = min(self.loc_assign_topk, int(norm_dist.numel()))
                if topk <= 0:
                    continue
                chosen = cand_idx[norm_dist.topk(topk, largest=False).indices]
                cls_idx = int(gt_labels[bi, gi, 0].clamp(0, nc - 1).item())
                new_only = ~loc_mask[bi, chosen]
                if not new_only.any():
                    continue
                chosen = chosen[new_only]
                loc_mask[bi, chosen] = True
                loc_bboxes[bi, chosen] = gt_box
                loc_scores[bi, chosen, cls_idx] = torch.maximum(
                    loc_scores[bi, chosen, cls_idx],
                    loc_scores.new_full((chosen.numel(),), self.loc_assign_weight),
                )
        return loc_bboxes, loc_scores, loc_mask

    @staticmethod
    def _slice_from_box(start: float, end: float, limit: int) -> tuple[int, int]:
        start_i = max(0, min(limit - 1, int(math.floor(start))))
        end_i = max(start_i + 1, min(limit, int(math.ceil(end))))
        return start_i, end_i

    @staticmethod
    def _dgfe_aux_list(preds: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
        aux = preds.get("dgfe_aux")
        if aux is None:
            return []
        return aux if isinstance(aux, list) else [aux]

    def _dgfe_reconstruction_loss(self, preds: dict[str, Any], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        aux_list = self._dgfe_aux_list(preds)
        if not aux_list or "img" not in batch:
            return preds["boxes"].sum() * 0.0

        losses = []
        for aux in aux_list:
            recon = aux.get("recon")
            if recon is None:
                continue
            target = batch["img"].to(device=recon.device, dtype=recon.dtype)
            if target.shape[-2:] != recon.shape[-2:]:
                target = F.interpolate(target, size=recon.shape[-2:], mode="bilinear", align_corners=False)
            losses.append(F.smooth_l1_loss(recon, target))
        return torch.stack(losses).mean() if losses else preds["boxes"].sum() * 0.0

    def _dgfe_spatial_target(
        self,
        logits: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz: torch.Tensor,
        q_by_gt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = torch.zeros_like(logits)
        batch_size, _, h, w = logits.shape
        image_h = max(float(imgsz[0]), 1.0)
        image_w = max(float(imgsz[1]), 1.0)
        ring = max(self.dgfe_boundary_ring, 0.0)
        inner_value = max(min(self.dgfe_inner_value, 1.0), 0.0)

        def write_max(region: torch.Tensor, value: float) -> None:
            if region.numel():
                region.copy_(torch.maximum(region, torch.full_like(region, value)))

        for bi in range(batch_size):
            valid_gt = mask_gt[bi, :, 0].bool()
            boxes = gt_bboxes[bi][valid_gt]
            quality = q_by_gt[bi][valid_gt].clamp(0, 1) if q_by_gt is not None else None
            for box_idx, box in enumerate(boxes):
                if quality is None:
                    q, left_value, top_value, right_value, bottom_value = 1.0, 1.0, 1.0, 1.0, 1.0
                else:
                    values = quality[box_idx]
                    if values.ndim == 0:
                        q = left_value = top_value = right_value = bottom_value = float(values)
                    else:
                        q, left_value, top_value, right_value, bottom_value = [float(v) for v in values[:5]]
                inside_value = inner_value * q
                boundary_value = max(left_value, top_value, right_value, bottom_value)
                x1, y1, x2, y2 = [float(v) for v in box]
                fx1, fx2 = x1 * w / image_w, x2 * w / image_w
                fy1, fy2 = y1 * h / image_h, y2 * h / image_h
                if fx2 <= fx1 or fy2 <= fy1:
                    continue

                ix1, ix2 = self._slice_from_box(fx1, fx2, w)
                iy1, iy2 = self._slice_from_box(fy1, fy2, h)
                if (ix2 - ix1) * (iy2 - iy1) <= self.dgfe_tiny_area:
                    write_max(target[bi, :, iy1:iy2, ix1:ix2], boundary_value)
                    continue

                write_max(target[bi, :, iy1:iy2, ix1:ix2], inside_value)
                edge = max(int(math.ceil(ring)), 1)
                write_max(target[bi, :, iy1:min(iy1 + edge, iy2), ix1:ix2], top_value)
                write_max(target[bi, :, max(iy2 - edge, iy1):iy2, ix1:ix2], bottom_value)
                write_max(target[bi, :, iy1:iy2, ix1:min(ix1 + edge, ix2)], left_value)
                write_max(target[bi, :, iy1:iy2, max(ix2 - edge, ix1):ix2], right_value)

                ox1, ox2 = self._slice_from_box(fx1 - ring, fx2 + ring, w)
                oy1, oy2 = self._slice_from_box(fy1 - ring, fy2 + ring, h)
                write_max(target[bi, :, oy1:iy1, ox1:ox2], top_value)
                write_max(target[bi, :, iy2:oy2, ox1:ox2], bottom_value)
                write_max(target[bi, :, iy1:iy2, ox1:ix1], left_value)
                write_max(target[bi, :, iy1:iy2, ix2:ox2], right_value)
        return target

    def _dgfe_quality_by_gt(
        self,
        assigned_iou: torch.Tensor | None,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> torch.Tensor | None:
        if assigned_iou is None or not fg_mask.any():
            return None
        quality = torch.zeros((*mask_gt.shape[:2], 5), device=mask_gt.device, dtype=assigned_iou.dtype)
        offset = 0
        for bi in range(fg_mask.shape[0]):
            fg = fg_mask[bi]
            n = int(fg.sum().item())
            if not n:
                continue
            gt_idx = target_gt_idx[bi, fg].long()
            gt_iou = assigned_iou.detach()[offset : offset + n]
            pred = pred_bboxes.detach()[bi, fg]
            target = target_bboxes.detach()[bi, fg]
            offset += n
            for gi in gt_idx.unique():
                group = gt_idx == gi
                best = gt_iou[group].argmax()
                q = gt_iou[group][best]
                side_values = q.repeat(4)
                if self.dgfe_spatial_target_mode == "edge_error":
                    pred_box = pred[group][best]
                    target_box = target[group][best]
                    w = (target_box[2] - target_box[0]).clamp_min(1e-9)
                    h = (target_box[3] - target_box[1]).clamp_min(1e-9)
                    side_error = torch.stack(
                        (
                            (pred_box[0] - target_box[0]).abs() / w,
                            (pred_box[1] - target_box[1]).abs() / h,
                            (pred_box[2] - target_box[2]).abs() / w,
                            (pred_box[3] - target_box[3]).abs() / h,
                        )
                    ).div(self.dgfe_edge_error_norm).clamp(0, 1)
                    side_values = torch.maximum(side_values, side_error)
                quality[bi, gi] = torch.cat((q.reshape(1), side_values))
        return quality

    def _dgfe_spatial_loss(
        self,
        preds: dict[str, Any],
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz: torch.Tensor,
        q_by_gt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        aux_list = self._dgfe_aux_list(preds)
        if not aux_list:
            return preds["boxes"].sum() * 0.0

        losses = []
        for aux in aux_list:
            logits = aux.get("spatial_logits")
            if logits is None:
                continue
            target = self._dgfe_spatial_target(logits, gt_bboxes, mask_gt, imgsz, q_by_gt).to(dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            pos_mask = target > 0
            if not pos_mask.any():
                losses.append(logits.sum() * 0.0)
                continue

            pos_loss = bce[pos_mask].mean()
            neg_loss = logits.sum() * 0.0
            neg = bce[~pos_mask]
            if neg.numel():
                k = min(int(pos_mask.sum().item()) * max(self.dgfe_neg_pos_ratio, 1), neg.numel())
                neg_loss = neg.topk(k).values.mean()
            losses.append(pos_loss + self.dgfe_neg_gain * neg_loss)
        return torch.stack(losses).mean() if losses else preds["boxes"].sum() * 0.0

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(
            3
            + int(self.boundary_loss is not None)
            + int(self.loc_quality_loss is not None)
            + int(self.quality_head)
            + int(self.rank_gain > 0)
            + int(self.dgfe_rec_gain > 0)
            + int(self.dgfe_spatial_gain > 0),
            device=self.device,
        )  # box, cls, dfl[, boundary][, loc_quality][, quality][, rank][, dgfe_rec][, dgfe_spatial]
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        pred_residual = preds.get("dfl_residual")
        pred_residual = pred_residual.permute(0, 2, 1).contiguous() if pred_residual is not None else None
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_residual)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        fg_mask = fg_mask.bool()

        target_scores_sum = max(target_scores.sum(), 1)
        cls_target_scores = target_scores
        cls_target_scores_sum = target_scores_sum
        assigned_iou = None
        # Use the same coordinate scale as bbox loss. If target_bboxes has already been divided by stride_tensor,
        # do not divide again.
        target_bboxes_scaled = target_bboxes / stride_tensor
        loc_target_bboxes, loc_target_scores, loc_fg_mask = self.build_localization_targets(
            anchor_points,
            stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            target_bboxes,
            target_scores,
            fg_mask,
        )
        loc_target_bboxes_scaled = loc_target_bboxes / stride_tensor
        loc_target_scores_sum = max(loc_target_scores.sum(), 1)

        # Cls loss with optional class weighting
        if (self.vfl is not None or getattr(self.hyp, "cls_iou_target", False)) and fg_mask.sum():
            iou_pred_bboxes = pred_bboxes[fg_mask]
            if self.vfl is None or bool(getattr(self.hyp, "vfl_iou_detach", True)):
                iou_pred_bboxes = iou_pred_bboxes.detach()
            assigned_iou = bbox_iou(
                iou_pred_bboxes,
                target_bboxes_scaled[fg_mask],
                xywh=False,
                CIoU=False,
            ).squeeze(-1).clamp(0)
            if self.vfl is not None and bool(getattr(self.hyp, "vfl_iou_detach", True)):
                assigned_iou = assigned_iou.detach()
        if self.vfl is not None:
            if assigned_iou is not None:
                cls_target_scores = target_scores.clone()
                cls_target_scores[fg_mask] = torch.where(
                    target_scores[fg_mask] > 0,
                    assigned_iou.unsqueeze(-1).to(dtype=target_scores.dtype),
                    target_scores[fg_mask],
                )
            cls_target_scores_sum = max(cls_target_scores.sum(), 1)
            cls_labels = (cls_target_scores > 0).to(dtype)
            loss[1] = self.vfl(pred_scores, cls_target_scores.to(dtype), cls_labels) / cls_target_scores_sum
        elif getattr(self.hyp, "cls_iou_target", False) and assigned_iou is not None:
            with torch.no_grad():
                cls_target_scores = target_scores.clone()
                cls_target_scores[fg_mask] = torch.where(
                    target_scores[fg_mask] > 0,
                    assigned_iou.unsqueeze(-1),
                    target_scores[fg_mask],
                )
                cls_target_scores_sum = max(cls_target_scores.sum(), 1)
            bce_loss = self.bce(pred_scores, cls_target_scores.to(dtype))  # (bs, num_anchors, nc)
            if self.class_weights is not None:
                bce_loss *= self.class_weights
            loss[1] = bce_loss.sum() / cls_target_scores_sum  # BCE
        else:
            bce_loss = self.bce(pred_scores, cls_target_scores.to(dtype))  # (bs, num_anchors, nc)
            if self.class_weights is not None:
                bce_loss *= self.class_weights
            loss[1] = bce_loss.sum() / cls_target_scores_sum  # BCE

        # Bbox loss. Optionally use a localization-only positive set that does not affect cls targets.
        if loc_fg_mask.sum():
            quality_weights = None
            if bool(getattr(self.hyp, "vfl_weight_box_by_q", False)):
                with torch.no_grad():
                    box_assigned_iou = bbox_iou(
                        pred_bboxes.detach()[loc_fg_mask],
                        loc_target_bboxes_scaled[loc_fg_mask],
                        xywh=False,
                        CIoU=False,
                    ).squeeze(-1).clamp(0)
                quality_weights = box_assigned_iou.detach().clamp_min(float(getattr(self.hyp, "box_q_weight_min", 0.25)))
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                loc_target_bboxes_scaled,
                loc_target_scores,
                loc_target_scores_sum,
                loc_fg_mask,
                imgsz,
                stride_tensor,
                quality_weights,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        if self.boundary_loss is not None:
            # EXPERIMENTAL: add feature-space boundary separation only during training loss computation.
            loss[3] = self.boundary_loss(preds["feats"], gt_bboxes, mask_gt, self.stride) * self.boundary_gain
        if self.loc_quality_loss is not None:
            # EXPERIMENTAL: train-only smooth localization map supervision.
            loc_idx = 3 + int(self.boundary_loss is not None)
            loc_maps = preds.get("loc_maps")
            if loc_maps is None:
                loc_maps = [self.loc_quality_heads[i](x) for i, x in enumerate(preds["feats"])]
            loss[loc_idx] = self.loc_quality_loss(loc_maps, gt_bboxes, mask_gt, self.stride) * self.loc_quality_gain
        if self.quality_head:
            quality_idx = 3 + int(self.boundary_loss is not None) + int(self.loc_quality_loss is not None)
            quality_logits = preds.get("quality_logits")
            if quality_logits is None:
                raise KeyError("quality_head=True but model predictions do not contain 'quality_logits'.")
            quality_logits = quality_logits.permute(0, 2, 1).contiguous().squeeze(-1)
            quality_target = torch.zeros_like(quality_logits)
            max_iou_to_gt = torch.zeros_like(quality_logits)
            pred_bboxes_for_quality = pred_bboxes.detach() if self.quality_detach_target else pred_bboxes
            for bi in range(batch_size):
                valid_gt = mask_gt[bi].squeeze(-1).bool()
                if valid_gt.any():
                    pred_pixel = pred_bboxes_for_quality[bi] * stride_tensor
                    max_iou_to_gt[bi] = box_iou(pred_pixel, gt_bboxes[bi, valid_gt]).clamp(0).amax(-1)
            if fg_mask.sum():
                quality_iou = bbox_iou(
                    pred_bboxes.detach()[fg_mask] if self.quality_detach_target else pred_bboxes[fg_mask],
                    target_bboxes_scaled[fg_mask],
                    xywh=False,
                    CIoU=False,
                ).squeeze(-1).clamp(0)
                if self.quality_detach_target:
                    quality_iou = quality_iou.detach()
                quality_target[fg_mask] = quality_iou.to(dtype=quality_target.dtype)
            if self.quality_loss_type == "bce_balanced":
                if self.quality_detach_target:
                    max_iou_to_gt = max_iou_to_gt.detach()
                q_pos_mask = fg_mask | (max_iou_to_gt > self.quality_pos_iou_thr)
                q_target_value = max_iou_to_gt.clamp(0, 1)
                if self.quality_target_mode == "iou_power":
                    q_target_value = q_target_value.pow(self.quality_target_power)
                else:
                    q_target_value = ((q_target_value - self.quality_ramp_low) / (
                        self.quality_ramp_high - self.quality_ramp_low
                    )).clamp(0, 1)
                if self.quality_detach_target:
                    q_target_value = q_target_value.detach()
                quality_target[q_pos_mask] = q_target_value[q_pos_mask].to(dtype=quality_target.dtype)
                pos_loss = (
                    F.binary_cross_entropy_with_logits(
                        quality_logits[q_pos_mask],
                        quality_target[q_pos_mask],
                        reduction="mean",
                    )
                    if q_pos_mask.any()
                    else quality_logits.sum() * 0.0
                )
                if self.quality_neg_mode == "hard":
                    cls_score = pred_scores.detach().sigmoid().amax(-1)
                    neg_mask = (
                        (~q_pos_mask)
                        & (max_iou_to_gt < self.quality_hard_neg_iou_thr)
                        & (cls_score > self.quality_hard_neg_score_thr)
                    )
                else:
                    neg_mask = ~q_pos_mask
                neg_loss = (
                    F.binary_cross_entropy_with_logits(
                        quality_logits[neg_mask],
                        torch.zeros_like(quality_logits[neg_mask]),
                        reduction="mean",
                    )
                    if neg_mask.any()
                    else quality_logits.sum() * 0.0
                )
                loss[quality_idx] = self.quality_gain * (pos_loss + self.quality_neg_gain * neg_loss)
                debug_pos_loss, debug_neg_loss = pos_loss, neg_loss
                debug_q_pos_mask, debug_neg_mask = q_pos_mask, neg_mask
            elif self.quality_loss_type == "l1":
                pos_loss = (
                    F.l1_loss(quality_logits.sigmoid()[fg_mask], quality_target[fg_mask], reduction="sum")
                    if fg_mask.sum()
                    else quality_logits.sum() * 0.0
                )
                neg_loss = self.bce(quality_logits, quality_target).masked_fill(fg_mask, 0).sum()
                loss[quality_idx] = (pos_loss + neg_loss) / fg_mask.sum().clamp_min(1)
                debug_pos_loss, debug_neg_loss = pos_loss, neg_loss
                debug_q_pos_mask, debug_neg_mask = fg_mask, ~fg_mask
            else:
                loss[quality_idx] = self.bce(quality_logits, quality_target).sum() / fg_mask.sum().clamp_min(1)
                debug_pos_loss = loss[quality_idx]
                debug_neg_loss = quality_logits.sum() * 0.0
                debug_q_pos_mask, debug_neg_mask = fg_mask, ~fg_mask
            if self.quality_debug and self.quality_debug_seen < self.quality_debug_batches:
                self.quality_debug_seen += 1
                num_tal_pos = int(fg_mask.sum().item())
                num_q_pos = int(debug_q_pos_mask.sum().item())
                num_neg = int(debug_neg_mask.sum().item())
                if debug_q_pos_mask.any():
                    q_iou_dbg = quality_target[debug_q_pos_mask].detach().float()
                    iou_stats = (
                        float(q_iou_dbg.min().item()),
                        float(q_iou_dbg.mean().item()),
                        float(q_iou_dbg.max().item()),
                    )
                    max_iou_dbg = max_iou_to_gt[debug_q_pos_mask].detach().float()
                    max_iou_stats = (
                        float(max_iou_dbg.min().item()),
                        float(max_iou_dbg.mean().item()),
                        float(max_iou_dbg.max().item()),
                    )
                    pred_sample = pred_bboxes.detach()[debug_q_pos_mask][:5].float().cpu().tolist()
                    target_sample = quality_target.detach()[debug_q_pos_mask][:5].float().cpu().tolist()
                else:
                    iou_stats = (float("nan"), float("nan"), float("nan"))
                    max_iou_stats = (float("nan"), float("nan"), float("nan"))
                    pred_sample, target_sample = [], []
                LOGGER.info(
                    "quality_debug batch=%s/%s num_tal_pos=%s num_q_pos=%s num_neg=%s "
                    "iou_min/mean/max=(%.6g, %.6g, %.6g) max_iou_min/mean/max=(%.6g, %.6g, %.6g) "
                    "pos_loss=%.6g neg_loss=%.6g quality_gain=%.6g quality_neg_gain=%.6g loss_q=%.6g "
                    "quality_neg_mode=%s pos_thr=%.6g hard_neg_iou_thr=%.6g hard_neg_score_thr=%.6g "
                    "target_mode=%s target_power=%.6g ramp_low=%.6g ramp_high=%.6g",
                    self.quality_debug_seen,
                    self.quality_debug_batches,
                    num_tal_pos,
                    num_q_pos,
                    num_neg,
                    *iou_stats,
                    *max_iou_stats,
                    float(debug_pos_loss.detach().item()),
                    float(debug_neg_loss.detach().item()),
                    self.quality_gain,
                    self.quality_neg_gain,
                    float(loss[quality_idx].detach().item()),
                    self.quality_neg_mode,
                    self.quality_pos_iou_thr,
                    self.quality_hard_neg_iou_thr,
                    self.quality_hard_neg_score_thr,
                    self.quality_target_mode,
                    self.quality_target_power,
                    self.quality_ramp_low,
                    self.quality_ramp_high,
                )
                LOGGER.info("quality_debug pred_bboxes_sample=%s", pred_sample)
                LOGGER.info("quality_debug quality_target_sample=%s", target_sample)
        if self.rank_gain > 0:
            rank_idx = (
                3 + int(self.boundary_loss is not None) + int(self.loc_quality_loss is not None) + int(self.quality_head)
            )
            loss[rank_idx] = (
                self.pairwise_ranking_loss(
                    pred_scores,
                    pred_bboxes,
                    target_bboxes_scaled,
                    target_scores,
                    target_gt_idx,
                    fg_mask,
                )
                * self.rank_gain
            )
        dgfe_idx = (
            3
            + int(self.boundary_loss is not None)
            + int(self.loc_quality_loss is not None)
            + int(self.quality_head)
            + int(self.rank_gain > 0)
        )
        if self.dgfe_rec_gain > 0:
            loss[dgfe_idx] = self._dgfe_reconstruction_loss(preds, batch) * self.dgfe_rec_gain
            dgfe_idx += 1
        if self.dgfe_spatial_gain > 0:
            if fg_mask.sum() and assigned_iou is None:
                with torch.no_grad():
                    assigned_iou = bbox_iou(
                        pred_bboxes.detach()[fg_mask],
                        target_bboxes_scaled[fg_mask],
                        xywh=False,
                        CIoU=False,
                    ).squeeze(-1).clamp(0)
            q_by_gt = self._dgfe_quality_by_gt(
                assigned_iou, pred_bboxes, target_bboxes_scaled, target_gt_idx, fg_mask, mask_gt
            )
            loss[dgfe_idx] = self._dgfe_spatial_loss(preds, gt_bboxes, mask_gt, imgsz, q_by_gt) * self.dgfe_spatial_gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl[, boundary][, loc_quality][, quality][, rank][, dgfe_rec][, dgfe_spatial])

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate detection loss using assigned targets."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semantic
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semantic = proto
        else:
            pred_semantic = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        if fg_mask.sum():
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semantic is not None:
                sem_masks = batch["sem_masks"].to(self.device)  # NxHxW
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW

                if self.overlap:
                    mask_zero = masks == 0  # NxHxW
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semantic, sem_masks)
                loss[4] *= self.hyp.box  # seg gain

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if pred_semantic is not None:
                loss[4] += (pred_semantic * 0).sum()

        loss[1] *= self.hyp.box  # seg gain
        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl, semantic)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # Vectorized fill: compute within-batch position for each keypoint using cumulative offsets
        batch_idx_long = batch_idx.long()
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=keypoints.device)
        offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(batch_idx), device=keypoints.device) - offsets[batch_idx_long]
        batched_keypoints[batch_idx_long, within_idx] = keypoints

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLO26 pose estimation with RLE loss support."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(
            6 if self.rle_loss else 5, device=self.device
        )  # box, kpt_location, kpt_visibility, cls, dfl[, rle]
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        return loss * batch_size, loss.detach()  # loss(box, kpt_location, kpt_visibility, cls, dfl[, rle])

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, num_keypoints, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        if not kpt_mask.any():
            return pred_kpt[..., :0].sum()

        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)
        if not error.numel():
            return pred_kpt[..., :0].sum()

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return pred_kpt[..., :0].sum()

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=float(getattr(self.hyp, "tal_alpha", 0.5)),
            beta=float(getattr(self.hyp, "tal_beta", 6.0)),
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            packed_targets = targets[:, 1:].clone()
            packed_targets[:, 1:5].mul_(scale_tensor)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = packed_targets
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xywhr, (b, h*w, 5)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).
            target_bboxes (torch.Tensor): Target bounding boxes with shape [N, 5] (x, y, w, h, theta).
            fg_mask (torch.Tensor): Foreground mask indicating valid predictions.
            weight (torch.Tensor): Loss weights for each prediction.
            target_scores_sum (torch.Tensor): Sum of target scores for normalization.
            lambda_val (int): Controls the sensitivity to aspect ratio.

        Returns:
            (torch.Tensor): The calculated angle loss.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model: torch.nn.Module):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model: torch.nn.Module, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # init gain
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # final gain
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model: torch.nn.Module, tal_topk=10):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]


class SemanticSegmentationLoss(nn.Module):
    """Loss function for semantic segmentation using cross-entropy and Dice terms.

    Attributes:
        nc (int): Number of semantic classes.
        ce (nn.CrossEntropyLoss): Cross-entropy loss with ignore_index=255.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize semantic segmentation loss.

        Args:
            model (torch.nn.Module): Model containing the SemanticSegment head.
        """
        super().__init__()
        m = model.model[-1]
        self.nc = m.nc
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        data_name = Path(str(getattr(model.args, "data", "") or "")).stem.lower()
        self.use_cityscapes_weight = data_name in {"cityscapes", "cityscapes8"} and self.nc == len(CITYSCAPES_WEIGHT)
        if self.nc == 1:
            self.ce = nn.BCEWithLogitsLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=255).to(device=self.device, dtype=self.dtype)
            if self.use_cityscapes_weight:
                # Non-persistent: weight is a deterministic constant, no need to serialize into ckpt state_dict.
                weight = torch.from_numpy(CITYSCAPES_WEIGHT).to(device=self.device, dtype=self.dtype)
                self.ce.register_buffer("weight", weight, persistent=False)

    def _resize_masks(self, masks, target_shape):
        """Resize masks to match prediction spatial dimensions."""
        if masks.shape[1:] != target_shape:
            return (
                F.interpolate(masks.float().unsqueeze(1), size=target_shape, mode="nearest").squeeze(1).to(torch.int32)
            )
        return masks

    def _ce_loss(self, preds, masks):
        """Compute cross-entropy on flattened pixels to avoid the CUDA nll_loss2d path."""
        if self.nc == 1:
            flat = masks.reshape(-1)
            valid = flat != 255
            logits = preds.reshape(-1)[valid]
            target = flat[valid].float()
        else:
            logits = preds.permute(0, 2, 3, 1).reshape(-1, self.nc)
            target = masks.reshape(-1).long()
        return self.ce(logits, target)

    def _dice_loss(self, preds, masks):
        """Compute Dice loss excluding ignore pixels."""
        if self.nc == 1:
            return self._binary_dice_loss(preds, masks)
        flat_target = masks.reshape(-1)
        valid = flat_target != 255
        if not valid.any():
            return preds.sum() * 0

        pred_soft = F.softmax(preds, dim=1)
        target = flat_target[valid].long()
        flat_pred = pred_soft.float().permute(0, 2, 3, 1).reshape(-1, self.nc)[valid]
        intersection = torch.zeros(self.nc, device=preds.device, dtype=torch.float32)
        intersection.scatter_add_(0, target, flat_pred.gather(1, target[:, None]).squeeze(1))
        pred_sum = flat_pred.sum(dim=0)
        target_sum = torch.bincount(target, minlength=self.nc).to(device=preds.device, dtype=torch.float32)
        cardinality = pred_sum + target_sum
        return (1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

    def _binary_dice_loss(self, preds, masks):
        """Compute Dice loss for single-class (binary) segmentation.

        Pixels with value 255 are excluded from Dice terms to match BCE valid-pixel filtering.
        """
        valid = (masks != 255).float()
        pred_soft = preds.squeeze(1).sigmoid()
        target = (masks == 1).float()
        intersection = (pred_soft * target * valid).sum()
        cardinality = ((pred_soft + target) * valid).sum()
        return 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

    def forward(self, preds, batch):
        """Compute semantic segmentation loss with optional auxiliary loss.

        Args:
            preds (torch.Tensor | tuple): Main logits [B, nc, H', W'], or (main, aux) tuple.
            batch (dict): Batch dict with 'semantic_mask' [B, H, W] containing class IDs (255=ignore).

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): (total_loss * batch_size, detached loss items [ce, dice, aux]).
        """
        # Unpack auxiliary logits when present.
        aux_logits = None
        if isinstance(preds, tuple):
            preds, aux_logits = preds

        masks = batch["semantic_mask"].to(preds.device)
        if preds.shape[2:] != masks.shape[1:]:
            preds = F.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)

        # Main cross-entropy and Dice loss.
        ce_loss = self._ce_loss(preds, masks)
        dice_loss = self._dice_loss(preds, masks)
        total = ce_loss + dice_loss

        # Auxiliary cross-entropy loss. Match ce_loss dtype so torch.stack below succeeds under AMP.
        aux_loss = torch.tensor(0.0, device=preds.device, dtype=ce_loss.dtype)
        if aux_logits is not None:
            if aux_logits.shape[2:] != masks.shape[1:]:
                aux_logits = F.interpolate(aux_logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            aux_loss = self._ce_loss(aux_logits, masks) * 0.4
            total += aux_loss

        loss_items = torch.stack([ce_loss, dice_loss, aux_loss]).detach()
        return total * preds.shape[0], loss_items
