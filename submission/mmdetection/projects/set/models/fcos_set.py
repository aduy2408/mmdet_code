from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate

from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.registry import MODELS


class DNResBlock(nn.Module):
    """SET scale-adaptive background smoothing block."""

    def __init__(self, in_channels: int = 256, reduction: int = 4,
                 kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size,
                      stride=1, padding=padding, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size,
                      stride=1, padding=padding, bias=True))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv_block(x)


@MODELS.register_module()
class FCOS_set(SingleStageDetector):
    """FCOS with SET HBS and API training enhancements.

    This is a MMDetection 3.x port of SET's legacy FCOS_set detector. The
    enhancement is active only during loss computation, so inference keeps the
    ordinary FCOS prediction path and has zero SET denoiser cost.
    """

    def __init__(self, backbone, neck, bbox_head, set_cfg=None,
                 train_cfg=None, test_cfg=None, data_preprocessor=None,
                 init_cfg=None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        set_cfg = set_cfg or {}
        self.reg_factor_range = list(set_cfg.get(
            'reg_factor_range', [0, 0.01, 0.1, 0.5, 1, 2, 5]))
        self.reg_factors = list(set_cfg.get('reg_factors', [4, 4, 4, 4, 4]))
        self.scale = float(set_cfg.get('scale', 1.0))
        kernel_sizes = [
            (int(math.log2(stride)) // 2 * 2) + 1
            for stride in self.bbox_head.strides
        ]
        self.denoisers = nn.ModuleList([
            DNResBlock(
                in_channels=self.bbox_head.in_channels,
                reduction=4,
                kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        ])

    @staticmethod
    def _gradient(losses: dict, name: str, feature: Tensor) -> Tensor:
        gradient = torch.autograd.grad(
            losses[name], feature, retain_graph=True, allow_unused=True)[0]
        if gradient is None:
            gradient = torch.zeros_like(feature)
        return gradient / (torch.linalg.vector_norm(gradient) + 1e-12)

    def _foreground_mask(self, batch_inputs: Tensor,
                         batch_data_samples) -> Tensor:
        mask = batch_inputs.new_zeros(
            (batch_inputs.shape[0], 1, batch_inputs.shape[2],
             batch_inputs.shape[3]))
        for image_index, data_sample in enumerate(batch_data_samples):
            boxes = data_sample.gt_instances.bboxes
            for box in boxes.detach():
                x1, y1, x2, y2 = box.round().long().tolist()
                x1 = max(0, min(x1, mask.shape[3]))
                x2 = max(0, min(x2, mask.shape[3]))
                y1 = max(0, min(y1, mask.shape[2]))
                y2 = max(0, min(y2, mask.shape[2]))
                if x2 > x1 and y2 > y1:
                    mask[image_index, 0, y1:y2, x1:x2] = 1
        return mask

    def loss(self, batch_inputs: Tensor, batch_data_samples):
        features = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(features, batch_data_samples)
        foreground = self._foreground_mask(batch_inputs, batch_data_samples)
        enhanced_features = []
        for index, feature in enumerate(features):
            background = 1 - interpolate(
                foreground, size=feature.shape[-2:], mode='nearest')
            grad_cls = self._gradient(losses, 'loss_cls', feature)
            grad_bbox = self._gradient(losses, 'loss_bbox', feature)
            grad_center = self._gradient(losses, 'loss_centerness', feature)
            denoised_background = self.denoisers[index](feature * background)
            denoised = denoised_background * background + feature * (1 - background)
            factor_index = min(index, len(self.reg_factors) - 1)
            factor = self.reg_factor_range[self.reg_factors[factor_index]]
            noise = (grad_bbox + grad_cls + grad_center) / 3.0 * factor
            enhanced_features.append(denoised + noise)

        enhanced_losses = self.bbox_head.loss(
            enhanced_features, batch_data_samples)
        losses['loss_set_cls'] = (
            self.bbox_head.loss_cls.loss_weight * enhanced_losses['loss_cls'] *
            self.scale)
        losses['loss_set_bbox'] = (
            self.bbox_head.loss_bbox.loss_weight * enhanced_losses['loss_bbox'] *
            self.scale)
        losses['loss_set_centerness'] = (
            self.bbox_head.loss_centerness.loss_weight *
            enhanced_losses['loss_centerness'] * self.scale)
        return losses
