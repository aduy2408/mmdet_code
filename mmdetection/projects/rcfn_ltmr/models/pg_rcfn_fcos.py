import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.detectors import FCOS
from mmdet.registry import MODELS
from mmdet.structures import SampleList


def _box_tensor(boxes) -> Tensor:
    return boxes.tensor if hasattr(boxes, 'tensor') else boxes


@MODELS.register_module()
class PGRCFNFCOS(FCOS):
    """FCOS with a Gaussian-supervised position-guided RCFN neck."""

    def __init__(self, *args, tiny_max_sqrt_area: float = 16.0,
                 gaussian_alpha: float = 1.0,
                 gaussian_sigma_min: float = 1.0,
                 position_positive_weight: float = 4.0,
                 loss_pos_weight: float = 1.0,
                 position_stride: int = 8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tiny_max_sqrt_area = tiny_max_sqrt_area
        self.gaussian_alpha = gaussian_alpha
        self.gaussian_sigma_min = gaussian_sigma_min
        self.position_positive_weight = position_positive_weight
        self.loss_pos_weight = loss_pos_weight
        self.position_stride = position_stride
        if not hasattr(self.neck, 'forward_with_position'):
            raise TypeError(
                'PGRCFNFCOS requires a neck with forward_with_position()')

    def position_targets(
            self, position: Tensor, batch_data_samples: SampleList
    ) -> tuple[Tensor, Tensor]:
        """Build Gaussian targets and a valid loss mask on the P3 grid."""
        batch_size, _, height, width = position.shape
        target = position.new_zeros((batch_size, 1, height, width))
        valid = torch.zeros_like(target, dtype=torch.bool)
        ys = torch.arange(height, device=position.device,
                          dtype=position.dtype)[:, None]
        xs = torch.arange(width, device=position.device,
                          dtype=position.dtype)[None, :]

        for index, sample in enumerate(batch_data_samples):
            img_h, img_w = sample.metainfo['img_shape'][:2]
            valid_h = min(height, math.ceil(img_h / self.position_stride))
            valid_w = min(width, math.ceil(img_w / self.position_stride))
            valid[index, :, :valid_h, :valid_w] = True

            boxes = _box_tensor(sample.gt_instances.bboxes).to(
                device=position.device, dtype=position.dtype)
            if boxes.numel():
                sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
                tiny = self.tiny_mask(boxes, sample.metainfo)
                for box, size in zip(
                        boxes[tiny], sizes[tiny]):
                    center_x = (box[0] + box[2]) / (2 * self.position_stride)
                    center_y = (box[1] + box[3]) / (2 * self.position_stride)
                    sigma_x = torch.clamp(
                        self.gaussian_alpha * size[0] / self.position_stride,
                        min=self.gaussian_sigma_min)
                    sigma_y = torch.clamp(
                        self.gaussian_alpha * size[1] / self.position_stride,
                        min=self.gaussian_sigma_min)
                    gaussian = torch.exp(
                        -(xs - center_x).square() / (2 * sigma_x.square())
                        -(ys - center_y).square() / (2 * sigma_y.square()))
                    target[index, 0] = torch.maximum(
                        target[index, 0], gaussian)

            ignored = sample.get('ignored_instances', None)
            if ignored is None or not hasattr(ignored, 'bboxes'):
                continue
            ignored_boxes = _box_tensor(ignored.bboxes).to(
                device=position.device, dtype=position.dtype)
            for box in ignored_boxes:
                x1 = min(
                    width, max(
                        0, math.floor(float(box[0] / self.position_stride))))
                y1 = min(
                    height, max(
                        0, math.floor(float(box[1] / self.position_stride))))
                x2 = min(
                    width, max(
                        0, math.ceil(float(box[2] / self.position_stride))))
                y2 = min(
                    height, max(
                        0, math.ceil(float(box[3] / self.position_stride))))
                if x2 > x1 and y2 > y1:
                    valid[index, :, y1:y2, x1:x2] = False
        return target, valid

    def tiny_mask(self, boxes: Tensor, metainfo: dict) -> Tensor:
        """Select tiny boxes by their size in original-image coordinates."""
        if boxes.numel() == 0:
            return torch.zeros(
                boxes.shape[:-1], dtype=torch.bool, device=boxes.device)
        scale_factor = metainfo.get('scale_factor')
        if scale_factor is None:
            img_shape = metainfo.get('img_shape')
            ori_shape = metainfo.get('ori_shape')
            if img_shape is not None and ori_shape is not None:
                scale_factor = (
                    img_shape[1] / ori_shape[1],
                    img_shape[0] / ori_shape[0])
            else:
                scale_factor = (1.0, 1.0)
        scale = boxes.new_tensor(scale_factor).flatten()[:2]
        if scale.numel() != 2 or bool((scale <= 0).any()):
            raise ValueError(
                'scale_factor must contain positive (width, height) scales')
        transformed_sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
        original_sizes = transformed_sizes / scale
        sqrt_area = (
            original_sizes[:, 0] * original_sizes[:, 1]).sqrt()
        return sqrt_area <= self.tiny_max_sqrt_area

    def position_loss(
            self, position: Tensor, batch_data_samples: SampleList
    ) -> Tensor:
        target, valid = self.position_targets(position, batch_data_samples)
        weight = 1 + (self.position_positive_weight - 1) * target
        loss = F.binary_cross_entropy(
            position, target, reduction='none') * weight
        return loss[valid].sum() / valid.sum().clamp_min(1)

    def _loss_once(self, batch_inputs: Tensor,
                   batch_data_samples: SampleList) -> dict:
        backbone_features = self.backbone(batch_inputs)
        features, position, _ = self.neck.forward_with_position(
            backbone_features)
        losses = self.bbox_head.loss(features, batch_data_samples)
        losses['loss_pos'] = (
            self.position_loss(position, batch_data_samples)
            * self.loss_pos_weight)
        return losses

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        if self.has_api_loss():
            self.capture_api()
        losses = self._loss_once(batch_inputs, batch_data_samples)
        if self.has_api_loss():
            losses = self.api_augmented_losses(
                batch_inputs, batch_data_samples, losses,
                lambda: self._loss_once(batch_inputs, batch_data_samples))
        return losses

    def position_maps(
            self, batch_inputs: Tensor
    ) -> tuple[tuple[Tensor, ...], Tensor, Optional[Tensor]]:
        """Expose feature and reliability maps for diagnostics."""
        return self.neck.forward_with_position(self.backbone(batch_inputs))
