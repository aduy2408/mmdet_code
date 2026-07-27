from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.detectors import FCOS
from mmdet.models.losses import GaussianFocalLoss
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList


def _box_tensor(boxes) -> Tensor:
    return boxes.tensor if hasattr(boxes, 'tensor') else boxes


@MODELS.register_module()
class PAHRFCOS(FCOS):
    """FCOS with supervised position-aware Haar recomposition."""

    def __init__(self,
                 *args,
                 tiny_max_sqrt_area: float = 16.0,
                 position_stride: int = 8,
                 loss_pos_weight: float = 0.1,
                 loss_offset_weight: float = 0.1,
                 use_phase_shift: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self.neck, 'forward_with_aux'):
            raise TypeError('PAHRFCOS requires a neck with forward_with_aux()')
        self.tiny_max_sqrt_area = float(tiny_max_sqrt_area)
        self.position_stride = int(position_stride)
        self.loss_pos_weight = float(loss_pos_weight)
        self.loss_offset_weight = float(loss_offset_weight)
        self.use_phase_shift = bool(use_phase_shift)
        self.position_loss_module = GaussianFocalLoss(reduction='mean')

    def tiny_mask(self, boxes: Tensor, metainfo: dict) -> Tensor:
        if boxes.numel() == 0:
            return torch.zeros(
                boxes.shape[:-1], dtype=torch.bool, device=boxes.device)
        scale_factor = metainfo.get('scale_factor')
        if scale_factor is None:
            image_shape = metainfo.get('img_shape')
            original_shape = metainfo.get('ori_shape')
            if image_shape is not None and original_shape is not None:
                scale_factor = (
                    image_shape[1] / original_shape[1],
                    image_shape[0] / original_shape[0])
            else:
                scale_factor = (1.0, 1.0)
        scale = boxes.new_tensor(scale_factor).flatten()[:2]
        if scale.numel() != 2 or bool((scale <= 0).any()):
            raise ValueError(
                'scale_factor must contain positive width and height scales')
        sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0) / scale
        return sizes.prod(dim=1).sqrt() <= self.tiny_max_sqrt_area

    def auxiliary_targets(
            self, position_logits: Tensor, batch_data_samples: SampleList
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, _, height, width = position_logits.shape
        target = position_logits.new_zeros((batch_size, 1, height, width))
        offset_target = position_logits.new_zeros(
            (batch_size, 2, height, width))
        valid = torch.zeros_like(target, dtype=torch.bool)
        offset_valid = torch.zeros_like(target, dtype=torch.bool)
        selected_area = position_logits.new_full(
            (batch_size, height, width), torch.inf)
        ys = torch.arange(
            height, device=position_logits.device,
            dtype=position_logits.dtype)[:, None]
        xs = torch.arange(
            width, device=position_logits.device,
            dtype=position_logits.dtype)[None, :]

        for image_index, sample in enumerate(batch_data_samples):
            image_height, image_width = sample.metainfo['img_shape'][:2]
            valid_height = min(
                height, math.ceil(image_height / self.position_stride))
            valid_width = min(
                width, math.ceil(image_width / self.position_stride))
            valid[image_index, :, :valid_height, :valid_width] = True

            boxes = _box_tensor(sample.gt_instances.bboxes).to(
                device=position_logits.device, dtype=position_logits.dtype)
            sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
            tiny = self.tiny_mask(boxes, sample.metainfo)
            for box, size in zip(boxes[tiny], sizes[tiny]):
                center_x = (box[0] + box[2]) / (2 * self.position_stride)
                center_y = (box[1] + box[3]) / (2 * self.position_stride)
                sigma_x = (size[0] / (2 * self.position_stride)).clamp(
                    min=0.5, max=1.0)
                sigma_y = (size[1] / (2 * self.position_stride)).clamp(
                    min=0.5, max=1.0)
                gaussian = torch.exp(
                    -(xs - center_x).square() / (2 * sigma_x.square())
                    -(ys - center_y).square() / (2 * sigma_y.square()))
                cell_x = int(torch.floor(center_x).item())
                cell_y = int(torch.floor(center_y).item())
                if 0 <= cell_x < valid_width and 0 <= cell_y < valid_height:
                    gaussian[cell_y, cell_x] = 1
                    area = size.prod()
                    if area < selected_area[image_index, cell_y, cell_x]:
                        offset_target[image_index, 0, cell_y, cell_x] = (
                            center_x - cell_x)
                        offset_target[image_index, 1, cell_y, cell_x] = (
                            center_y - cell_y)
                        offset_valid[image_index, 0, cell_y, cell_x] = True
                        selected_area[image_index, cell_y, cell_x] = area
                target[image_index, 0] = torch.maximum(
                    target[image_index, 0], gaussian)

            ignored = sample.get('ignored_instances', None)
            if ignored is None or not hasattr(ignored, 'bboxes'):
                continue
            ignored_boxes = _box_tensor(ignored.bboxes).to(
                device=position_logits.device, dtype=position_logits.dtype)
            for box in ignored_boxes:
                x1 = max(0, min(width, math.floor(
                    float(box[0] / self.position_stride))))
                y1 = max(0, min(height, math.floor(
                    float(box[1] / self.position_stride))))
                x2 = max(0, min(width, math.ceil(
                    float(box[2] / self.position_stride))))
                y2 = max(0, min(height, math.ceil(
                    float(box[3] / self.position_stride))))
                valid[image_index, :, y1:y2, x1:x2] = False
                offset_valid[image_index, :, y1:y2, x1:x2] = False
        return target, offset_target, valid, offset_valid

    def auxiliary_losses(self, aux: dict[str, Tensor],
                         batch_data_samples: SampleList) -> dict[str, Tensor]:
        position_logits = aux['position_logits']
        target, offset_target, valid, offset_valid = self.auxiliary_targets(
            position_logits, batch_data_samples)
        center_count = offset_valid.sum().clamp_min(1)
        loss_position = self.position_loss_module(
            position_logits.sigmoid(),
            target,
            weight=valid.to(position_logits.dtype),
            avg_factor=center_count)
        if offset_valid.any():
            mask = offset_valid.expand_as(aux['offsets'])
            loss_offset = F.smooth_l1_loss(
                aux['offsets'][mask],
                offset_target[mask],
                reduction='sum') / center_count
        else:
            loss_offset = aux['offsets'].sum() * 0
        return dict(
            loss_pos=loss_position * self.loss_pos_weight,
            loss_offset=loss_offset * self.loss_offset_weight)

    def phase_adjust_bbox_preds(
            self, bbox_preds: list[Tensor],
            aux: dict[str, Tensor]) -> list[Tensor]:
        """Shift P3 boxes toward the fractional center predicted by PAHR."""
        adjusted = list(bbox_preds)
        if not self.use_phase_shift:
            return adjusted
        position = aux['position_logits'].sigmoid()
        offsets = aux['offsets']
        shift_scale = float(self.position_stride)
        if self.bbox_head.norm_on_bbox and self.training:
            shift_scale = 1.0
        dx = position * (offsets[:, 0:1] - 0.5) * shift_scale
        dy = position * (offsets[:, 1:2] - 0.5) * shift_scale
        left, top, right, bottom = adjusted[0].chunk(4, dim=1)
        adjusted[0] = torch.cat(
            (left - dx, top - dy, right + dx, bottom + dy),
            dim=1).clamp_min(0)
        return adjusted

    def bbox_outputs(
            self, features: tuple[Tensor, ...],
            aux: dict[str, Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        cls_scores, bbox_preds, centernesses = self.bbox_head(features)
        bbox_preds = self.phase_adjust_bbox_preds(bbox_preds, aux)
        return cls_scores, bbox_preds, centernesses

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict[str, Tensor]:
        backbone_features = self.backbone(batch_inputs)
        features, aux = self.neck.forward_with_aux(backbone_features)
        outputs = self.bbox_outputs(features, aux)
        gt_instances, ignored_instances, image_metas = unpack_gt_instances(
            batch_data_samples)
        losses = self.bbox_head.loss_by_feat(
            *outputs, gt_instances, image_metas, ignored_instances)
        losses.update(self.auxiliary_losses(aux, batch_data_samples))
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        backbone_features = self.backbone(batch_inputs)
        features, aux = self.neck.forward_with_aux(backbone_features)
        outputs = self.bbox_outputs(features, aux)
        image_metas = [sample.metainfo for sample in batch_data_samples]
        results = self.bbox_head.predict_by_feat(
            *outputs,
            batch_img_metas=image_metas,
            rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, results)

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        backbone_features = self.backbone(batch_inputs)
        features, aux = self.neck.forward_with_aux(backbone_features)
        return self.bbox_outputs(features, aux)

    def position_maps(
            self, batch_inputs: Tensor
    ) -> tuple[tuple[Tensor, ...], dict[str, Tensor]]:
        return self.neck.forward_with_aux(self.backbone(batch_inputs))
