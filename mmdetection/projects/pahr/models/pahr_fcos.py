from __future__ import annotations

import math

import torch
import torch.nn as nn
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
                 use_tiny_measurement: bool = False,
                 measurement_channels: int = 16,
                 measurement_center_weight: float = 0.1,
                 measurement_phase_weight: float = 0.1,
                 measurement_size_weight: float = 0.1,
                 measurement_size_blend: float = 0.5,
                 measurement_tiny_limit: float = 24.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self.neck, 'forward_with_aux'):
            raise TypeError('PAHRFCOS requires a neck with forward_with_aux()')
        self.tiny_max_sqrt_area = float(tiny_max_sqrt_area)
        self.position_stride = int(position_stride)
        self.loss_pos_weight = float(loss_pos_weight)
        self.loss_offset_weight = float(loss_offset_weight)
        self.use_phase_shift = bool(use_phase_shift)
        self.use_tiny_measurement = bool(use_tiny_measurement)
        self.position_loss_module = GaussianFocalLoss(reduction='mean')
        if self.use_tiny_measurement:
            if measurement_channels < 1:
                raise ValueError('measurement_channels must be positive')
            self.measurement_center_weight = float(measurement_center_weight)
            self.measurement_phase_weight = float(measurement_phase_weight)
            self.measurement_size_weight = float(measurement_size_weight)
            self.measurement_size_blend = float(measurement_size_blend)
            self.measurement_tiny_limit = float(measurement_tiny_limit)
            self.measurement_stem = nn.Sequential(
                nn.Conv2d(3, measurement_channels, 3, stride=2, padding=1),
                nn.Conv2d(
                    measurement_channels,
                    measurement_channels,
                    3,
                    padding=1,
                    groups=measurement_channels),
                nn.SiLU(inplace=True))
            packed_channels = 16 * measurement_channels
            self.measurement_center = nn.Conv2d(
                measurement_channels, 1, 1)
            self.measurement_phase = nn.Conv2d(packed_channels, 16, 1)
            self.measurement_size = nn.Conv2d(packed_channels, 2, 1)
            self.measurement_refine_scale = nn.Parameter(torch.zeros(()))
            phase_axis = torch.tensor([-3., -1., 1., 3.])
            phase_y, phase_x = torch.meshgrid(
                phase_axis, phase_axis, indexing='ij')
            self.register_buffer(
                'measurement_phase_x', phase_x.flatten(), persistent=False)
            self.register_buffer(
                'measurement_phase_y', phase_y.flatten(), persistent=False)
            self._init_measurement_outputs()

    def _init_measurement_outputs(self) -> None:
        if not self.use_tiny_measurement:
            return
        nn.init.zeros_(self.measurement_phase.weight)
        nn.init.zeros_(self.measurement_phase.bias)
        nn.init.zeros_(self.measurement_size.weight)
        nn.init.zeros_(self.measurement_size.bias)
        nn.init.zeros_(self.measurement_refine_scale)
        nn.init.constant_(self.measurement_center.bias, -2.19)

    def init_weights(self) -> None:
        super().init_weights()
        self._init_measurement_outputs()

    def measurement_maps(self, batch_inputs: Tensor) -> dict[str, Tensor]:
        if not self.use_tiny_measurement:
            return {}
        if batch_inputs.shape[-2] % 8 or batch_inputs.shape[-1] % 8:
            raise ValueError('tiny measurement input must be divisible by 8')
        feature = self.measurement_stem(batch_inputs)
        packed = F.pixel_unshuffle(feature, 4)
        return dict(
            measurement_center_logits=self.measurement_center(feature),
            measurement_phase_logits=self.measurement_phase(packed),
            measurement_log_sizes=self.measurement_size(packed))

    def measurement_targets(
            self, maps: dict[str, Tensor],
            batch_data_samples: SampleList
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        logits = maps['measurement_center_logits']
        batch_size, _, height, width = logits.shape
        p3_height, p3_width = height // 4, width // 4
        center_target = logits.new_zeros((batch_size, 1, height, width))
        center_valid = torch.zeros_like(center_target, dtype=torch.bool)
        phase_target = torch.full(
            (batch_size, p3_height, p3_width),
            -1, dtype=torch.long, device=logits.device)
        size_target = logits.new_zeros(
            (batch_size, 2, p3_height, p3_width))
        size_valid = torch.zeros(
            (batch_size, 1, p3_height, p3_width),
            dtype=torch.bool, device=logits.device)
        selected_area = logits.new_full(
            (batch_size, p3_height, p3_width), torch.inf)
        ys = torch.arange(
            height, device=logits.device, dtype=logits.dtype)[:, None]
        xs = torch.arange(
            width, device=logits.device, dtype=logits.dtype)[None, :]

        for image_index, sample in enumerate(batch_data_samples):
            image_height, image_width = sample.metainfo['img_shape'][:2]
            valid_height = min(height, math.ceil(image_height / 2))
            valid_width = min(width, math.ceil(image_width / 2))
            center_valid[image_index, :, :valid_height, :valid_width] = True
            boxes = _box_tensor(sample.gt_instances.bboxes).to(
                device=logits.device, dtype=logits.dtype)
            sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
            tiny = self.tiny_mask(boxes, sample.metainfo)
            for box, size in zip(boxes[tiny], sizes[tiny]):
                center_px = (box[:2] + box[2:]) * 0.5
                center = center_px / 2
                sigma = (size / 4).clamp(min=0.5, max=2.0)
                gaussian = torch.exp(
                    -(xs - center[0]).square() / (2 * sigma[0].square())
                    -(ys - center[1]).square() / (2 * sigma[1].square()))
                center_target[image_index, 0] = torch.maximum(
                    center_target[image_index, 0], gaussian)
                cell = torch.floor(center_px / self.position_stride).long()
                cell_x, cell_y = int(cell[0]), int(cell[1])
                if not (0 <= cell_x < p3_width
                        and 0 <= cell_y < p3_height):
                    continue
                area = size.prod()
                if area >= selected_area[image_index, cell_y, cell_x]:
                    continue
                phase = torch.floor(
                    (center_px - cell * self.position_stride) / 2).long()
                phase = phase.clamp(0, 3)
                phase_target[image_index, cell_y, cell_x] = (
                    phase[1] * 4 + phase[0])
                size_target[image_index, :, cell_y, cell_x] = (
                    size.clamp_min(1e-6) / 2).log()
                size_valid[image_index, 0, cell_y, cell_x] = True
                selected_area[image_index, cell_y, cell_x] = area
            ignored = sample.get('ignored_instances', None)
            if ignored is None or not hasattr(ignored, 'bboxes'):
                continue
            ignored_boxes = _box_tensor(ignored.bboxes).to(
                device=logits.device, dtype=logits.dtype)
            for box in ignored_boxes:
                x1 = max(0, min(width, math.floor(float(box[0] / 2))))
                y1 = max(0, min(height, math.floor(float(box[1] / 2))))
                x2 = max(0, min(width, math.ceil(float(box[2] / 2))))
                y2 = max(0, min(height, math.ceil(float(box[3] / 2))))
                center_valid[image_index, :, y1:y2, x1:x2] = False
                px1 = max(0, min(
                    p3_width,
                    math.floor(float(box[0] / self.position_stride))))
                py1 = max(0, min(
                    p3_height,
                    math.floor(float(box[1] / self.position_stride))))
                px2 = max(0, min(
                    p3_width,
                    math.ceil(float(box[2] / self.position_stride))))
                py2 = max(0, min(
                    p3_height,
                    math.ceil(float(box[3] / self.position_stride))))
                phase_target[image_index, py1:py2, px1:px2] = -1
                size_valid[image_index, :, py1:py2, px1:px2] = False
        return (center_target, center_valid, phase_target, size_target,
                size_valid)

    def measurement_losses(
            self, maps: dict[str, Tensor],
            batch_data_samples: SampleList) -> dict[str, Tensor]:
        if not self.use_tiny_measurement:
            return {}
        targets = self.measurement_targets(maps, batch_data_samples)
        center_target, center_valid, phase_target, size_target, valid = targets
        count = valid.sum().clamp_min(1)
        center_loss = self.position_loss_module(
            maps['measurement_center_logits'].sigmoid(),
            center_target,
            weight=center_valid.to(center_target.dtype),
            avg_factor=count)
        phase_loss = F.cross_entropy(
            maps['measurement_phase_logits'],
            phase_target,
            ignore_index=-1,
            reduction='sum') / count
        mask = valid.expand_as(size_target)
        if mask.any():
            size_loss = F.smooth_l1_loss(
                maps['measurement_log_sizes'][mask],
                size_target[mask],
                reduction='sum') / count
        else:
            size_loss = maps['measurement_log_sizes'].sum() * 0
        return dict(
            loss_measure_center=center_loss * self.measurement_center_weight,
            loss_measure_phase=phase_loss * self.measurement_phase_weight,
            loss_measure_size=size_loss * self.measurement_size_weight)

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
        position = aux.get(
            'phase_gate', aux['position_logits'].sigmoid())
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

    def measurement_adjust_bbox_preds(
            self, bbox_preds: list[Tensor],
            maps: dict[str, Tensor]) -> list[Tensor]:
        adjusted = list(bbox_preds)
        if not self.use_tiny_measurement:
            return adjusted
        bbox = adjusted[0]
        pixel_bbox = bbox
        normalized = self.bbox_head.norm_on_bbox and self.training
        if normalized:
            pixel_bbox = bbox * self.position_stride
        left, top, right, bottom = pixel_bbox.chunk(4, dim=1)
        width = left + right
        height = top + bottom

        probabilities = maps['measurement_phase_logits'].softmax(dim=1)
        phase_x = (
            probabilities * self.measurement_phase_x.view(1, -1, 1, 1)
        ).sum(dim=1, keepdim=True)
        phase_y = (
            probabilities * self.measurement_phase_y.view(1, -1, 1, 1)
        ).sum(dim=1, keepdim=True)
        center_probability = F.max_pool2d(
            maps['measurement_center_logits'].sigmoid(), 4, stride=4)
        size_gate = torch.sigmoid(
            (self.measurement_tiny_limit
             - (width * height).clamp_min(0).sqrt()) / 2)
        gate = (center_probability * size_gate).detach()
        strength = self.measurement_refine_scale.tanh()
        dx = gate * strength * phase_x
        dy = gate * strength * phase_y
        left = left - dx
        right = right + dx
        top = top - dy
        bottom = bottom + dy

        measured_size = maps['measurement_log_sizes'].clamp(
            min=-2, max=4).exp() * 2
        measured_width, measured_height = measured_size.chunk(2, dim=1)
        blend = gate * strength * self.measurement_size_blend
        width_delta = (measured_width - width) * blend * 0.5
        height_delta = (measured_height - height) * blend * 0.5
        refined = torch.cat((
            left + width_delta,
            top + height_delta,
            right + width_delta,
            bottom + height_delta), dim=1).clamp_min(0)
        if normalized:
            refined = refined / self.position_stride
        adjusted[0] = refined
        return adjusted

    def bbox_outputs(
            self, features: tuple[Tensor, ...],
            aux: dict[str, Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        cls_scores, bbox_preds, centernesses = self.bbox_head(features)
        bbox_preds = self.phase_adjust_bbox_preds(bbox_preds, aux)
        bbox_preds = self.measurement_adjust_bbox_preds(bbox_preds, aux)
        return cls_scores, bbox_preds, centernesses

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict[str, Tensor]:
        backbone_features = self.backbone(batch_inputs)
        features, aux = self.neck.forward_with_aux(backbone_features)
        aux.update(self.measurement_maps(batch_inputs))
        outputs = self.bbox_outputs(features, aux)
        gt_instances, ignored_instances, image_metas = unpack_gt_instances(
            batch_data_samples)
        losses = self.bbox_head.loss_by_feat(
            *outputs, gt_instances, image_metas, ignored_instances)
        losses.update(self.auxiliary_losses(aux, batch_data_samples))
        losses.update(self.measurement_losses(aux, batch_data_samples))
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        backbone_features = self.backbone(batch_inputs)
        features, aux = self.neck.forward_with_aux(backbone_features)
        aux.update(self.measurement_maps(batch_inputs))
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
        aux.update(self.measurement_maps(batch_inputs))
        return self.bbox_outputs(features, aux)

    def position_maps(
            self, batch_inputs: Tensor
    ) -> tuple[tuple[Tensor, ...], dict[str, Tensor]]:
        features, aux = self.neck.forward_with_aux(
            self.backbone(batch_inputs))
        aux.update(self.measurement_maps(batch_inputs))
        return features, aux
