from __future__ import annotations

import torch
from torch import Tensor

from mmdet.models.detectors.cascade_rcnn import CascadeRCNN
from mmdet.models.detectors.faster_rcnn import FasterRCNN
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor

from ..dgfe_core.spatial_targets import build_quality_spatial_target


def _aligned_iou(boxes: Tensor, targets: Tensor) -> Tensor:
    lt = boxes[:, :2].maximum(targets[:, :2])
    rb = boxes[:, 2:].minimum(targets[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    area_b = (targets[:, 2] - targets[:, 0]).clamp(min=0) * (
        targets[:, 3] - targets[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter).clamp(min=1e-6)


class DGFETwoStageRoIMixin:
    """Export exact sampled ROI assignments for DGFE targets."""

    def dgfe_assignment_records(self) -> list[dict]:
        return list(getattr(self, '_dgfe_assignment_records', []))

    def dgfe_localization_loss_names(self, losses: dict) -> set[str]:
        return {
            name for name, value in losses.items()
            if 'loss_bbox' in name.lower()
            and not name.lower().startswith('rpn_')
            and (isinstance(value, (list, tuple)) or hasattr(value, 'mean'))
        }

    def build_dgfe_spatial_target(self, detector, logits: Tensor,
                                  batch_inputs: Tensor,
                                  batch_data_samples):
        return build_quality_spatial_target(
            detector, logits, batch_inputs, batch_data_samples,
            self.dgfe_assignment_records())

    @staticmethod
    def _records_from_bbox(stage: int, bbox_head, bbox_results: dict,
                           sampling_results) -> list[dict]:
        bbox_pred = bbox_results.get('bbox_pred')
        if bbox_pred is None:
            return []
        records = []
        offset = 0
        for batch_idx, result in enumerate(sampling_results):
            count = len(result.priors)
            num_pos = int(result.pos_inds.numel())
            if num_pos == 0:
                offset += count
                continue
            pred = bbox_pred[offset:offset + num_pos]
            pos_labels = result.pos_gt_labels
            if not bbox_head.reg_class_agnostic:
                pred = pred.reshape(num_pos, bbox_head.num_classes, -1)
                pred = pred[torch.arange(num_pos, device=pred.device),
                            pos_labels.long()]
            else:
                pred = pred.reshape(num_pos, -1)
            decoded = get_box_tensor(
                bbox_head.bbox_coder.decode(result.pos_priors, pred))
            gt_boxes = get_box_tensor(result.pos_gt_bboxes).to(decoded)
            qualities = _aligned_iou(decoded, gt_boxes).detach()
            for idx in range(num_pos):
                records.append({
                    'batch_idx': batch_idx,
                    'stage': stage,
                    'gt_idx': int(result.pos_assigned_gt_inds[idx]),
                    'quality': float(qualities[idx]),
                    'roi': result.pos_priors[idx].detach(),
                    'decoded_bbox': decoded[idx].detach(),
                    'gt_bbox': gt_boxes[idx].detach(),
                })
            offset += count
        return records


@MODELS.register_module()
class DGFEStandardRoIHead(DGFETwoStageRoIMixin, StandardRoIHead):

    def loss(self, *args, **kwargs):
        self._dgfe_assignment_records = []
        return super().loss(*args, **kwargs)

    def bbox_loss(self, x, sampling_results):
        results = super().bbox_loss(x, sampling_results)
        self._dgfe_assignment_records = self._records_from_bbox(
            0, self.bbox_head, results, sampling_results)
        return results


@MODELS.register_module()
class DGFECascadeRoIHead(DGFETwoStageRoIMixin, CascadeRoIHead):

    def loss(self, *args, **kwargs):
        self._dgfe_stage_records = {}
        self._dgfe_assignment_records = []
        return super().loss(*args, **kwargs)

    def bbox_loss(self, stage, x, sampling_results):
        results = super().bbox_loss(stage, x, sampling_results)
        records = self._records_from_bbox(
            stage, self.bbox_head[stage], results, sampling_results)
        self._dgfe_stage_records[stage] = records
        # Cascade spatial quality is deliberately final-stage only.
        if stage == self.num_stages - 1:
            self._dgfe_assignment_records = records
        return results

    def dgfe_stage_records(self) -> dict[int, list[dict]]:
        return {
            stage: list(records)
            for stage, records in getattr(self, '_dgfe_stage_records', {}).items()
        }


class _DGFEHybridDetectorMixin:

    def dgfe_adapters(self) -> list:
        adapter = getattr(self, 'roi_head', None)
        return ([adapter] if adapter is not None
                and hasattr(adapter, 'build_dgfe_spatial_target') else [])

    def dgfe_replay_losses(self, features, batch_data_samples) -> dict:
        """Head-only replay seam; feature perturbation remains design-only."""
        return self.loss_from_features(features, batch_data_samples)


@MODELS.register_module()
class DGFEFasterRCNN(_DGFEHybridDetectorMixin, FasterRCNN):
    pass


@MODELS.register_module()
class DGFECascadeRCNN(_DGFEHybridDetectorMixin, CascadeRCNN):
    pass
