from __future__ import annotations

import torch

from mmdet.models.dense_heads.tood_head import TOODHead
from mmdet.registry import MODELS

from .atss_dgfe import DGFEDenseHeadMixin
from ..dgfe_core.spatial_targets import max_iou_per_box


@MODELS.register_module()
class TOODDGFEHead(DGFEDenseHeadMixin, TOODHead):
    """TOOD head with DGFE task-aligned metadata export."""

    def get_targets(self, *args, **kwargs):
        targets = super().get_targets(*args, **kwargs)
        self._dgfe_last_targets = targets
        return targets

    def loss_by_feat(self, cls_scores, bbox_preds, batch_gt_instances,
                     batch_img_metas, batch_gt_instances_ignore=None) -> dict:
        losses = super().loss_by_feat(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas,
            batch_gt_instances_ignore)
        self._collect_dgfe_records(batch_gt_instances)
        return losses

    def _collect_dgfe_records(self, batch_gt_instances) -> None:
        targets = getattr(self, '_dgfe_last_targets', None)
        if targets is None:
            self._dgfe_assignment_records = []
            return
        _, labels_list, _, bbox_targets_list, alignment_metrics_list = targets
        records = []
        for level, (labels, bbox_targets, metrics) in enumerate(
                zip(labels_list, bbox_targets_list, alignment_metrics_list)):
            device = labels.device
            labels_flat = labels.reshape(labels.shape[0], -1)
            targets_flat = bbox_targets.reshape(bbox_targets.shape[0], -1, 4)
            metrics_flat = metrics.reshape(metrics.shape[0], -1)
            for batch_idx, gt_instances in enumerate(batch_gt_instances):
                pos = (labels_flat[batch_idx] >= 0) & (
                    labels_flat[batch_idx] < self.num_classes)
                if not pos.any() or gt_instances.bboxes.numel() == 0:
                    continue
                gt_boxes = gt_instances.bboxes
                gt_boxes = gt_boxes.tensor if hasattr(gt_boxes, 'tensor') else gt_boxes
                fallback_iou, gt_ids = max_iou_per_box(
                    targets_flat[batch_idx][pos], gt_boxes.to(device=device))
                quality = metrics_flat[batch_idx][pos].detach()
                quality = torch.where(quality > 0, quality,
                                      fallback_iou.detach())
                for gt_idx, q in zip(gt_ids.tolist(), quality.tolist()):
                    records.append({
                        'batch_idx': batch_idx,
                        'level': level,
                        'gt_idx': int(gt_idx),
                        'quality': float(q),
                    })
        self._dgfe_assignment_records = records
