from __future__ import annotations

from torch import Tensor

from mmdet.models.dense_heads.atss_head import ATSSHead
from mmdet.registry import MODELS

from ..dgfe_core.api_boxgrad import localization_loss_names
from ..dgfe_core.spatial_targets import (build_quality_spatial_target,
                                         max_iou_per_box)


class DGFEDenseHeadMixin:
    dgfe_loss_prefixes: tuple[str, ...] = ()

    def dgfe_localization_loss_names(self, losses: dict) -> set[str]:
        return localization_loss_names(losses, self.dgfe_loss_prefixes)

    def dgfe_assignment_records(self) -> list[dict]:
        return list(getattr(self, '_dgfe_assignment_records', []))

    def build_dgfe_spatial_target(self, detector, logits: Tensor,
                                  batch_inputs: Tensor,
                                  batch_data_samples):
        return build_quality_spatial_target(
            detector, logits, batch_inputs, batch_data_samples,
            self.dgfe_assignment_records())


@MODELS.register_module()
class ATSSDGFEHead(DGFEDenseHeadMixin, ATSSHead):
    """ATSS head with DGFE assignment metadata export."""

    def get_targets(self, *args, **kwargs):
        targets = super().get_targets(*args, **kwargs)
        self._dgfe_last_targets = targets
        return targets

    def loss_by_feat(self, cls_scores, bbox_preds, centernesses,
                     batch_gt_instances, batch_img_metas,
                     batch_gt_instances_ignore=None) -> dict:
        losses = super().loss_by_feat(
            cls_scores, bbox_preds, centernesses, batch_gt_instances,
            batch_img_metas, batch_gt_instances_ignore)
        self._collect_dgfe_records(bbox_preds, batch_gt_instances)
        return losses

    def _collect_dgfe_records(self, bbox_preds, batch_gt_instances) -> None:
        device = bbox_preds[0].device
        targets = getattr(self, '_dgfe_last_targets', None)
        if targets is None:
            self._dgfe_assignment_records = []
            return
        anchors_list, labels_list, _, bbox_targets_list, _, _ = targets
        records = []
        for level, (anchors, labels, bbox_targets, bbox_pred) in enumerate(
                zip(anchors_list, labels_list, bbox_targets_list, bbox_preds)):
            decoded = self.bbox_coder.decode(
                anchors.reshape(-1, 4),
                bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flat = labels.reshape(labels.shape[0], -1)
            targets_flat = bbox_targets.reshape(bbox_targets.shape[0], -1, 4)
            decoded = decoded.reshape(labels.shape[0], -1, 4)
            for batch_idx, gt_instances in enumerate(batch_gt_instances):
                pos = (labels_flat[batch_idx] >= 0) & (
                    labels_flat[batch_idx] < self.num_classes)
                if not pos.any() or gt_instances.bboxes.numel() == 0:
                    continue
                gt_boxes = gt_instances.bboxes
                gt_boxes = gt_boxes.tensor if hasattr(gt_boxes, 'tensor') else gt_boxes
                _, gt_ids = max_iou_per_box(targets_flat[batch_idx][pos],
                                            gt_boxes.to(device=device))
                pred_iou, _ = max_iou_per_box(decoded[batch_idx][pos],
                                              gt_boxes.to(device=device))
                for gt_idx, quality in zip(gt_ids.tolist(), pred_iou.detach().tolist()):
                    records.append({
                        'batch_idx': batch_idx,
                        'level': level,
                        'gt_idx': int(gt_idx),
                        'quality': float(quality),
                    })
        self._dgfe_assignment_records = records
