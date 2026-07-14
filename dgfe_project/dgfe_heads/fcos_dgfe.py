from __future__ import annotations

from mmdet.models.dense_heads.fcos_head import FCOSHead
from mmdet.registry import MODELS

from .atss_dgfe import DGFEDenseHeadMixin
from ..dgfe_core.spatial_targets import aligned_iou, max_iou_per_box


@MODELS.register_module()
class FCOSDGFEHead(DGFEDenseHeadMixin, FCOSHead):
    """FCOS head exporting assigned-GT localization quality for DGFE."""

    def get_targets(self, *args, **kwargs):
        targets = super().get_targets(*args, **kwargs)
        self._dgfe_last_targets = targets
        return targets

    def loss_by_feat(self, cls_scores, bbox_preds, centernesses,
                     batch_gt_instances, batch_img_metas,
                     batch_gt_instances_ignore=None) -> dict:
        self._dgfe_assignment_records = []
        losses = super().loss_by_feat(
            cls_scores, bbox_preds, centernesses, batch_gt_instances,
            batch_img_metas, batch_gt_instances_ignore)
        self._collect_dgfe_records(bbox_preds, batch_gt_instances)
        return losses

    def _collect_dgfe_records(self, bbox_preds, batch_gt_instances) -> None:
        targets = getattr(self, '_dgfe_last_targets', None)
        if targets is None:
            self._dgfe_assignment_records = []
            return
        labels_list, bbox_targets_list = targets
        featmap_sizes = [pred.shape[-2:] for pred in bbox_preds]
        points_list = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        num_imgs = bbox_preds[0].shape[0]
        records = []

        for level, (labels, bbox_targets, bbox_pred, points, stride) in enumerate(
                zip(labels_list, bbox_targets_list, bbox_preds, points_list,
                    self.strides)):
            labels = labels.reshape(num_imgs, -1)
            bbox_targets = bbox_targets.reshape(num_imgs, -1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, 4)
            distance_scale = float(stride[0] if isinstance(stride, tuple)
                                   else stride) if self.norm_on_bbox else 1.0

            for batch_idx, gt_instances in enumerate(batch_gt_instances):
                pos = ((labels[batch_idx] >= 0) &
                       (labels[batch_idx] < self.num_classes))
                gt_boxes = gt_instances.bboxes
                gt_boxes = (gt_boxes.tensor
                            if hasattr(gt_boxes, 'tensor') else gt_boxes)
                if not pos.any() or gt_boxes.numel() == 0:
                    continue
                pos_points = points[pos]
                target_boxes = self.bbox_coder.decode(
                    pos_points, bbox_targets[batch_idx][pos] * distance_scale)
                pred_boxes = self.bbox_coder.decode(
                    pos_points, bbox_pred[batch_idx][pos] * distance_scale)
                target_boxes = (target_boxes.tensor if hasattr(
                    target_boxes, 'tensor') else target_boxes)
                pred_boxes = (pred_boxes.tensor if hasattr(
                    pred_boxes, 'tensor') else pred_boxes)
                gt_boxes = gt_boxes.to(device=pred_boxes.device,
                                       dtype=pred_boxes.dtype)
                _, gt_ids = max_iou_per_box(target_boxes, gt_boxes)
                assigned_gt = gt_boxes[gt_ids]
                qualities = aligned_iou(pred_boxes, assigned_gt).detach()
                for local_idx, (gt_idx, quality) in enumerate(zip(
                        gt_ids.tolist(), qualities.tolist())):
                    records.append({
                        'batch_idx': batch_idx,
                        'level': level,
                        'gt_idx': int(gt_idx),
                        'quality': float(quality),
                        'pred_box': pred_boxes[local_idx].detach().tolist(),
                        'target_box': target_boxes[local_idx].detach().tolist(),
                    })
        self._dgfe_assignment_records = records
