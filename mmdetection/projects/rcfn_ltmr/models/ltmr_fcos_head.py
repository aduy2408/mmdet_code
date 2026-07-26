from typing import List

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.dense_heads import FCOSHead
from mmdet.registry import MODELS
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean


def _inside_boxes(points: Tensor, boxes: Tensor) -> Tensor:
    if boxes.numel() == 0:
        return torch.zeros(
            points.size(0), dtype=torch.bool, device=points.device)
    x, y = points[:, 0:1], points[:, 1:2]
    return ((x >= boxes[:, 0]) & (x < boxes[:, 2])
            & (y >= boxes[:, 1]) & (y < boxes[:, 3])).any(dim=1)


def fcos_assigned_gt_inds(
        head: FCOSHead, points: Tensor, gt_instances: InstanceData,
        regress_ranges: Tensor, num_points_per_lvl: List[int]) -> Tensor:
    """Recover the exact FCOS GT assignment; background is -1."""
    labels, bbox_targets = FCOSHead._get_targets_single(
        head, gt_instances, points, regress_ranges, num_points_per_lvl)
    assigned = labels.new_full(labels.shape, -1)
    positive = labels < head.num_classes
    if not positive.any():
        return assigned
    pos_points = points[positive]
    targets = bbox_targets[positive]
    decoded = torch.stack(
        (pos_points[:, 0] - targets[:, 0],
         pos_points[:, 1] - targets[:, 1],
         pos_points[:, 0] + targets[:, 2],
         pos_points[:, 1] + targets[:, 3]),
        dim=1)
    distance = (
        decoded[:, None] - gt_instances.bboxes[None]).abs().sum(dim=2)
    assigned[positive] = distance.argmin(dim=1)
    return assigned


@MODELS.register_module()
class LTMRFCOSHead(FCOSHead):
    """FCOS head with training-only local tiny-object margin loss."""

    def __init__(self, *args, tiny_max_sqrt_area: float = 16.0,
                 radius: int = 2, topk: int = 5, margin: float = 1.0,
                 loss_weight: float = 0.05, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tiny_max_sqrt_area = tiny_max_sqrt_area
        self.radius = radius
        self.topk = topk
        self.margin = margin
        self.loss_weight = loss_weight
        self.ltmr_weight = loss_weight

    def assigned_gt_inds(
            self, points: Tensor, gt_instances: InstanceData,
            regress_ranges: Tensor, num_points_per_lvl: List[int]) -> Tensor:
        return fcos_assigned_gt_inds(
            self, points, gt_instances, regress_ranges, num_points_per_lvl)

    def _p3_assignments(
            self, points: List[Tensor],
            batch_gt_instances: InstanceList) -> List[Tensor]:
        num_points = [item.size(0) for item in points]
        all_points = torch.cat(points)
        ranges = torch.cat([
            point.new_tensor(self.regress_ranges[level])[None].expand_as(point)
            for level, point in enumerate(points)
        ])
        p3_count = num_points[0]
        return [
            self.assigned_gt_inds(
                all_points, instances, ranges, num_points)[:p3_count]
            for instances in batch_gt_instances
        ]

    def local_tiny_margin_loss(
            self, cls_score_p3: Tensor, points_p3: Tensor,
            assignments: List[Tensor], batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> Tensor:
        total = cls_score_p3.sum() * 0
        count = 0
        stride = self.strides[0]

        for image_index, gt_instances in enumerate(batch_gt_instances):
            boxes = gt_instances.bboxes
            labels = gt_instances.labels
            if boxes.numel() == 0:
                continue
            areas = ((boxes[:, 2] - boxes[:, 0])
                     * (boxes[:, 3] - boxes[:, 1])).clamp_min(0)
            tiny = areas.sqrt() <= self.tiny_max_sqrt_area
            assigned = assignments[image_index]
            img_h, img_w = batch_img_metas[image_index]['img_shape'][:2]
            valid = ((points_p3[:, 0] < img_w)
                     & (points_p3[:, 1] < img_h))
            inside_gt = _inside_boxes(points_p3, boxes)
            ignored = torch.zeros_like(valid)
            if (batch_gt_instances_ignore is not None
                    and batch_gt_instances_ignore[image_index] is not None):
                ignored = _inside_boxes(
                    points_p3,
                    batch_gt_instances_ignore[image_index].bboxes)

            for gt_index in tiny.nonzero().flatten():
                positive = assigned == gt_index
                if not positive.any():
                    continue
                class_id = labels[gt_index]
                class_logits = cls_score_p3[image_index, class_id].reshape(-1)
                pos_score = class_logits[positive].mean()
                box = boxes[gt_index]
                expansion = box.new_tensor(self.radius * stride)
                expanded = torch.stack((
                    box[0] - expansion, box[1] - expansion,
                    box[2] + expansion, box[3] + expansion))
                in_expanded = _inside_boxes(
                    points_p3, expanded.unsqueeze(0))
                in_object = _inside_boxes(points_p3, box.unsqueeze(0))
                negative = (in_expanded & ~in_object & (assigned < 0)
                            & ~inside_gt & ~ignored & valid)
                neg_logits = class_logits[negative]
                if neg_logits.numel() == 0:
                    continue
                hard_negative = neg_logits.topk(
                    min(self.topk, neg_logits.numel())).values.mean()
                total = total + F.softplus(
                    self.margin - (pos_score - hard_negative))
                count += 1
        normalizer = reduce_mean(total.new_tensor(float(count))).clamp_min(1)
        return total / normalizer

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> dict[str, Tensor]:
        losses = super().loss_by_feat(
            cls_scores, bbox_preds, centernesses, batch_gt_instances,
            batch_img_metas, batch_gt_instances_ignore)
        featmap_sizes = [score.shape[-2:] for score in cls_scores]
        points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        assignments = self._p3_assignments(points, batch_gt_instances)
        losses['loss_ltmr'] = self.local_tiny_margin_loss(
            cls_scores[0], points[0], assignments, batch_gt_instances,
            batch_img_metas, batch_gt_instances_ignore) * self.ltmr_weight
        return losses
