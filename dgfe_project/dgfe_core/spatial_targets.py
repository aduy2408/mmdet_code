from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor


def slice_from_box(start: float, end: float, limit: int) -> tuple[int, int]:
    start_i = max(0, min(limit - 1, int(math.floor(start))))
    end_i = max(start_i + 1, min(limit, int(math.ceil(end))))
    return start_i, end_i


def max_iou_per_box(boxes: Tensor, gt_boxes: Tensor) -> tuple[Tensor, Tensor]:
    if boxes.numel() == 0 or gt_boxes.numel() == 0:
        empty = boxes.new_zeros((boxes.shape[0], ))
        return empty, empty.to(dtype=torch.long)
    lt = torch.maximum(boxes[:, None, :2], gt_boxes[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], gt_boxes[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) *
              (boxes[:, 3] - boxes[:, 1]).clamp(min=0))
    area_b = ((gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) *
              (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0))
    ious = inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-6)
    return ious.max(dim=1)


def aligned_iou(boxes: Tensor, targets: Tensor) -> Tensor:
    """Return IoU for aligned pairs of xyxy boxes."""
    if boxes.shape != targets.shape or boxes.shape[-1] != 4:
        raise ValueError('aligned_iou expects matching [N, 4] tensors.')
    lt = torch.maximum(boxes[:, :2], targets[:, :2])
    rb = torch.minimum(boxes[:, 2:], targets[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) *
              (boxes[:, 3] - boxes[:, 1]).clamp(min=0))
    area_b = ((targets[:, 2] - targets[:, 0]).clamp(min=0) *
              (targets[:, 3] - targets[:, 1]).clamp(min=0))
    return inter / (area_a + area_b - inter).clamp(min=1e-6)


def build_quality_spatial_target(detector, logits: Tensor, batch_inputs: Tensor,
                                 batch_data_samples, records: Iterable[dict]) -> Tensor | None:
    records = list(records)
    if not records:
        return None
    target = torch.zeros_like(logits)
    _, _, img_h, img_w = batch_inputs.shape
    feat_h, feat_w = logits.shape[-2:]
    ring = max(float(getattr(detector.neck, 'dgfe_boundary_ring', 1.0)), 0.0)
    inner_value = max(
        min(float(getattr(detector.neck, 'dgfe_inner_value', 0.3)), 1.0), 0.0)
    tiny_area = float(getattr(detector.neck, 'dgfe_tiny_area', 4.0))
    edge_norm = max(float(getattr(detector.neck, 'dgfe_edge_error_norm',
                                  0.25)), 1e-9)

    by_gt: dict[tuple[int, int], list[float]] = {}
    for record in records:
        batch_idx = int(record['batch_idx'])
        gt_idx = int(record['gt_idx'])
        quality = float(record.get('quality', 0.0))
        by_gt.setdefault((batch_idx, gt_idx), []).append(quality)

    for (batch_idx, gt_idx), qualities in by_gt.items():
        if batch_idx >= len(batch_data_samples):
            continue
        bboxes = batch_data_samples[batch_idx].gt_instances.bboxes
        bboxes = bboxes.tensor if hasattr(bboxes, 'tensor') else bboxes
        if gt_idx >= bboxes.shape[0]:
            continue
        quality = max(0.0, min(1.0, max(qualities)))
        edge_value = max(0.0, min(1.0, 1.0 - quality / edge_norm))
        x1, y1, x2, y2 = [float(v) for v in bboxes[gt_idx]]
        fx1, fx2 = x1 * feat_w / img_w, x2 * feat_w / img_w
        fy1, fy2 = y1 * feat_h / img_h, y2 * feat_h / img_h
        if fx2 <= fx1 or fy2 <= fy1:
            continue
        ix1, ix2 = slice_from_box(fx1, fx2, feat_w)
        iy1, iy2 = slice_from_box(fy1, fy2, feat_h)
        if (ix2 - ix1) * (iy2 - iy1) <= tiny_area:
            region = target[batch_idx, :, iy1:iy2, ix1:ix2]
            region.copy_(torch.maximum(region, torch.full_like(region, edge_value)))
            continue
        region = target[batch_idx, :, iy1:iy2, ix1:ix2]
        region.copy_(torch.maximum(region, torch.full_like(region, inner_value)))
        edge = max(int(math.ceil(ring)), 1)
        regions = (
            target[batch_idx, :, iy1:min(iy1 + edge, iy2), ix1:ix2],
            target[batch_idx, :, max(iy2 - edge, iy1):iy2, ix1:ix2],
            target[batch_idx, :, iy1:iy2, ix1:min(ix1 + edge, ix2)],
            target[batch_idx, :, iy1:iy2, max(ix2 - edge, ix1):ix2],
        )
        for region in regions:
            if region.numel():
                region.copy_(torch.maximum(region, torch.full_like(region, edge_value)))
    return target
