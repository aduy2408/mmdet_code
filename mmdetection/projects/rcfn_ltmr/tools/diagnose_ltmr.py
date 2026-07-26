#!/usr/bin/env python3
"""Export per-tiny-GT FCOS P3 local-margin diagnostics."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mmdet.models.dense_heads import FCOSHead
from mmdet.utils import register_all_modules
from projects.rcfn_ltmr.models.ltmr_fcos_head import (
    _inside_boxes, fcos_assigned_gt_inds)


def assignments_for_p3(head, points, gt_instances):
    counts = [len(item) for item in points]
    all_points = torch.cat(points)
    ranges = torch.cat([
        point.new_tensor(head.regress_ranges[level])[None].expand_as(point)
        for level, point in enumerate(points)
    ])
    return fcos_assigned_gt_inds(
        head, all_points, gt_instances, ranges, counts)[:counts[0]]


def size_bin(sqrt_area: float) -> str:
    return 'tiny-1' if sqrt_area <= 8 else 'tiny-2'


def rows_for_image(head, logits, points, gt_instances, img_meta,
                   radius, topk, score_threshold):
    assigned = assignments_for_p3(head, points, gt_instances)
    p3_points = points[0]
    boxes, labels = gt_instances.bboxes, gt_instances.labels
    inside_any = _inside_boxes(p3_points, boxes)
    img_h, img_w = img_meta['img_shape'][:2]
    valid = ((p3_points[:, 0] < img_w) & (p3_points[:, 1] < img_h))
    stride = head.strides[0]
    output = []

    for gt_index, box in enumerate(boxes):
        sqrt_area = float(
            ((box[2] - box[0]) * (box[3] - box[1])).clamp_min(0).sqrt())
        if sqrt_area > 16:
            continue
        positive = assigned == gt_index
        if not positive.any():
            continue
        class_logits = logits[labels[gt_index]].reshape(-1)
        pos_score = class_logits[positive].mean()
        expansion = box.new_tensor(radius * stride)
        expanded = torch.stack((
            box[0] - expansion, box[1] - expansion,
            box[2] + expansion, box[3] + expansion))
        ring = (_inside_boxes(p3_points, expanded[None])
                & ~_inside_boxes(p3_points, box[None]))
        negative = ring & (assigned < 0) & ~inside_any & valid
        neg_logits = class_logits[negative]
        if neg_logits.numel() == 0:
            continue
        hard = neg_logits.topk(min(topk, neg_logits.numel())).values
        neg_score = hard.mean()
        other_boxes = torch.cat((boxes[:gt_index], boxes[gt_index + 1:]))
        crowded = bool(((
            torch.minimum(other_boxes[:, 2:], expanded[None, 2:])
            - torch.maximum(other_boxes[:, :2], expanded[None, :2])
        ) > 0).all(dim=1).any()) if other_boxes.numel() else False
        sigmoid_score = pos_score.sigmoid()
        output.append({
            'img_id': img_meta.get('img_id', ''),
            'gt_index': gt_index,
            'size_bin': size_bin(sqrt_area),
            'sqrt_area': sqrt_area,
            'class_id': int(labels[gt_index]),
            'num_p3_positive': int(positive.sum()),
            's_pos': float(pos_score),
            'sigmoid_score': float(sigmoid_score),
            's_neg': float(neg_score),
            'margin': float(pos_score - neg_score),
            'local_bg_over_threshold': int(
                (neg_logits.sigmoid() >= score_threshold).sum()),
            'matched_but_low_confidence': bool(
                sigmoid_score < score_threshold),
            'crowded': crowded,
        })
    return output


def summarize(rows, split_crowding=True):
    result = {'count': len(rows)}
    if not rows:
        return result
    for key in ('s_pos', 's_neg', 'margin', 'sigmoid_score'):
        values = np.asarray([row[key] for row in rows])
        result[key] = {
            'median': float(np.median(values)),
            'q1': float(np.quantile(values, 0.25)),
        }
    scores = np.asarray([row['sigmoid_score'] for row in rows])
    result['score_threshold_rates'] = {
        str(threshold): float(np.mean(scores >= threshold))
        for threshold in (0.05, 0.1, 0.2)
    }
    result['matched_but_low_confidence_rate'] = float(np.mean(
        [row['matched_but_low_confidence'] for row in rows]))
    result['mean_local_bg_over_threshold'] = float(np.mean(
        [row['local_bg_over_threshold'] for row in rows]))
    if split_crowding:
        result['by_crowding'] = {
            str(crowded).lower(): summarize(
                [row for row in rows if row['crowded'] == crowded], False)
            for crowded in (False, True)
            if any(row['crowded'] == crowded for row in rows)
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--score-threshold', type=float, default=0.05)
    parser.add_argument('--max-images', type=int, default=0)
    args = parser.parse_args()

    register_all_modules()
    cfg = Config.fromfile(args.config)
    cfg.load_from = str(args.checkpoint)
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model
    model.eval()
    if not isinstance(model.bbox_head, FCOSHead):
        raise TypeError('diagnostic requires an FCOSHead-compatible model')

    rows = []
    with torch.inference_mode():
        for batch_index, data_batch in enumerate(runner.test_dataloader):
            if args.max_images and batch_index >= args.max_images:
                break
            data = model.data_preprocessor(data_batch, training=False)
            features = model.extract_feat(data['inputs'])
            cls_scores = model.bbox_head(features)[0]
            sizes = [score.shape[-2:] for score in cls_scores]
            points = model.bbox_head.prior_generator.grid_priors(
                sizes, dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            for index, sample in enumerate(data['data_samples']):
                rows.extend(rows_for_image(
                    model.bbox_head, cls_scores[0][index], points,
                    sample.gt_instances, sample.metainfo, args.radius,
                    args.topk, args.score_threshold))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'tiny_gt_margins.csv'
    if rows:
        with csv_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows)
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
