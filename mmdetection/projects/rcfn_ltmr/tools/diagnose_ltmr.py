#!/usr/bin/env python3
"""Export per-tiny-GT FCOS P3 local-margin diagnostics."""

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from PIL import Image

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


def position_statistics(model, position, samples):
    target, valid = model.position_targets(position, samples)
    centers = []
    for index, sample in enumerate(samples):
        boxes = sample.gt_instances.bboxes
        boxes = boxes.tensor if hasattr(boxes, 'tensor') else boxes
        tiny = boxes[model.tiny_mask(boxes, sample.metainfo)]
        for box in tiny:
            x = int(((box[0] + box[2]) / (2 * model.position_stride))
                    .round().clamp(0, position.shape[3] - 1))
            y = int(((box[1] + box[3]) / (2 * model.position_stride))
                    .round().clamp(0, position.shape[2] - 1))
            centers.append(float(position[index, 0, y, x]))
    background = valid & (target < 0.1)
    false_count = int(((position >= 0.5) & background).sum())
    return target, {
        'center_values': centers,
        'background_false_count': false_count,
        'background_count': int(background.sum()),
    }


def save_position_overlay(position, target, output_path):
    predicted = (position.detach().cpu().numpy() * 255).astype(np.uint8)
    expected = (target.detach().cpu().numpy() * 255).astype(np.uint8)
    image = np.zeros((*predicted.shape, 3), dtype=np.uint8)
    image[..., 0] = expected
    image[..., 1] = predicted
    Image.fromarray(image).resize(
        (predicted.shape[1] * 8, predicted.shape[0] * 8),
        resample=Image.Resampling.NEAREST).save(output_path)


def prediction_index(path):
    """Load dumped MMDetection predictions keyed by image id."""
    with Path(path).open('rb') as handle:
        predictions = pickle.load(handle)
    return {int(item['img_id']): item['pred_instances']
            for item in predictions}


def boxes_to_original(boxes, metainfo):
    """Map transformed GT boxes back to original-image coordinates."""
    scale_factor = metainfo.get('scale_factor')
    if scale_factor is None:
        img_shape = metainfo.get('img_shape')
        ori_shape = metainfo.get('ori_shape')
        scale_factor = (
            (img_shape[1] / ori_shape[1], img_shape[0] / ori_shape[0])
            if img_shape is not None and ori_shape is not None else (1.0, 1.0))
    scale = boxes.new_tensor(scale_factor).flatten()[:2]
    if scale.numel() != 2 or bool((scale <= 0).any()):
        raise ValueError(
            'scale_factor must contain positive (width, height) scales')
    return boxes / scale.repeat(2)


def match_predictions(
        gt_boxes, gt_labels, prediction, score_threshold=0.05,
        iou_threshold=0.5):
    """Greedily match score-sorted predictions to GT boxes."""
    matched = torch.zeros(
        len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)
    used = set()
    false_positives = 0
    scores = prediction['scores']
    order = scores.argsort(descending=True)
    pred_boxes = prediction['bboxes'].to(gt_boxes)
    pred_labels = prediction['labels'].to(gt_labels.device)
    for pred_index in order.tolist():
        if float(scores[pred_index]) < score_threshold:
            continue
        box = pred_boxes[pred_index]
        candidates = (
            (gt_labels == pred_labels[pred_index]).nonzero().flatten().tolist())
        candidates = [index for index in candidates if index not in used]
        best_iou, best_index = 0.0, -1
        for gt_index in candidates:
            gt = gt_boxes[gt_index]
            top_left = torch.maximum(box[:2], gt[:2])
            bottom_right = torch.minimum(box[2:], gt[2:])
            intersection = (
                bottom_right - top_left).clamp_min(0).prod()
            union = (
                (box[2:] - box[:2]).clamp_min(0).prod()
                + (gt[2:] - gt[:2]).clamp_min(0).prod()
                - intersection)
            overlap = float(intersection / union.clamp_min(1e-9))
            if overlap > best_iou:
                best_iou, best_index = overlap, gt_index
        if best_iou >= iou_threshold:
            used.add(best_index)
            matched[best_index] = True
        else:
            false_positives += 1
    return matched, false_positives


def map_center_and_max3(feature_map, center_x, center_y):
    """Sample a map at the rounded center and in its clipped 3x3 window."""
    height, width = feature_map.shape[-2:]
    x = int(round(float(center_x)))
    y = int(round(float(center_y)))
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    patch = feature_map[
        max(0, y - 1):min(height, y + 2),
        max(0, x - 1):min(width, x + 2)]
    return float(feature_map[y, x]), float(patch.max())


def paired_gate_rows(
        model, sample, position, contrast, reference_prediction,
        candidate_prediction, score_threshold=0.05, iou_threshold=0.5):
    """Return per-tiny-GT gate rows and prediction false positives."""
    boxes = sample.gt_instances.bboxes
    boxes = boxes.tensor if hasattr(boxes, 'tensor') else boxes
    labels = sample.gt_instances.labels
    original_boxes = boxes_to_original(boxes, sample.metainfo)
    reference_matches, reference_fp = match_predictions(
        original_boxes, labels, reference_prediction,
        score_threshold, iou_threshold)
    candidate_matches, candidate_fp = match_predictions(
        original_boxes, labels, candidate_prediction,
        score_threshold, iou_threshold)
    tiny = model.tiny_mask(boxes, sample.metainfo)
    rows = []
    for gt_index in tiny.nonzero().flatten().tolist():
        reference_hit = bool(reference_matches[gt_index])
        candidate_hit = bool(candidate_matches[gt_index])
        if reference_hit and candidate_hit:
            group = 'retained'
        elif reference_hit:
            group = 'lost'
        elif candidate_hit:
            group = 'gained'
        else:
            group = 'r2_miss'
        box = boxes[gt_index]
        center_x = (box[0] + box[2]) / (2 * model.position_stride)
        center_y = (box[1] + box[3]) / (2 * model.position_stride)
        h_center, h_max3 = map_center_and_max3(
            position, center_x, center_y)
        row = {
            'img_id': sample.metainfo.get('img_id', ''),
            'gt_index': gt_index,
            'group': group,
            'h_center': h_center,
            'h_max3': h_max3,
            'c_center': None,
            'c_max3': None,
            'hc_center': None,
            'hc_max3': None,
            'gate_center': 1.0,
            'gate_max3': 1.0,
        }
        if contrast is not None:
            c_center, c_max3 = map_center_and_max3(
                contrast, center_x, center_y)
            combined = position * contrast
            hc_center, hc_max3 = map_center_and_max3(
                combined, center_x, center_y)
            final_gate = model.neck.floor_gate(combined)
            gate_center, gate_max3 = map_center_and_max3(
                final_gate, center_x, center_y)
            row.update(
                c_center=c_center, c_max3=c_max3,
                hc_center=hc_center, hc_max3=hc_max3,
                gate_center=gate_center, gate_max3=gate_max3)
        rows.append(row)
    return rows, reference_fp, candidate_fp


def summarize_paired_gates(
        rows, reference_fp, candidate_fp, reference_map=None,
        candidate_map=None, max_map_drop=0.005):
    """Aggregate paired groups and evaluate the decision criteria."""
    fields = (
        'h_center', 'h_max3', 'c_center', 'c_max3', 'hc_center',
        'hc_max3', 'gate_center', 'gate_max3')
    summary = {
        'count': len(rows),
        'groups': dict(Counter(row['group'] for row in rows)),
        'reference_false_positives': reference_fp,
        'candidate_false_positives': candidate_fp,
        'by_group': {},
    }
    for group in ('retained', 'lost', 'r2_miss', 'gained'):
        selected = [row for row in rows if row['group'] == group]
        metrics = {'count': len(selected)}
        for field in fields:
            values = [
                row[field] for row in selected if row[field] is not None]
            if values:
                metrics[field] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                }
        summary['by_group'][group] = metrics
    map_ok = None
    if reference_map is not None and candidate_map is not None:
        map_ok = candidate_map >= reference_map - max_map_drop
    checks = {
        'tiny_lost_le_3': summary['groups'].get('lost', 0) <= 3,
        'candidate_fp_below_r2': candidate_fp < reference_fp,
        'map_not_materially_lower': map_ok,
    }
    available = [value for value in checks.values() if value is not None]
    summary['acceptance'] = {
        **checks,
        'accepted': all(available) if len(available) == 3 else None,
        'reference_map': reference_map,
        'candidate_map': candidate_map,
        'max_map_drop': max_map_drop,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--score-threshold', type=float, default=0.05)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument(
        '--variant',
        choices=(
            'baseline', 'r2', 'pg_aux', 'pg_aux_w01', 'pg_h', 'pg_ch',
            'pg_ch_w01_floor', 'l1'),
        default='baseline')
    parser.add_argument('--max-position-maps', type=int, default=16)
    parser.add_argument('--reference-predictions', type=Path)
    parser.add_argument('--candidate-predictions', type=Path)
    parser.add_argument('--reference-test-metrics', type=Path)
    parser.add_argument('--candidate-test-metrics', type=Path)
    parser.add_argument('--match-iou-threshold', type=float, default=0.5)
    parser.add_argument('--max-map-drop', type=float, default=0.005)
    args = parser.parse_args()
    if bool(args.reference_predictions) != bool(args.candidate_predictions):
        parser.error(
            '--reference-predictions and --candidate-predictions '
            'must be provided together')
    if bool(args.reference_test_metrics) != bool(args.candidate_test_metrics):
        parser.error(
            '--reference-test-metrics and --candidate-test-metrics '
            'must be provided together')

    register_all_modules()
    cfg = Config.fromfile(args.config)
    if args.variant == 'r2':
        cfg.model.neck.update(
            type='RCFNFPN', eps=1e-4, gamma_init=0.0)
    elif args.variant == 'l1':
        cfg.model.bbox_head.update(
            type='LTMRFCOSHead',
            tiny_max_sqrt_area=16.0,
            radius=args.radius,
            topk=args.topk,
            margin=1.0,
            loss_weight=0.05)
    cfg.work_dir = str(args.output_dir / 'runner')
    cfg.load_from = str(args.checkpoint)
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model
    diagnostic_model = model.module if hasattr(model, 'module') else model
    model.eval()
    if not isinstance(diagnostic_model.bbox_head, FCOSHead):
        raise TypeError('diagnostic requires an FCOSHead-compatible model')

    rows = []
    position_stats = {
        'center_values': [], 'background_false_count': 0,
        'background_count': 0,
    }
    map_count = 0
    map_dir = args.output_dir / 'position_maps'
    paired_rows = []
    reference_fp = 0
    candidate_fp = 0
    reference_predictions = (
        prediction_index(args.reference_predictions)
        if args.reference_predictions else None)
    candidate_predictions = (
        prediction_index(args.candidate_predictions)
        if args.candidate_predictions else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch_index, data_batch in enumerate(runner.test_dataloader):
            if args.max_images and batch_index >= args.max_images:
                break
            data = model.data_preprocessor(data_batch, training=False)
            if hasattr(diagnostic_model, 'position_maps'):
                features, position, contrast = diagnostic_model.position_maps(
                    data['inputs'])
                target, stats = position_statistics(
                    diagnostic_model, position, data['data_samples'])
                position_stats['center_values'].extend(
                    stats['center_values'])
                position_stats['background_false_count'] += (
                    stats['background_false_count'])
                position_stats['background_count'] += stats['background_count']
                for index in range(position.shape[0]):
                    if map_count >= args.max_position_maps:
                        break
                    map_dir.mkdir(parents=True, exist_ok=True)
                    save_position_overlay(
                        position[index, 0], target[index, 0],
                        map_dir / f'{map_count:04d}.png')
                    map_count += 1
                if reference_predictions is not None:
                    for index, sample in enumerate(data['data_samples']):
                        img_id = int(sample.metainfo['img_id'])
                        if (img_id not in reference_predictions
                                or img_id not in candidate_predictions):
                            raise KeyError(
                                f'Missing paired prediction for img_id={img_id}')
                        image_rows, ref_count, candidate_count = (
                            paired_gate_rows(
                                diagnostic_model, sample,
                                position[index, 0],
                                contrast[index, 0]
                                if contrast is not None else None,
                                reference_predictions[img_id],
                                candidate_predictions[img_id],
                                args.score_threshold,
                                args.match_iou_threshold))
                        paired_rows.extend(image_rows)
                        reference_fp += ref_count
                        candidate_fp += candidate_count
            else:
                features = model.extract_feat(data['inputs'])
            cls_scores = diagnostic_model.bbox_head(features)[0]
            sizes = [score.shape[-2:] for score in cls_scores]
            points = diagnostic_model.bbox_head.prior_generator.grid_priors(
                sizes, dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            for index, sample in enumerate(data['data_samples']):
                rows.extend(rows_for_image(
                    diagnostic_model.bbox_head, cls_scores[0][index], points,
                    sample.gt_instances, sample.metainfo, args.radius,
                    args.topk, args.score_threshold))

    csv_path = args.output_dir / 'tiny_gt_margins.csv'
    if rows:
        with csv_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows)
    if hasattr(diagnostic_model, 'position_maps'):
        values = np.asarray(position_stats['center_values'])
        summary['position'] = {
            'tiny_center_count': len(values),
            'mean_at_tiny_centers': (
                float(values.mean()) if len(values) else None),
            'center_recall_at_0.5': (
                float(np.mean(values >= 0.5)) if len(values) else None),
            'background_false_activation_at_0.5': (
                position_stats['background_false_count']
                / max(position_stats['background_count'], 1)),
            'background_cell_count': position_stats['background_count'],
            'saved_maps': map_count,
            'overlay_channels': {'red': 'target', 'green': 'prediction'},
        }
    if reference_predictions is not None:
        paired_csv = args.output_dir / 'paired_gate_gt.csv'
        if paired_rows:
            with paired_csv.open(
                    'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=paired_rows[0].keys())
                writer.writeheader()
                writer.writerows(paired_rows)

        def read_map(path):
            if path is None:
                return None
            return float(json.loads(
                path.read_text(encoding='utf-8'))['coco/bbox_mAP'])

        paired_summary = summarize_paired_gates(
            paired_rows, reference_fp, candidate_fp,
            read_map(args.reference_test_metrics),
            read_map(args.candidate_test_metrics),
            args.max_map_drop)
        (args.output_dir / 'paired_gate_summary.json').write_text(
            json.dumps(paired_summary, indent=2), encoding='utf-8')
        summary['paired_gate'] = paired_summary
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
