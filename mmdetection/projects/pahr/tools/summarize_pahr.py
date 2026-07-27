#!/usr/bin/env python3
"""Summarize PAHR COCO metrics, tiny localization, and P3 correction."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.apis import init_detector
from mmdet.utils import register_all_modules
from projects.pahr import PAHRFPN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tiny-max-sqrt-area', type=float, default=16.0)
    return parser.parse_args()


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return box.new_zeros(0)
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=1)
    box_area = (box[2:] - box[:2]).clamp_min(0).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
    return intersection / (box_area + areas - intersection).clamp_min(1e-7)


def load_coco_metrics(prediction_path: Path) -> dict:
    candidates = sorted(prediction_path.parent.glob('*/*.json'))
    for candidate in reversed(candidates):
        payload = json.loads(candidate.read_text(encoding='utf-8'))
        if 'coco/bbox_mAP' in payload:
            return payload
    return {}


def tiny_statistics(annotation_path: Path, prediction_path: Path,
                    tiny_limit: float) -> dict:
    annotations = json.loads(annotation_path.read_text(encoding='utf-8'))
    tiny_by_image = defaultdict(list)
    for annotation in annotations['annotations']:
        if float(annotation['area'])**0.5 <= tiny_limit:
            x, y, width, height = annotation['bbox']
            tiny_by_image[annotation['image_id']].append(
                [x, y, x + width, y + height])

    with prediction_path.open('rb') as stream:
        predictions = pickle.load(stream)
    matched_50 = matched_75 = 0
    center_errors = []
    tiny_count = sum(map(len, tiny_by_image.values()))
    for prediction in predictions:
        gt_boxes = tiny_by_image.get(prediction['img_id'], [])
        if not gt_boxes:
            continue
        instances = prediction['pred_instances']
        boxes = torch.as_tensor(instances['bboxes']).float()
        labels = torch.as_tensor(instances['labels'])
        boxes = boxes[labels == 0]
        for values in gt_boxes:
            gt = boxes.new_tensor(values)
            ious = box_iou(gt, boxes)
            best_iou, best_index = (
                ious.max(dim=0) if ious.numel() else
                (gt.new_tensor(0), gt.new_tensor(0, dtype=torch.long)))
            matched_50 += int(best_iou >= 0.5)
            matched_75 += int(best_iou >= 0.75)
            if best_iou >= 0.5:
                predicted = boxes[best_index]
                gt_center = (gt[:2] + gt[2:]) * 0.5
                predicted_center = (
                    predicted[:2] + predicted[2:]) * 0.5
                center_errors.append(
                    float(torch.linalg.vector_norm(
                        predicted_center - gt_center)))
    errors = torch.tensor(center_errors)
    statistics = {
        'tiny_gt': tiny_count,
        'tiny_recall_50': matched_50 / max(tiny_count, 1),
        'tiny_recall_75': matched_75 / max(tiny_count, 1),
        'tiny_center_error_mean_px': (
            float(errors.mean()) if errors.numel() else None),
        'tiny_center_error_median_px': (
            float(errors.median()) if errors.numel() else None),
    }


def correction_statistics(config: Config, checkpoint: str,
                          device: str) -> dict:
    model = init_detector(config, checkpoint, device=device)
    loader = Runner.build_dataloader(config.val_dataloader)
    data = next(iter(loader))
    batch = model.data_preprocessor(data, training=False)
    with torch.inference_mode():
        backbone = model.backbone(batch['inputs'])
        baseline = super(PAHRFPN, model.neck).forward(backbone)
        enhanced, aux = model.neck.forward_with_aux(backbone)
        correction = enhanced[0] - baseline[0]
        position = aux['position_logits'].sigmoid()
        detail_output = model.neck.detail_mixer[-1]
        _, _, _, centers = model.auxiliary_targets(
            aux['position_logits'], batch['data_samples'])
        correction_energy = correction.abs().mean(dim=1, keepdim=True)
        center_energy = (
            correction_energy[centers].mean()
            if centers.any() else correction_energy.new_zeros(()))
        background_energy = (
            correction_energy[~centers].mean()
            if (~centers).any() else correction_energy.new_zeros(()))
    return {
        'p3_correction_ratio': float(
            correction.norm() / baseline[0].norm().clamp_min(1e-12)),
        'p3_correction_abs_mean': float(correction.abs().mean()),
        'position_mean': float(position.mean()),
        'position_max': float(position.max()),
        'correction_gate_mean': float(aux['correction_gate'].mean()),
        'correction_gate_max': float(aux['correction_gate'].max()),
        'phase_gate_mean': float(aux['phase_gate'].mean()),
        'phase_gate_max': float(aux['phase_gate'].max()),
        'guidance_rms': float(aux['guidance_rms']),
        'raw_correction_rms': float(aux['raw_correction_rms']),
        'applied_correction_rms': float(aux['applied_correction_rms']),
        'tiny_center_correction_abs_mean': float(center_energy),
        'background_correction_abs_mean': float(background_energy),
        'tiny_correction_concentration': float(
            center_energy / background_energy.clamp_min(1e-12)),
        'detail_output_weight_norm': float(
            detail_output.weight.detach().norm()),
    }
    if 'measurement_phase_logits' in aux:
        phase_probability = aux['measurement_phase_logits'].softmax(dim=1)
        targets = model.measurement_targets(aux, batch['data_samples'])
        _, _, _, size_target, size_valid = targets
        predicted_sizes = aux['measurement_log_sizes'].exp() * 2
        target_sizes = size_target.exp() * 2
        size_mask = size_valid.expand_as(predicted_sizes)
        statistics.update(
            measurement_center_mean=float(
                aux['measurement_center_logits'].sigmoid().mean()),
            measurement_center_max=float(
                aux['measurement_center_logits'].sigmoid().max()),
            measurement_phase_entropy=float(
                -(phase_probability * phase_probability.clamp_min(
                    1e-12).log()).sum(dim=1).mean()),
            measurement_size_mae_px=(
                float((predicted_sizes[size_mask]
                       - target_sizes[size_mask]).abs().mean())
                if size_mask.any() else None),
            measurement_refine_scale=float(
                model.measurement_refine_scale.tanh()))
    return statistics


def main() -> None:
    args = parse_args()
    register_all_modules()
    config = Config.fromfile(args.config)
    prediction_path = Path(args.predictions)
    annotation_path = Path(config.test_dataloader.dataset.ann_file)
    summary = {
        'variant': dict(config.get('pahr_variant', {})),
        'coco': load_coco_metrics(prediction_path),
        'tiny': tiny_statistics(
            annotation_path, prediction_path, args.tiny_max_sqrt_area),
        'correction': correction_statistics(
            config, args.checkpoint, args.device),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
