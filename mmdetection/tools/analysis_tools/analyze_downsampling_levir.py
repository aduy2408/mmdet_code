#!/usr/bin/env python3
"""Measure tiny-object feature loss across ResNet downsampling transitions."""

import argparse
import csv
import json
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('annotations')
    parser.add_argument('image_root')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-images', type=int, default=100)
    parser.add_argument('--output', default='downsampling_analysis.csv')
    return parser.parse_args()


def phase_deviation(feature, x, y):
    """Mean channel L2 deviation among the 2x2 phase samples."""
    height, width = feature.shape[-2:]
    x = min(max(int(x) // 2 * 2, 0), width - 2)
    y = min(max(int(y) // 2 * 2, 0), height - 2)
    phases = feature[:, y:y + 2, x:x + 2].reshape(feature.shape[0], 4)
    mean = phases.mean(dim=1, keepdim=True)
    return float(torch.linalg.vector_norm(phases - mean, dim=0).mean())


def local_separability(feature, x, y, radius=4):
    """Distance from the GT-center feature to nearby background features."""
    height, width = feature.shape[-2:]
    x, y = int(round(x)), int(round(y))
    x, y = min(max(x, 0), width - 1), min(max(y, 0), height - 1)
    center = feature[:, y, x]
    samples = []
    for dx, dy in ((-radius, 0), (radius, 0), (0, -radius), (0, radius)):
        bx, by = x + dx, y + dy
        if 0 <= bx < width and 0 <= by < height:
            samples.append(feature[:, by, bx])
    if not samples:
        return float('nan')
    background = torch.stack(samples).mean(dim=0)
    return float(torch.linalg.vector_norm(center - background))


def detection_score(result, box):
    predictions = result.pred_instances
    boxes = predictions.bboxes.detach().cpu().numpy()
    scores = predictions.scores.detach().cpu().numpy()
    if not len(boxes):
        return 0.0
    center = (box[:2] + box[2:]) / 2
    pred_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    half_size = np.maximum((box[2:] - box[:2]) * 0.75, 2)
    candidates = np.all(np.abs(pred_centers - center) <= half_size, axis=1)
    return float(scores[candidates].max()) if candidates.any() else 0.0


def main():
    args = parse_args()
    payload = json.loads(Path(args.annotations).read_text())
    annotations = {}
    for annotation in payload['annotations']:
        annotations.setdefault(annotation['image_id'], []).append(annotation)

    model = init_detector(args.config, args.checkpoint, device=args.device)
    captured = {}
    handles = []
    # layer2 and layer3 are C2->C3 and C3->C4.
    for name in ('layer2', 'layer3'):
        module = getattr(model.backbone, name)

        def hook(_, inputs, output, stage=name):
            captured[stage] = (
                inputs[0][0].detach().float().cpu(),
                output[0].detach().float().cpu())

        handles.append(module.register_forward_hook(hook))

    rows = []
    try:
        images = [image for image in payload['images']
                  if image['id'] in annotations][:args.max_images]
        for image in images:
            image_path = Path(args.image_root) / image['file_name']
            result = inference_detector(model, mmcv.imread(str(image_path)))
            scale = result.metainfo.get('scale_factor', (1.0, 1.0))
            scale_x, scale_y = float(scale[0]), float(scale[1])
            for annotation in annotations[image['id']]:
                x, y, width, height = map(float, annotation['bbox'])
                box = np.array([x, y, x + width, y + height])
                center_x = (x + width / 2) * scale_x
                center_y = (y + height / 2) * scale_y
                row = {
                    'image_id': image['id'],
                    'annotation_id': annotation['id'],
                    'width': width,
                    'height': height,
                    'area': width * height,
                    'detection_score': detection_score(result, box),
                }
                for stage, (before, after) in captured.items():
                    # ResNet feature strides before layer2/layer3 are 4/8.
                    pre_stride = 4 if stage == 'layer2' else 8
                    px, py = center_x / pre_stride, center_y / pre_stride
                    pre_sep = local_separability(before, px, py)
                    post_sep = local_separability(after, px / 2, py / 2)
                    row[f'{stage}_pre_deviation'] = phase_deviation(
                        before, px, py)
                    row[f'{stage}_pre_separability'] = pre_sep
                    row[f'{stage}_post_separability'] = post_sep
                    row[f'{stage}_separability_ratio'] = post_sep / max(
                        pre_sep, 1e-12)
                rows.append(row)
    finally:
        for handle in handles:
            handle.remove()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} object measurements to {output}')


if __name__ == '__main__':
    main()
