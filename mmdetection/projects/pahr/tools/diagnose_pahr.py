#!/usr/bin/env python3
"""Export PAHR maps and tiny-object center overlays for a small batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.apis import init_detector
from mmdet.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output-dir', default='work_dirs/pahr_diagnostics')
    parser.add_argument('--max-images', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def color_map(values: np.ndarray) -> np.ndarray:
    values = np.clip(values * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(values, cv2.COLORMAP_TURBO)


def main() -> None:
    args = parse_args()
    register_all_modules()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Config.fromfile(args.config)
    model = init_detector(config, args.checkpoint, device=args.device)
    loader = Runner.build_dataloader(config.val_dataloader)
    rows = []

    for index, data in enumerate(loader):
        if index >= args.max_images:
            break
        processed = model.data_preprocessor(data, training=False)
        inputs = processed['inputs']
        samples = processed['data_samples']
        with torch.inference_mode():
            _, aux = model.position_maps(inputs)
            targets = model.auxiliary_targets(
                aux['position_logits'], samples)
            losses = model.auxiliary_losses(aux, samples)

        probability = aux['position_logits'][0, 0].sigmoid()
        offsets = aux['offsets'][0]
        target, _, _, centers = targets
        maps = {
            'position': probability,
            'offset_x': offsets[0],
            'offset_y': offsets[1],
            'target': target[0, 0],
        }
        for name, tensor in maps.items():
            cv2.imwrite(
                str(output_dir / f'{index:04d}_{name}.png'),
                color_map(tensor.detach().float().cpu().numpy()))

        overlay = color_map(probability.detach().float().cpu().numpy())
        for y, x in centers[0, 0].nonzero().cpu().tolist():
            cv2.drawMarker(
                overlay, (x, y), (255, 255, 255),
                cv2.MARKER_CROSS, 5, 1)
        cv2.imwrite(str(output_dir / f'{index:04d}_overlay.png'), overlay)
        rows.append({
            'index': index,
            'loss_pos': float(losses['loss_pos']),
            'loss_offset': float(losses['loss_offset']),
            'position_min': float(probability.min()),
            'position_max': float(probability.max()),
        })

    summary = {
        'images': rows,
        'detail_output_weight_norm': float(
            model.neck.detail_mixer[-1].weight.detach().norm()),
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
