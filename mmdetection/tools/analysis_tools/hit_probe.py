#!/usr/bin/env python3
"""Fit HIT reconstructors on frozen P3 features and measure localization."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from PIL import Image, ImageDraw

from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output-dir', default='work_dirs/hit_probe')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--visualizations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default={})
    return parser.parse_args()


def load_baseline(model, checkpoint: str) -> tuple[list[str], list[str]]:
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = payload.get('state_dict', payload)
    remapped = {}
    for key, value in state.items():
        key = key.removeprefix('module.')
        if key.startswith('neck.') and not key.startswith('neck.base_neck.'):
            key = f'neck.base_neck.{key[len("neck."):]}'
        remapped[key] = value
    incompatible = model.load_state_dict(remapped, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        key for key in missing if not key.startswith('neck.hit_modules.')
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            'Baseline checkpoint is incompatible.\n'
            f'Non-HIT missing keys: {invalid_missing}\n'
            f'Unexpected keys: {unexpected}')
    print(f'Loaded baseline; {len(missing)} expected HIT keys are new.')
    return missing, unexpected


def build_loader(cfg: Config, split: str, seed: int):
    loader_cfg = cfg[f'{split}_dataloader']
    return Runner.build_dataloader(loader_cfg, seed=seed, diff_rank_seed=False)


def hit_module(model):
    modules = list(model.neck.hit_modules.values())
    if len(modules) != 1:
        raise RuntimeError('H probe requires exactly one HIT level.')
    return modules[0]


def prepare_model(cfg: Config, checkpoint: str, device: torch.device):
    model = MODELS.build(cfg.model)
    load_baseline(model, checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hit = hit_module(model)
    for block in (hit.spatial_reconstruct, hit.channel_reconstruct):
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    model.to(device).eval()
    hit.train()
    return model, hit


def fit_reconstructors(model, hit, loader, epochs: int, lr: float) -> None:
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    for epoch in range(epochs):
        total = steps = 0
        for batch in loader:
            data = model.data_preprocessor(batch, training=True)
            model.extract_feat(data['inputs'])
            losses = hit.auxiliary_losses(data['data_samples'])
            loss = (losses['loss_hit_recon_spatial'] +
                    losses['loss_hit_recon_channel'])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        print(f'epoch={epoch + 1} recon_loss={total / max(steps, 1):.6f}')


def box_mask(boxes: torch.Tensor, height: int, width: int, stride: int,
             margin: int) -> tuple[np.ndarray, list[np.ndarray]]:
    union = np.zeros((height, width), dtype=bool)
    per_box = []
    for box in boxes.cpu().numpy() / stride:
        x1, y1, x2, y2 = box
        left = max(math.floor(x1) - margin, 0)
        top = max(math.floor(y1) - margin, 0)
        right = min(math.ceil(x2) + margin + 1, width)
        bottom = min(math.ceil(y2) + margin + 1, height)
        current = np.zeros_like(union)
        current[top:bottom, left:right] = True
        union |= current
        per_box.append(current)
    return union, per_box


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = labels.size - positives
    if not positives or not negatives:
        return float('nan')
    _, inverse, counts = np.unique(
        scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average_ranks = (ends - counts + 1 + ends) / 2
    ranks = average_ranks[inverse]
    return float((ranks[labels].sum() -
                  positives * (positives + 1) / 2) /
                 (positives * negatives))


def random_hit_probability(population: int, support: int,
                           draws: int) -> float:
    if support <= 0:
        return 0.
    if draws >= population - support + 1:
        return 1.
    miss = 1.
    for index in range(draws):
        miss *= (population - support - index) / (population - index)
    return 1. - miss


def image_metrics(hard: np.ndarray, boxes: torch.Tensor, stride: int,
                  margin: int, topq: float) -> dict[str, float]:
    height, width = hard.shape
    union, per_box = box_mask(boxes, height, width, stride, margin)
    count = max(1, math.ceil(hard.size * topq))
    selected = np.argpartition(hard.ravel(), -count)[-count:]
    selected_mask = np.zeros(hard.size, dtype=bool)
    selected_mask[selected] = True
    selected_mask = selected_mask.reshape(hard.shape)

    recalls, random_recalls, distances = [], [], []
    selected_y, selected_x = np.nonzero(selected_mask)
    for box, support in zip(boxes.cpu().numpy() / stride, per_box):
        recalls.append(float((selected_mask & support).any()))
        random_recalls.append(
            random_hit_probability(hard.size, int(support.sum()), count))
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        distances.append(float(np.sqrt(
            (selected_x + .5 - cx)**2 + (selected_y + .5 - cy)**2).min()))
    return {
        'auroc': auroc(hard.ravel(), union.ravel()),
        'top_precision': float(union.ravel()[selected].mean()),
        'random_precision': float(union.mean()),
        'gt_recall': float(np.mean(recalls)) if recalls else float('nan'),
        'random_gt_recall': (
            float(np.mean(random_recalls)) if random_recalls else float('nan')),
        'center_distance_p3': (
            float(np.mean(distances)) if distances else float('nan')),
    }


def save_visualization(path: Path, hard: np.ndarray, boxes: torch.Tensor,
                       stride: int) -> None:
    normalized = hard - hard.min()
    normalized /= max(float(normalized.max()), 1e-12)
    rgb = np.zeros((*hard.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (normalized * 255).astype(np.uint8)
    rgb[..., 1] = (np.sqrt(normalized) * 180).astype(np.uint8)
    image = Image.fromarray(rgb).resize(
        (hard.shape[1] * stride, hard.shape[0] * stride), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    for box in boxes.cpu().tolist():
        draw.rectangle(box, outline=(0, 255, 0), width=2)
    image.save(path)


def confidence_interval(values: list[float], rng: np.random.Generator,
                        samples: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {'mean': float('nan'), 'low': float('nan'),
                'high': float('nan')}
    boot = np.empty(samples)
    for index in range(samples):
        boot[index] = rng.choice(array, size=array.size).mean()
    return {
        'mean': float(array.mean()),
        'low': float(np.quantile(boot, .025)),
        'high': float(np.quantile(boot, .975)),
    }


@torch.no_grad()
def evaluate(model, hit, loader, output_dir: Path, bootstrap_samples: int,
             visualizations: int, seed: int) -> dict:
    per_image: list[dict[str, float]] = []
    visualized = 0
    for batch in loader:
        data = model.data_preprocessor(batch, training=False)
        model.extract_feat(data['inputs'])
        hard_batch = hit.last_aux['hard_raw'][:, 0].detach().cpu().numpy()
        for hard, sample in zip(hard_batch, data['data_samples']):
            boxes = get_box_tensor(sample.gt_instances.bboxes).detach().cpu()
            per_image.append(image_metrics(
                hard, boxes, hit.stride, hit.offset_target_margin,
                hit.source_topq))
            if visualized < visualizations:
                save_visualization(
                    output_dir / f'h_{visualized:04d}.png', hard, boxes,
                    hit.stride)
                visualized += 1

    rng = np.random.default_rng(seed)
    names = per_image[0].keys() if per_image else []
    summary = {
        name: confidence_interval(
            [item[name] for item in per_image], rng, bootstrap_samples)
        for name in names
    }
    precision_delta = [
        item['top_precision'] - item['random_precision']
        for item in per_image
    ]
    recall_delta = [
        item['gt_recall'] - item['random_gt_recall']
        for item in per_image
    ]
    summary['top_precision_delta'] = confidence_interval(
        precision_delta, rng, bootstrap_samples)
    summary['gt_recall_delta'] = confidence_interval(
        recall_delta, rng, bootstrap_samples)
    passed = (
        summary['auroc']['low'] > .5 and
        (summary['top_precision_delta']['low'] > 0 or
         summary['gt_recall_delta']['low'] > 0))
    return {
        'gate_passed': passed,
        'gate': ('auroc.low > 0.5 and '
                 '(top_precision_delta.low > 0 or gt_recall_delta.low > 0)'),
        'images': len(per_image),
        'metrics': summary,
        'per_image': per_image,
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.bootstrap_samples < 1:
        raise ValueError('epochs and bootstrap-samples must be positive.')
    register_all_modules(init_default_scope=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    os.chdir(Path(__file__).resolve().parents[2])
    cfg = Config.fromfile(config_path)
    cfg.merge_from_dict(args.cfg_options)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, hit = prepare_model(cfg, str(checkpoint_path), device)
    fit_reconstructors(
        model, hit, build_loader(cfg, 'train', args.seed), args.epochs,
        args.lr)
    torch.save({
        'spatial_reconstruct': hit.spatial_reconstruct.state_dict(),
        'channel_reconstruct': hit.channel_reconstruct.state_dict(),
    }, output_dir / 'reconstructors.pth')
    report = evaluate(
        model, hit, build_loader(cfg, 'val', args.seed), output_dir,
        args.bootstrap_samples, args.visualizations, args.seed)
    report.update(
        checkpoint=str(checkpoint_path),
        config=str(config_path),
        seed=args.seed,
    )
    (output_dir / 'metrics.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))
    print('PASS: continue to sparse transport' if report['gate_passed']
          else 'STOP: H did not pass the preregistered gate')


if __name__ == '__main__':
    main()
