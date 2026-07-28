#!/usr/bin/env python3
"""Train the FCOS DBSS ablations on LEVIR-Ship."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import train_all_levir_baseline as levir


VARIANTS = {
    'baseline': 'configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py',
    'ridge': 'projects/dbss/configs/fcos_dbss_ridge.py',
    'softmax': 'projects/dbss/configs/fcos_dbss_softmax.py',
    'ridge_haar': 'projects/dbss/configs/fcos_dbss_ridge_haar.py',
}


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def set_resize_scale(obj: Any, image_size: int) -> None:
    if isinstance(obj, dict):
        if obj.get('type') == 'Resize':
            obj['scale'] = (image_size, image_size)
        for value in obj.values():
            set_resize_scale(value, image_size)
    elif isinstance(obj, list):
        for value in obj:
            set_resize_scale(value, image_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='LevirShipData')
    parser.add_argument(
        '--dataset-out', default='mmdetection/data/levir_ship_coco')
    parser.add_argument(
        '--work-dir', default='mmdetection/work_dirs/levir_dbss')
    parser.add_argument(
        '--variants', default=','.join(VARIANTS),
        help=f"Comma-separated: {', '.join(VARIANTS)}")
    parser.add_argument('--image-size', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--num-machines', type=int, default=1)
    parser.add_argument('--machine-index', type=int, default=0)
    return parser.parse_args()


def write_config(
        variant: str,
        args: argparse.Namespace,
        dataset_out: Path,
        image_dir: Path) -> Path:
    root = str(levir.mmdet_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from mmengine.config import Config
    from mmdet.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(str(levir.mmdet_root() / VARIANTS[variant]))
    cfg = levir.patch_config(cfg, variant, args, dataset_out, image_dir)
    for dataloader in (
            cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        set_resize_scale(dataloader.dataset.pipeline, args.image_size)
    for scheduler in cfg.param_scheduler:
        if scheduler.type == 'MultiStepLR':
            scheduler.end = args.epochs
            scheduler.milestones = [
                round(args.epochs * 2 / 3),
                round(args.epochs * 11 / 12)]
    cfg.dbss_experiment = dict(
        variant=variant,
        image_size=args.image_size,
        feature_stride=8,
        seed=args.seed)
    output = Path(cfg.work_dir) / 'patched_config.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    return output


def latest_metrics(work_dir: Path) -> dict[str, float]:
    scalar_files = sorted(work_dir.glob('**/vis_data/scalars.json'))
    if not scalar_files:
        return {}
    metrics: dict[str, float] = {}
    for line in scalar_files[-1].read_text(encoding='utf-8').splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key, value in record.items():
            if isinstance(value, (int, float)) and (
                    key.startswith('coco/') or key.startswith('dbss_')
                    or key == 'loss_dbss_sep'):
                metrics[key] = float(value)
    return metrics


def dbss_diagnostics(config: Path, checkpoint: Path) -> dict[str, Any]:
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(str(config))
    model = init_detector(cfg, str(checkpoint), device='cuda:0')
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if not hasattr(model.neck, 'forward_with_aux'):
        return dict(parameters=parameters)
    loader = Runner.build_dataloader(cfg.val_dataloader)
    batch = model.data_preprocessor(next(iter(loader)), training=False)
    samples = batch['data_samples']
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        features, aux = model._extract_feat_with_dbss_aux(
            batch['inputs'], samples)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    objective = model.separation_objective(aux, samples)
    return dict(
        parameters=parameters,
        batch_latency_seconds=elapsed,
        batch_size=len(samples),
        displacement_ratio=float(aux['displacement_ratio']),
        basis_max_cosine=float(aux['basis_max_cosine'].mean()),
        basis_effective_rank=float(aux['basis_effective_rank'].mean()),
        gap_pre=float(objective['dbss_gap_pre']),
        gap_post=float(objective['dbss_gap_post']),
        gap_gain=float(objective['dbss_gap_gain']),
        active_ratio=float(objective['dbss_active_ratio']),
        p3_shape=list(features[0].shape))


def run_variant(
        variant: str,
        args: argparse.Namespace,
        dataset_out: Path,
        image_dir: Path) -> None:
    config = write_config(variant, args, dataset_out, image_dir)
    print(f'CONFIG {variant}: {config}')
    if args.dry_run:
        return
    work_dir = config.parent
    if not args.test_only:
        command = [
            sys.executable, str(levir.mmdet_root() / 'tools/train.py'),
            str(config), '--work-dir', str(work_dir), '--auto-scale-lr']
        if args.amp:
            command.append('--amp')
        levir.run(command)
    checkpoint = levir.find_checkpoint(work_dir)
    predictions = work_dir / 'test_results/predictions.pkl'
    predictions.parent.mkdir(parents=True, exist_ok=True)
    levir.run([
        sys.executable, str(levir.mmdet_root() / 'tools/test.py'),
        str(config), str(checkpoint), '--work-dir', str(predictions.parent),
        '--out', str(predictions)])
    summary = dict(
        variant=variant,
        checkpoint=str(checkpoint),
        metrics=latest_metrics(work_dir))
    if variant != 'baseline':
        summary['dbss'] = dbss_diagnostics(config, checkpoint)
    (work_dir / 'experiment_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')


def main() -> None:
    args = parse_args()
    variants = comma_list(args.variants)
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    if args.num_machines < 1:
        raise ValueError('--num-machines must be positive')
    if not 0 <= args.machine_index < args.num_machines:
        raise ValueError('--machine-index must be in [0, num_machines)')
    dataset_out, image_dir = levir.prepare_coco_dataset(args)
    assigned = [
        variant for index, variant in enumerate(variants)
        if index % args.num_machines == args.machine_index]
    print(f'Assigned variants: {assigned}')
    for variant in assigned:
        run_variant(variant, args, dataset_out, image_dir)


if __name__ == '__main__':
    main()
