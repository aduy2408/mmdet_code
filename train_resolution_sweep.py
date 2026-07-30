#!/usr/bin/env python3
"""Run the controlled FCOS resolution sweep on LEVIR-Ship."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import train_all_levir_baseline as levir


VARIANTS = {
    'raw': 'configs/hit/fcos_r50-caffe_fpn_30e_levir-ship-768.py',
    'dbss_gamma06': 'projects/dbss/configs/fcos_dbss_ridge_gamma06.py',
    'rcfn_subpixel_v3':
    'configs/hit/fcos_r50-caffe_fpn_subpixel-inr_30e_levir-ship-768.py',
    'pg_ch': 'projects/rcfn_ltmr/configs/fcos_pg_ch.py',
}
DEFAULT_VARIANTS = tuple(VARIANTS)
DEFAULT_RESOLUTIONS = (1024, 1376)
MICRO_BATCH = {1024: 4, 1376: 2}
ACCUMULATION = {1024: 2, 1376: 4}


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def int_list(value: str) -> list[int]:
    return [int(item) for item in comma_list(value)]


def set_resize_scale(obj: Any, image_size: int) -> None:
    if isinstance(obj, dict):
        if obj.get('type') == 'Resize':
            obj['scale'] = (image_size, image_size)
        for value in obj.values():
            set_resize_scale(value, image_size)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            set_resize_scale(value, image_size)


def git_sha() -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def patch_scheduler(cfg: Any, epochs: int) -> None:
    for scheduler in cfg.param_scheduler:
        if scheduler.type == 'MultiStepLR':
            scheduler.end = epochs
            scheduler.milestones = [
                round(epochs * 2 / 3),
                round(epochs * 11 / 12),
            ]


def write_config(
    variant: str,
    resolution: int,
    args: argparse.Namespace,
    dataset_out: Path,
    image_dir: Path,
    sha: str,
) -> Path:
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
        set_resize_scale(dataloader.dataset.pipeline, resolution)

    cfg.train_dataloader.batch_size = MICRO_BATCH[resolution]
    cfg.optim_wrapper.accumulative_counts = ACCUMULATION[resolution]
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)
    patch_scheduler(cfg, args.epochs)
    cfg.work_dir = str(
        Path(args.work_dir).resolve() / sha / str(resolution) / variant)
    cfg.resolution_sweep = dict(
        variant=variant,
        resolution=resolution,
        epochs=args.epochs,
        seed=args.seed,
        micro_batch=MICRO_BATCH[resolution],
        accumulation=ACCUMULATION[resolution],
        effective_batch=(
            MICRO_BATCH[resolution] * ACCUMULATION[resolution]),
        git_sha=sha,
    )
    output = Path(cfg.work_dir) / 'patched_config.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    return output


def latest_validation_metrics(work_dir: Path) -> dict[str, float]:
    best: dict[str, float] = {}
    for scalar_file in sorted(work_dir.glob('**/vis_data/scalars.json')):
        for line in scalar_file.read_text(encoding='utf-8').splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = record.get('coco/bbox_mAP')
            if not isinstance(value, (int, float)):
                continue
            if not best or float(value) > best['coco/bbox_mAP']:
                best = {
                    key: float(metric)
                    for key, metric in record.items()
                    if key.startswith('coco/')
                    and isinstance(metric, (int, float))
                }
    return best


def write_state(root: Path, **state: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / 'sweep_state.json').write_text(
        json.dumps(state, indent=2), encoding='utf-8')


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def train_config(config: Path, amp: bool) -> None:
    command = [
        sys.executable,
        str(levir.mmdet_root() / 'tools/train.py'),
        str(config),
        '--work-dir',
        str(config.parent),
    ]
    if amp:
        command.append('--amp')
    run(command)


def test_config(config: Path) -> None:
    checkpoint = levir.find_checkpoint(config.parent)
    test_dir = config.parent / 'test_results'
    test_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        str(levir.mmdet_root() / 'tools/test.py'),
        str(config),
        str(checkpoint),
        '--work-dir',
        str(test_dir),
        '--out',
        str(test_dir / 'predictions.pkl'),
    ])


def select_winner(
    configs: dict[tuple[int, str], Path],
) -> tuple[int, str]:
    candidates = []
    for (resolution, variant), config in configs.items():
        metrics = latest_validation_metrics(config.parent)
        if 'coco/bbox_mAP' not in metrics:
            raise RuntimeError(f'No validation mAP found in {config.parent}')
        candidates.append((
            metrics['coco/bbox_mAP'],
            metrics.get('coco/bbox_mAP_75', float('-inf')),
            resolution,
            variant,
        ))
    best_map = max(item[0] for item in candidates)
    close = [item for item in candidates if best_map - item[0] < 0.003]
    _, _, resolution, variant = max(close, key=lambda item: (item[1], item[0]))
    return resolution, variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='LevirShipData')
    parser.add_argument(
        '--dataset-out', default='mmdetection/data/levir_ship_coco')
    parser.add_argument(
        '--work-dir', default='mmdetection/work_dirs/resolution_sweep')
    parser.add_argument('--variants', default=','.join(DEFAULT_VARIANTS))
    parser.add_argument(
        '--resolutions',
        default=','.join(str(value) for value in DEFAULT_RESOLUTIONS))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--screen-only', action='store_true')
    parser.add_argument('--skip-data-prepare', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = comma_list(args.variants)
    resolutions = int_list(args.resolutions)
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    unknown_resolutions = sorted(set(resolutions) - set(MICRO_BATCH))
    if unknown_variants:
        raise ValueError(f'Unknown variants: {unknown_variants}')
    if unknown_resolutions:
        raise ValueError(f'Unknown resolutions: {unknown_resolutions}')

    if args.skip_data_prepare:
        dataset_out = Path(args.dataset_out).resolve()
        image_dir = (Path(args.data_root) / 'All Images').resolve()
        for split in ('train', 'val', 'test'):
            annotation = dataset_out / 'annotations' / f'{split}.json'
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
    else:
        dataset_out, image_dir = levir.prepare_coco_dataset(args)

    sha = git_sha()
    state_root = Path(args.work_dir).resolve() / sha
    configs = {
        (resolution, variant): write_config(
            variant, resolution, args, dataset_out, image_dir, sha)
        for resolution in resolutions
        for variant in variants
    }
    for key, config in configs.items():
        print(f'CONFIG {key[0]} {key[1]}: {config}')
    if args.dry_run:
        return

    completed = []
    for (resolution, variant), config in configs.items():
        write_state(
            state_root,
            status='running',
            resolution=resolution,
            variant=variant,
            completed=completed,
            git_sha=sha,
        )
        try:
            train_config(config, args.amp)
        except Exception:
            write_state(
                state_root,
                status='failed',
                resolution=resolution,
                variant=variant,
                completed=completed,
                git_sha=sha,
            )
            raise
        completed.append([resolution, variant])

    winner_resolution, winner_variant = select_winner(configs)
    tested = []
    if not args.screen_only:
        keys = [(winner_resolution, winner_variant)]
        baseline_key = (winner_resolution, 'raw')
        if winner_variant != 'raw' and baseline_key in configs:
            keys.append(baseline_key)
        for key in keys:
            test_config(configs[key])
            tested.append(list(key))
    write_state(
        state_root,
        status='completed',
        completed=completed,
        winner=[winner_resolution, winner_variant],
        tested=tested,
        git_sha=sha,
    )


if __name__ == '__main__':
    main()
