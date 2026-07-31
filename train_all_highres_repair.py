#!/usr/bin/env python3
"""Run the repaired high-resolution FCOS sweep on LEVIR-Ship."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
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
LEARNING_RATE = 0.005
EPOCHS = {
    'raw': 30,
    'dbss_gamma06': 20,
    'rcfn_subpixel_v3': 30,
    'pg_ch': 30,
}
MILESTONES = {
    'raw': (8, 11),
    'dbss_gamma06': (13, 18),
    'rcfn_subpixel_v3': (8, 11),
    'pg_ch': (8, 11),
}
REFERENCE = {
    'raw': {
        'resolution': 768,
        'test_mAP': 0.259,
        'url': 'https://huggingface.co/datasets/duyle2408/'
               'phase-subpixel-p3-levir-ablation',
    },
    'dbss_gamma06': {
        'resolution': 768,
        'test_mAP': 0.282,
        'source_commit': 'dcf9db4293f7ec4486edf06a9b472c393e3271d3',
        'url': 'https://huggingface.co/datasets/duyle2408/fcos_test_dbss',
    },
    'rcfn_subpixel_v3': {
        'resolution': 768,
        'test_mAP': None,
        'provenance': 'reported 0.273 is not backed by the cited test artifact',
        'url': 'https://huggingface.co/datasets/duyle2408/'
               'phase-subpixel-p3-levir-ablation',
    },
    'pg_ch': {
        'resolution': 512,
        'test_mAP': 0.264,
        'url': 'https://huggingface.co/datasets/duyle2408/fcos_test_pg_rcfn',
    },
}
IMPLEMENTATION_FILES = {
    'raw': ('mmdetection/configs/hit/'
            'fcos_r50-caffe_fpn_30e_levir-ship-768.py',),
    'dbss_gamma06': (
        'mmdetection/projects/dbss/models/dbss_fpn.py',
        'mmdetection/projects/dbss/models/dbss_fcos.py',
        'mmdetection/projects/dbss/configs/fcos_dbss_ridge_gamma06.py',
    ),
    'rcfn_subpixel_v3': (
        'mmdetection/mmdet/models/necks/feature_augment_neck.py',
        'mmdetection/configs/hit/'
        'fcos_r50-caffe_fpn_subpixel-inr_30e_levir-ship-768.py',
    ),
    'pg_ch': (
        'mmdetection/projects/rcfn_ltmr/models/rcfn_fpn.py',
        'mmdetection/projects/rcfn_ltmr/models/pg_rcfn_fcos.py',
        'mmdetection/projects/rcfn_ltmr/configs/fcos_pg_ch.py',
    ),
}
WARMUP_OPTIMIZER_UPDATES = 500


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


def patch_scheduler(cfg: Any, variant: str, resolution: int) -> None:
    for scheduler in cfg.param_scheduler:
        if scheduler.type == 'ConstantLR':
            scheduler.end = (
                WARMUP_OPTIMIZER_UPDATES * ACCUMULATION[resolution])
        if scheduler.type == 'MultiStepLR':
            scheduler.end = EPOCHS[variant]
            scheduler.milestones = list(MILESTONES[variant])


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_counts(dataset_out: Path) -> dict[str, dict[str, int]]:
    counts = {}
    for split in ('train', 'val', 'test'):
        payload = json.loads(
            (dataset_out / 'annotations' / f'{split}.json').read_text())
        counts[split] = {
            'images': len(payload['images']),
            'annotations': len(payload['annotations']),
        }
    return counts


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
    variant_args = argparse.Namespace(**vars(args))
    variant_args.epochs = EPOCHS[variant]
    cfg = levir.patch_config(
        cfg, variant, variant_args, dataset_out, image_dir)
    for dataloader in (
            cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        set_resize_scale(dataloader.dataset.pipeline, resolution)

    cfg.train_dataloader.batch_size = MICRO_BATCH[resolution]
    cfg.optim_wrapper.accumulative_counts = ACCUMULATION[resolution]
    cfg.optim_wrapper.optimizer.lr = LEARNING_RATE
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)
    patch_scheduler(cfg, variant, resolution)
    cfg.work_dir = str(
        Path(args.work_dir).resolve() / sha / str(resolution) / variant)
    cfg.resolution_sweep = dict(
        variant=variant,
        resolution=resolution,
        protocol='highres_repair_v1',
        epochs=EPOCHS[variant],
        milestones=list(MILESTONES[variant]),
        seed=args.seed,
        micro_batch=MICRO_BATCH[resolution],
        accumulation=ACCUMULATION[resolution],
        effective_batch=(
            MICRO_BATCH[resolution] * ACCUMULATION[resolution]),
        learning_rate=LEARNING_RATE,
        warmup_dataloader_iterations=(
            WARMUP_OPTIMIZER_UPDATES * ACCUMULATION[resolution]),
        warmup_optimizer_updates=WARMUP_OPTIMIZER_UPDATES,
        git_sha=sha,
    )
    output = Path(cfg.work_dir) / 'patched_config.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    repo_root = Path(__file__).resolve().parent
    manifest = {
        **dict(cfg.resolution_sweep),
        'dataset': dataset_counts(dataset_out),
        'historical_reference': REFERENCE[variant],
        'implementation_fingerprints': {
            relative: file_sha256(repo_root / relative)
            for relative in IMPLEMENTATION_FILES[variant]
        },
    }
    (output.parent / 'protocol_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding='utf-8')
    return output


def latest_validation_metrics(work_dir: Path) -> dict[str, float]:
    best: dict[str, float] = {}
    for scalar_file in sorted(work_dir.glob('**/*.json')):
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


def run(command: list[str]) -> float:
    started = time.monotonic()
    subprocess.run(command, check=True)
    return time.monotonic() - started


def peak_vram_mb(root: Path) -> int | None:
    values = []
    pattern = re.compile(r'\bmemory:\s*(\d+)\b')
    for log in root.glob('**/*.log'):
        for match in pattern.finditer(
                log.read_text(encoding='utf-8', errors='replace')):
            values.append(int(match.group(1)))
    return max(values) if values else None


def write_summary(root: Path, **summary: Any) -> Path:
    output = root / 'run_summary.json'
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding='utf-8')
    return output


def upload_run(
    folder: Path,
    resolution: int,
    variant: str,
    path_suffix: str,
    args: argparse.Namespace,
    sha: str,
) -> None:
    if args.no_hf_upload:
        return
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN is required unless --no-hf-upload is set')
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.hf_repo_id,
        repo_type='dataset',
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(folder),
        path_in_repo=f'{sha}/{resolution}/{variant}/{path_suffix}',
        repo_id=args.hf_repo_id,
        repo_type='dataset',
    )


def train_config(config: Path, amp: bool) -> float:
    command = [
        sys.executable,
        str(levir.mmdet_root() / 'tools/train.py'),
        str(config),
        '--work-dir',
        str(config.parent),
    ]
    if amp:
        command.append('--amp')
    return run(command)


def test_config(config: Path, checkpoint: Path) -> tuple[Path, float]:
    test_dir = config.parent / 'checkpoint_tests' / checkpoint.stem
    test_dir.mkdir(parents=True, exist_ok=True)
    duration = run([
        sys.executable,
        str(levir.mmdet_root() / 'tools/test.py'),
        str(config),
        str(checkpoint),
        '--work-dir',
        str(test_dir),
        '--out',
        str(test_dir / 'predictions.pkl'),
    ])
    metrics = latest_validation_metrics(test_dir)
    write_summary(
        test_dir,
        checkpoint=checkpoint.name,
        metrics=metrics,
        duration_seconds=duration,
        peak_vram_mb=peak_vram_mb(test_dir),
        return_code=0,
    )
    return test_dir, duration


def checkpoints(config: Path) -> list[Path]:
    best = sorted(config.parent.glob('best_*.pth'))
    epoch_files = sorted(config.parent.glob('epoch_*.pth'))
    selected = best + epoch_files
    unique = []
    for checkpoint in selected:
        if checkpoint not in unique:
            unique.append(checkpoint)
    if len(best) != 1 or len(epoch_files) != 1:
        raise RuntimeError(
            f'Expected one best and one final checkpoint in {config.parent}, '
            f'got best={best}, epoch={epoch_files}')
    return unique


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
        '--work-dir', default='mmdetection/work_dirs/highres_sweep_repair')
    parser.add_argument('--variants', default=','.join(DEFAULT_VARIANTS))
    parser.add_argument(
        '--resolutions',
        default=','.join(str(value) for value in DEFAULT_RESOLUTIONS))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-data-prepare', action='store_true')
    parser.add_argument(
        '--hf-repo-id',
        default='duyle2408/levir-highres-sweep-repair')
    parser.add_argument('--no-hf-upload', action='store_true')
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
            duration = train_config(config, args.amp)
            write_summary(
                config.parent,
                phase='training',
                duration_seconds=duration,
                peak_vram_mb=peak_vram_mb(config.parent),
                return_code=0,
            )
            write_state(
                state_root,
                status='uploading',
                resolution=resolution,
                variant=variant,
                completed=completed,
                git_sha=sha,
            )
            upload_run(
                config.parent,
                resolution,
                variant,
                'training',
                args,
                sha,
            )
        except Exception as error:
            write_summary(
                config.parent,
                phase='training',
                return_code=getattr(error, 'returncode', 1),
                error=type(error).__name__,
            )
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
    try:
        for key, config in configs.items():
            for checkpoint in checkpoints(config):
                write_state(
                    state_root,
                    status='testing',
                    resolution=key[0],
                    variant=key[1],
                    checkpoint=checkpoint.name,
                    completed=completed,
                    tested=tested,
                    winner=[winner_resolution, winner_variant],
                    git_sha=sha,
                )
                test_dir, _ = test_config(config, checkpoint)
                write_state(
                    state_root,
                    status='uploading',
                    resolution=key[0],
                    variant=key[1],
                    checkpoint=checkpoint.name,
                    completed=completed,
                    tested=tested,
                    winner=[winner_resolution, winner_variant],
                    git_sha=sha,
                )
                upload_run(
                    test_dir,
                    key[0],
                    key[1],
                    checkpoint.stem,
                    args,
                    sha,
                )
                tested.append([key[0], key[1], checkpoint.name])
    except Exception:
        write_state(
            state_root,
            status='failed',
            completed=completed,
            tested=tested,
            winner=[winner_resolution, winner_variant],
            git_sha=sha,
        )
        raise
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
