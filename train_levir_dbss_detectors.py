#!/usr/bin/env python3
"""Run paired baseline/DBSS experiments across LEVIR-Ship detectors."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import train_all_levir_baseline as levir


MODEL_CONFIGS = {
    'atss': 'configs/atss/atss_r50_fpn_1x_coco.py',
    'retinanet': 'configs/retinanet/retinanet_r50_fpn_1x_coco.py',
    'faster_rcnn': 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
    'cascade_rcnn':
    'configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py',
}
DBSS_TYPES = {
    'atss': 'DBSSATSS',
    'retinanet': 'DBSSRetinaNet',
    'faster_rcnn': 'DBSSFasterRCNN',
    'cascade_rcnn': 'DBSSCascadeRCNN',
}
MICRO_BATCH = {
    'atss': 4,
    'retinanet': 4,
    'faster_rcnn': 2,
    'cascade_rcnn': 1,
}
DEFAULT_MODELS = tuple(MODEL_CONFIGS)
DEFAULT_VARIANTS = ('baseline', 'dbss_gamma06')
EFFECTIVE_BATCH = 8
REFERENCE_BATCH = 16
WARMUP_OPTIMIZER_UPDATES = 500
DBSS_GRAD_MAX_NORM = 10.0
MILESTONES = (13, 18)


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def git_sha() -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def set_resize_scale(obj: Any, image_size: int) -> None:
    if isinstance(obj, dict):
        if obj.get('type') == 'Resize':
            obj['scale'] = (image_size, image_size)
        for value in obj.values():
            set_resize_scale(value, image_size)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            set_resize_scale(value, image_size)


def first_stride(cfg: Any) -> int:
    head = cfg.model.get('bbox_head', cfg.model.get('rpn_head'))
    strides = head.anchor_generator.strides
    first = strides[0]
    return int(first[0] if isinstance(first, (list, tuple)) else first)


def patch_schedule(cfg: Any, epochs: int, accumulation: int) -> None:
    cfg.train_cfg.max_epochs = epochs
    cfg.train_cfg.val_interval = 1
    for scheduler in cfg.param_scheduler:
        if scheduler.type == 'LinearLR':
            scheduler.end = WARMUP_OPTIMIZER_UPDATES * accumulation
        if scheduler.type == 'MultiStepLR':
            scheduler.end = epochs
            scheduler.milestones = list(MILESTONES)


def patch_dbss(cfg: Any, model_name: str, target_stride: int) -> None:
    cfg.custom_imports = dict(imports=['projects.dbss'])
    cfg.model.type = DBSS_TYPES[model_name]
    cfg.model.target_stride = target_stride
    cfg.model.improvement_margin = 0.03
    cfg.model.loss_sep_weight = 0.5
    cfg.model.neck.update(
        type='DBSSFPN',
        target_level='lowest',
        embed_channels=64,
        candidate_grid=(8, 8),
        shortlist_size=24,
        num_bases=8,
        diversity_beta=1.0,
        basis_similarity_threshold=0.9,
        selector_mode='legacy_forced_k',
        residual_mode='ridge',
        projection_mode='ridge',
        ridge_lambda=1e-3,
        temperature=0.1,
        gamma_max=0.6,
        use_haar_reliability=False,
        hidden_channels=64,
        legacy_artifact_mode=True,
    )
    paramwise = cfg.optim_wrapper.get('paramwise_cfg', {})
    custom_keys = dict(paramwise.get('custom_keys', {}))
    custom_keys['neck.direction.2'] = dict(lr_mult=10.0, decay_mult=0.0)
    paramwise['custom_keys'] = custom_keys
    cfg.optim_wrapper.paramwise_cfg = paramwise
    cfg.optim_wrapper.clip_grad = dict(
        max_norm=DBSS_GRAD_MAX_NORM, norm_type=2)


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
    model_name: str,
    variant: str,
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
    cfg = Config.fromfile(
        str(levir.mmdet_root() / MODEL_CONFIGS[model_name]))
    patch_args = argparse.Namespace(**vars(args))
    patch_args.batch_size = MICRO_BATCH[model_name]
    cfg = levir.patch_config(
        cfg, model_name, patch_args, dataset_out, image_dir)
    for dataloader in (
            cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        set_resize_scale(dataloader.dataset.pipeline, args.image_size)

    target_stride = first_stride(cfg)
    micro_batch = MICRO_BATCH[model_name]
    accumulation = EFFECTIVE_BATCH // micro_batch
    cfg.train_dataloader.batch_size = micro_batch
    cfg.optim_wrapper.accumulative_counts = accumulation
    cfg.optim_wrapper.optimizer.lr *= EFFECTIVE_BATCH / REFERENCE_BATCH
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=EFFECTIVE_BATCH)
    patch_schedule(cfg, args.epochs, accumulation)
    if variant == 'dbss_gamma06':
        patch_dbss(cfg, model_name, target_stride)

    cfg.work_dir = str(
        Path(args.work_dir).resolve() / sha / model_name / variant)
    cfg.dbss_detector_sweep = dict(
        protocol='dbss_detector_sweep_v1',
        git_sha=sha,
        model=model_name,
        variant=variant,
        image_size=args.image_size,
        epochs=args.epochs,
        milestones=list(MILESTONES),
        seed=args.seed,
        target_level='lowest',
        target_stride=target_stride,
        target_pyramid_level=f'P{int(math.log2(target_stride))}',
        micro_batch=micro_batch,
        accumulation=accumulation,
        effective_batch=EFFECTIVE_BATCH,
        learning_rate=cfg.optim_wrapper.optimizer.lr,
        warmup_optimizer_updates=WARMUP_OPTIMIZER_UPDATES,
        warmup_dataloader_iterations=(
            WARMUP_OPTIMIZER_UPDATES * accumulation),
        grad_clip_max_norm=(
            DBSS_GRAD_MAX_NORM if variant == 'dbss_gamma06' else None),
    )
    output = Path(cfg.work_dir) / 'patched_config.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    manifest = dict(
        cfg.dbss_detector_sweep,
        dataset=dataset_counts(dataset_out),
        base_config=MODEL_CONFIGS[model_name],
    )
    (output.parent / 'protocol_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    return output


def run(command: list[str], cwd: Path) -> float:
    started = time.monotonic()
    env = os.environ.copy()
    env.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(levir.mmdet_root()) + (
        os.pathsep + pythonpath if pythonpath else '')
    subprocess.run(command, cwd=cwd, env=env, check=True)
    return time.monotonic() - started


def latest_metrics(root: Path) -> dict[str, float]:
    best: dict[str, float] = {}
    for path in sorted(root.glob('**/*.json')):
        for line in path.read_text(errors='replace').splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            value = record.get('coco/bbox_mAP')
            if isinstance(value, (int, float)):
                best = {
                    key: float(metric)
                    for key, metric in record.items()
                    if key.startswith('coco/')
                    and isinstance(metric, (int, float))
                }
    return best


def checkpoint(work_dir: Path) -> Path:
    best = sorted(work_dir.glob('best_*.pth'))
    if best:
        return best[0]
    latest = work_dir / 'latest.pth'
    if latest.is_file():
        return latest
    epochs = sorted(work_dir.glob('epoch_*.pth'))
    if epochs:
        return epochs[-1]
    raise FileNotFoundError(f'No checkpoint in {work_dir}')


def upload(work_dir: Path, model: str, variant: str,
           args: argparse.Namespace, sha: str) -> None:
    if args.no_hf_upload:
        return
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN is required unless --no-hf-upload is set')
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.hf_repo_id, repo_type='dataset', exist_ok=True)
    api.upload_folder(
        folder_path=str(work_dir),
        path_in_repo=f'{sha}/{model}/{variant}',
        repo_id=args.hf_repo_id,
        repo_type='dataset',
    )


def write_state(root: Path, **values: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / 'sweep_state.json').write_text(
        json.dumps(values, indent=2, sort_keys=True), encoding='utf-8')


def run_job(config: Path, model: str, variant: str,
            args: argparse.Namespace, sha: str) -> dict[str, Any]:
    work_dir = config.parent
    train_seconds = run([
        sys.executable,
        str(levir.mmdet_root() / 'tools/train.py'),
        str(config),
        '--work-dir',
        str(work_dir),
    ], levir.mmdet_root())
    best = checkpoint(work_dir)
    test_dir = work_dir / 'test_results'
    test_dir.mkdir(exist_ok=True)
    test_seconds = run([
        sys.executable,
        str(levir.mmdet_root() / 'tools/test.py'),
        str(config),
        str(best),
        '--work-dir',
        str(test_dir),
        '--out',
        str(test_dir / 'predictions.pkl'),
    ], levir.mmdet_root())
    summary = dict(
        model=model,
        variant=variant,
        checkpoint=best.name,
        validation_metrics=latest_metrics(work_dir),
        test_metrics=latest_metrics(test_dir),
        train_seconds=train_seconds,
        test_seconds=test_seconds,
        return_code=0,
    )
    (work_dir / 'run_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    upload(work_dir, model, variant, args, sha)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='LevirShipData')
    parser.add_argument(
        '--dataset-out', default='mmdetection/data/levir_ship_coco')
    parser.add_argument(
        '--work-dir',
        default='mmdetection/work_dirs/levir_dbss_detector_sweep')
    parser.add_argument('--models', default=','.join(DEFAULT_MODELS))
    parser.add_argument('--variants', default=','.join(DEFAULT_VARIANTS))
    parser.add_argument(
        '--jobs',
        default='',
        help='Optional model/variant pairs; overrides --models/--variants.')
    parser.add_argument('--image-size', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-data-prepare', action='store_true')
    parser.add_argument(
        '--hf-repo-id', default='duyle2408/levir-dbss-detector-sweep')
    parser.add_argument('--no-hf-upload', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = comma_list(args.models)
    variants = comma_list(args.variants)
    unknown_models = sorted(set(models) - set(MODEL_CONFIGS))
    unknown_variants = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown_models or unknown_variants:
        raise ValueError(
            f'Unknown models={unknown_models}, variants={unknown_variants}')
    if args.jobs:
        jobs = []
        for item in comma_list(args.jobs):
            try:
                model, variant = item.split('/', 1)
            except ValueError as error:
                raise ValueError(f'Invalid job {item!r}; use model/variant') from error
            if model not in MODEL_CONFIGS or variant not in DEFAULT_VARIANTS:
                raise ValueError(f'Unknown job: {item}')
            jobs.append((model, variant))
        models = list(dict.fromkeys(model for model, _ in jobs))
    else:
        jobs = [
            (model, variant)
            for model in models
            for variant in variants
        ]
    if EFFECTIVE_BATCH % max(MICRO_BATCH[model] for model in models):
        raise ValueError('Micro batches must divide effective batch')

    if args.skip_data_prepare:
        dataset_out = Path(args.dataset_out).resolve()
        image_dir = (Path(args.data_root) / 'All Images').resolve()
    else:
        dataset_out, image_dir = levir.prepare_coco_dataset(args)
    for split in ('train', 'val', 'test'):
        path = dataset_out / 'annotations' / f'{split}.json'
        if not path.is_file():
            raise FileNotFoundError(path)

    sha = git_sha()
    configs = {
        (model, variant): write_config(
            model, variant, args, dataset_out, image_dir, sha)
        for model, variant in jobs
    }
    for key, config in configs.items():
        print(f'CONFIG {key[0]}/{key[1]}: {config}')
    if args.dry_run:
        return

    state_root = Path(args.work_dir).resolve() / sha
    completed: list[list[str]] = []
    summaries = []
    for (model, variant), config in configs.items():
        write_state(
            state_root,
            status='running',
            current=[model, variant],
            completed=completed,
            git_sha=sha,
        )
        try:
            summaries.append(run_job(config, model, variant, args, sha))
        except Exception as error:
            write_state(
                state_root,
                status='failed',
                current=[model, variant],
                completed=completed,
                error=type(error).__name__,
                git_sha=sha,
            )
            raise
        completed.append([model, variant])
    write_state(
        state_root,
        status='completed',
        completed=completed,
        summaries=summaries,
        git_sha=sha,
    )


if __name__ == '__main__':
    main()
