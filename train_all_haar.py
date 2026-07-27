#!/usr/bin/env python3
"""Train, screen, test, and upload PAHR ablations on LEVIR-Ship."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import train_all_levir_baseline as levir


MODEL_CONFIG = 'projects/pahr/configs/fcos_pahr.py'
VARIANTS = {
    'haar_recompose_512': dict(image_size=512, phase_shift=False, giou=False),
    'haar_shift_512': dict(image_size=512, phase_shift=True, giou=False),
    'haar_shift_giou_512': dict(
        image_size=512, phase_shift=True, giou=True),
    'haar_shift_768': dict(image_size=768, phase_shift=True, giou=False),
    'haar_shift_giou_768': dict(
        image_size=768, phase_shift=True, giou=True),
    'haar_shift_sched_768': dict(
        image_size=768, phase_shift=True, giou=False, scaled_schedule=True),
    'haar_v3_gate_768': dict(
        image_size=768,
        phase_shift=True,
        giou=False,
        scaled_schedule=True,
        v3_gate=True),
    'haar_v3_gate_lr10_768': dict(
        image_size=768,
        phase_shift=True,
        giou=False,
        scaled_schedule=True,
        v3_gate=True,
        detail_lr_mult=10.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='LevirShipData')
    parser.add_argument(
        '--dataset-out', default='mmdetection/data/levir_ship_coco')
    parser.add_argument(
        '--work-dir', default='mmdetection/work_dirs/levir_haar_v2')
    parser.add_argument(
        '--variants',
        default=','.join(VARIANTS),
        help=f"Comma-separated variants: {', '.join(VARIANTS)}.")
    parser.add_argument(
        '--image-size',
        type=int,
        choices=(512, 768),
        help='Override the image size encoded by each selected variant.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Maximum images per split after scene-safe splitting; 0 uses all.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume each selected variant from its latest checkpoint.')
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Train and validate without evaluating the held-out test split.')
    parser.add_argument(
        '--hf-repo-id', default='duyle2408/fcos_test_haar')
    parser.add_argument('--hf-repo-type', default='dataset')
    parser.add_argument(
        '--hf-token',
        default='',
        help='Hugging Face token; defaults to the HF_TOKEN environment variable.')
    parser.add_argument(
        '--skip-upload',
        '--no-hf-upload',
        dest='no_hf_upload',
        action='store_true',
        help='Skip Hugging Face upload.')
    return parser.parse_args()


def set_resize_scale(obj, image_size: int) -> None:
    if isinstance(obj, dict):
        if obj.get('type') == 'Resize':
            obj['scale'] = (image_size, image_size)
        for value in obj.values():
            set_resize_scale(value, image_size)
    elif isinstance(obj, list):
        for value in obj:
            set_resize_scale(value, image_size)


def scale_schedule(cfg, epochs: int) -> None:
    milestones = [round(epochs * 2 / 3), round(epochs * 11 / 12)]
    for scheduler in cfg.param_scheduler:
        if scheduler.type == 'MultiStepLR':
            scheduler.end = epochs
            scheduler.milestones = milestones


def write_variant_config(variant: str, args: argparse.Namespace,
                         dataset_out: Path, image_dir: Path) -> Path:
    root = str(levir.mmdet_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from mmengine.config import Config
    from mmdet.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(str(levir.mmdet_root() / MODEL_CONFIG))
    cfg = levir.patch_config(
        cfg, variant, args, dataset_out, image_dir)
    settings = VARIANTS[variant]
    image_size = args.image_size or settings['image_size']
    for dataloader in (
            cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        set_resize_scale(dataloader.dataset.pipeline, image_size)
    cfg.model.use_phase_shift = settings['phase_shift']
    if settings.get('v3_gate'):
        cfg.model.neck.update(
            gate_power=0.5,
            correction_gate_floor=0.05,
            detach_position_gate=True)
    if settings.get('scaled_schedule'):
        scale_schedule(cfg, args.epochs)
    if detail_lr_mult := settings.get('detail_lr_mult'):
        paramwise = cfg.optim_wrapper.paramwise_cfg
        custom_keys = dict(paramwise.get('custom_keys', {}))
        custom_keys['neck.detail_mixer.3'] = dict(
            lr_mult=detail_lr_mult, decay_mult=0.0)
        paramwise.custom_keys = custom_keys
    if settings['giou']:
        cfg.model.bbox_head.loss_bbox = dict(
            type='GIoULoss', loss_weight=1.0)
    cfg.default_hooks.checkpoint.update(
        save_best='coco/bbox_mAP_75', rule='greater')
    cfg.resume = args.resume
    cfg.pahr_variant = dict(
        name=variant,
        image_size=image_size,
        phase_shift=settings['phase_shift'],
        giou=settings['giou'],
        scaled_schedule=settings.get('scaled_schedule', False),
        v3_gate=settings.get('v3_gate', False),
        detail_lr_mult=settings.get('detail_lr_mult', 1.0))
    output = Path(cfg.work_dir) / 'patched_config.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    return output


def upload_variant(variant: str, args: argparse.Namespace) -> None:
    if args.no_hf_upload:
        return
    token = args.hf_token or os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError(
            'Upload requires --hf-token or HF_TOKEN; use --skip-upload '
            'to keep results local.')
    from huggingface_hub import HfApi

    work_dir = levir.resolve_path(args.work_dir) / variant
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        private=False,
        exist_ok=True)
    print(
        f'UPLOAD {work_dir} -> '
        f'hf://{args.hf_repo_type}/{args.hf_repo_id}/{variant}')
    api.upload_folder(
        folder_path=str(work_dir),
        path_in_repo=variant,
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type)


def run_variant(variant: str, args: argparse.Namespace,
                dataset_out: Path, image_dir: Path) -> None:
    config = write_variant_config(variant, args, dataset_out, image_dir)
    work_dir = levir.resolve_path(args.work_dir) / variant
    if not args.test_only:
        command = [
            sys.executable,
            str(levir.mmdet_root() / 'tools/train.py'),
            str(config),
            '--work-dir',
            str(work_dir),
            '--auto-scale-lr',
        ]
        if args.amp:
            command.append('--amp')
        levir.run(command)
    if not args.skip_test:
        checkpoint = levir.find_checkpoint(work_dir)
        levir.run([
            sys.executable,
            str(levir.mmdet_root() / 'tools/test.py'),
            str(config),
            str(checkpoint),
            '--work-dir',
            str(work_dir / 'test_results'),
            '--out',
            str(work_dir / 'test_results/predictions.pkl'),
        ])
        levir.run([
            sys.executable,
            str(levir.mmdet_root() /
                'projects/pahr/tools/summarize_pahr.py'),
            '--config',
            str(config),
            '--checkpoint',
            str(checkpoint),
            '--predictions',
            str(work_dir / 'test_results/predictions.pkl'),
            '--output',
            str(work_dir / 'test_results/pahr_summary.json'),
        ])
    upload_variant(variant, args)


def main() -> None:
    args = parse_args()
    variants = levir.comma_list(args.variants)
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    if args.test_only and args.skip_test:
        raise ValueError('--test-only and --skip-test cannot be combined')
    dataset_out, image_dir = levir.prepare_coco_dataset(args)
    for variant in variants:
        if args.dry_run:
            config = write_variant_config(
                variant, args, dataset_out, image_dir)
            print(f'CONFIG {variant}: {config}')
        else:
            run_variant(variant, args, dataset_out, image_dir)


if __name__ == '__main__':
    main()
