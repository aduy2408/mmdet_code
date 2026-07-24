#!/usr/bin/env python3
"""Train, test, and upload FCOS-P3 HIT results for LEVIR-Ship."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from train_all_levir_baseline import prepare_coco_dataset


ROOT = Path(__file__).resolve().parent
MMDET_ROOT = ROOT / 'mmdetection'
CONFIG = MMDET_ROOT / 'configs/hit/fcos_r50-caffe_fpn-hit_12e_levir-ship.py'
WORK_DIR = MMDET_ROOT / 'work_dirs/levir_hit/fcos'
DATA_ROOT = ROOT / 'LevirShipData'
DATASET_OUT = MMDET_ROOT / 'data/levir_ship_coco'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--dataset-out', type=Path, default=DATASET_OUT)
    parser.add_argument('--dataset-seed', type=int, default=42)
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--auto-scale-lr', action='store_true')
    parser.add_argument(
        '--resume',
        nargs='?',
        const='auto',
        help='Resume automatically, or resume from the given checkpoint.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        default=[],
        help='Extra MMEngine overrides in key=value form.')
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Skip training and test the existing best/latest checkpoint.')
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip evaluation after training.')
    parser.add_argument(
        '--hf-repo-id',
        default='duyle2408/levir_ship_mmdet_runs')
    parser.add_argument('--hf-repo-type', default='dataset')
    parser.add_argument(
        '--hf-token',
        default='',
        help='Hugging Face token; defaults to the HF_TOKEN environment value.')
    parser.add_argument(
        '--no-hf-upload',
        action='store_true',
        help='Do not upload the completed work directory.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the resolved training command without running it.')
    return parser.parse_args()


def run(command: list[str], dry_run: bool = False) -> None:
    print(' '.join(command))
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    subprocess.run(command, cwd=MMDET_ROOT, env=env, check=True)


def find_checkpoint(work_dir: Path) -> Path:
    best = sorted(work_dir.glob('best_*.pth'))
    if best:
        return best[0]
    latest = work_dir / 'latest.pth'
    if latest.is_file():
        return latest
    raise FileNotFoundError(
        f'No best_*.pth or latest.pth found in {work_dir}')


def upload_to_huggingface(args: argparse.Namespace, work_dir: Path) -> None:
    if args.no_hf_upload or args.dry_run:
        return
    token = args.hf_token or os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError(
            'Hugging Face upload requires --hf-token or HF_TOKEN; '
            'pass --no-hf-upload to skip it.')
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            'Install huggingface_hub or pass --no-hf-upload.') from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        private=False,
        exist_ok=True)
    print(f'UPLOAD {work_dir} -> '
          f'hf://{args.hf_repo_type}/{args.hf_repo_id}/fcos_hit')
    api.upload_folder(
        folder_path=str(work_dir),
        path_in_repo='fcos_hit',
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('epochs/batch-size must be positive and workers >= 0')

    data_root = args.data_root.resolve()
    dataset_out = args.dataset_out.resolve()
    image_dir = data_root / 'All Images'
    annotation_dir = dataset_out / 'annotations'
    annotations = [
        annotation_dir / f'{split}.json'
        for split in ('train', 'val', 'test')
    ]
    if not all(path.is_file() for path in annotations):
        print(f'Preparing COCO annotations in {dataset_out}')
        prepare_coco_dataset(SimpleNamespace(
            data_root=data_root,
            dataset_out=dataset_out,
            seed=args.dataset_seed,
            limit=0,
        ))

    required = [
        CONFIG,
        image_dir,
        *annotations,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing LEVIR-HIT inputs:\n' +
                                '\n'.join(missing))

    image_alias = dataset_out / 'images'
    if image_alias.is_symlink():
        if image_alias.resolve() != image_dir:
            raise FileExistsError(
                f'{image_alias} points to {image_alias.resolve()}, '
                f'expected {image_dir}')
    elif image_alias.exists():
        raise FileExistsError(f'{image_alias} exists and is not a symlink')
    else:
        image_alias.symlink_to(image_dir, target_is_directory=True)
    image_prefix = f'{image_alias}/'
    cfg_options = [
        f'train_cfg.max_epochs={args.epochs}',
        f'train_dataloader.batch_size={args.batch_size}',
        f'train_dataloader.num_workers={args.num_workers}',
        f'train_dataloader.persistent_workers={args.num_workers > 0}',
        f'train_dataloader.dataset.ann_file={annotation_dir / "train.json"}',
        f'train_dataloader.dataset.data_prefix.img={image_prefix}',
        f'val_dataloader.dataset.ann_file={annotation_dir / "val.json"}',
        f'val_dataloader.dataset.data_prefix.img={image_prefix}',
        f'test_dataloader.dataset.ann_file={annotation_dir / "test.json"}',
        f'test_dataloader.dataset.data_prefix.img={image_prefix}',
        f'val_evaluator.ann_file={annotation_dir / "val.json"}',
        f'test_evaluator.ann_file={annotation_dir / "test.json"}',
        *args.cfg_options,
    ]
    train_command = [
        sys.executable,
        str(MMDET_ROOT / 'tools/train.py'),
        str(CONFIG),
        '--work-dir',
        str(args.work_dir.resolve()),
        '--cfg-options',
        *cfg_options,
    ]
    if args.amp:
        train_command.append('--amp')
    if args.auto_scale_lr:
        train_command.append('--auto-scale-lr')
    if args.resume:
        train_command.extend(['--resume', args.resume])

    if not args.test_only:
        run(train_command, args.dry_run)

    if not args.no_test:
        if args.dry_run:
            checkpoint = args.work_dir.resolve() / 'best_or_latest.pth'
        else:
            checkpoint = find_checkpoint(args.work_dir.resolve())
        result_dir = args.work_dir.resolve() / 'test_results'
        if not args.dry_run:
            result_dir.mkdir(parents=True, exist_ok=True)
        test_command = [
            sys.executable,
            str(MMDET_ROOT / 'tools/test.py'),
            str(CONFIG),
            str(checkpoint),
            '--work-dir',
            str(result_dir),
            '--out',
            str(result_dir / 'predictions.pkl'),
            '--cfg-options',
            *cfg_options,
        ]
        run(test_command, args.dry_run)

    upload_to_huggingface(args, args.work_dir.resolve())


if __name__ == '__main__':
    main()
