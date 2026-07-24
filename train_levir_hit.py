#!/usr/bin/env python3
"""Run the gated FCOS-P3 HIT ablation pipeline on LEVIR-Ship."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from train_all_levir_baseline import prepare_coco_dataset


ROOT = Path(__file__).resolve().parent
MMDET_ROOT = ROOT / 'mmdetection'
CONFIG_DIR = MMDET_ROOT / 'configs/hit'
SPARSE_CONFIG = CONFIG_DIR / (
    'fcos_r50-caffe_fpn-hit_12e_levir-ship.py')
PROBE_CONFIG = CONFIG_DIR / (
    'fcos_r50-caffe_fpn-hit-probe_levir-ship.py')
WARMUP_CONFIG = CONFIG_DIR / (
    'fcos_r50-caffe_fpn-hit-warmup_levir-ship.py')
JOINT_CONFIGS = {
    42: CONFIG_DIR / 'fcos_r50-caffe_fpn-hit-joint_levir-ship.py',
    43: CONFIG_DIR / 'fcos_r50-caffe_fpn-hit-joint-seed43_levir-ship.py',
}
PROBE_SCRIPT = MMDET_ROOT / 'tools/analysis_tools/hit_probe.py'
BASELINE_CHECKPOINT = ROOT / 'best_coco_bbox_mAP_epoch_12.pth'
WORK_DIR = MMDET_ROOT / 'work_dirs/levir_hit/pipeline'
DATA_ROOT = ROOT / 'LevirShipData'
DATASET_OUT = MMDET_ROOT / 'data/levir_ship_coco'
MAP_PATTERN = re.compile(
    r'coco/bbox_mAP(?![_A-Za-z0-9])["\']?\s*[:=]\s*'
    r'([0-9]*\.?[0-9]+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--baseline-checkpoint', type=Path, default=BASELINE_CHECKPOINT)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--dataset-out', type=Path, default=DATASET_OUT)
    parser.add_argument('--dataset-seed', type=int, default=42)
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--probe-epochs', type=int, default=3)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument(
        '--max-ap-drop',
        type=float,
        default=0.002,
        help='Largest accepted validation mAP drop in raw COCO units. '
        '0.002 equals 0.2 AP points.')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--auto-scale-lr', action='store_true')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume a stage automatically when its work directory exists.')
    parser.add_argument(
        '--force-continue',
        action='store_true',
        help='Continue even when the H or validation AP gate fails.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        default=[],
        help='Extra MMEngine overrides applied to every train/test stage.')
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip final test-set evaluation; validation gates still run.')
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
        help='Do not upload the completed pipeline directory.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print every command without running it or writing outputs.')
    return parser.parse_args()


def run(command: list[str], dry_run: bool = False) -> str:
    print('\n$', ' '.join(command), flush=True)
    if dry_run:
        return ''
    env = os.environ.copy()
    env.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    process = subprocess.Popen(
        command,
        cwd=MMDET_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    assert process.stdout is not None
    lines = []
    for line in process.stdout:
        print(line, end='', flush=True)
        lines.append(line)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return ''.join(lines)


def find_checkpoint(work_dir: Path) -> Path:
    best = sorted(work_dir.glob('best_*.pth'))
    if best:
        return best[0]
    latest = work_dir / 'latest.pth'
    if latest.is_file():
        return latest.resolve()
    epochs = sorted(
        work_dir.glob('epoch_*.pth'),
        key=lambda path: int(path.stem.rsplit('_', 1)[-1]))
    if epochs:
        return epochs[-1]
    raise FileNotFoundError(
        f'No best_*.pth, latest.pth, or epoch_*.pth found in {work_dir}')


def checkpoint_after(work_dir: Path, dry_run: bool) -> Path:
    return (work_dir / 'best_or_latest.pth'
            if dry_run else find_checkpoint(work_dir))


def prepare_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = args.data_root.resolve()
    dataset_out = args.dataset_out.resolve()
    image_dir = data_root / 'All Images'
    annotation_dir = dataset_out / 'annotations'
    annotations = [
        annotation_dir / f'{split}.json'
        for split in ('train', 'val', 'test')
    ]
    if not all(path.is_file() for path in annotations):
        if args.dry_run:
            raise FileNotFoundError(
                'Dry-run cannot prepare missing annotations: ' +
                ', '.join(str(path) for path in annotations
                          if not path.is_file()))
        print(f'Preparing COCO annotations in {dataset_out}')
        prepare_coco_dataset(SimpleNamespace(
            data_root=data_root,
            dataset_out=dataset_out,
            seed=args.dataset_seed,
            limit=0,
        ))

    required = [
        args.baseline_checkpoint.resolve(),
        image_dir,
        PROBE_SCRIPT,
        PROBE_CONFIG,
        WARMUP_CONFIG,
        SPARSE_CONFIG,
        *JOINT_CONFIGS.values(),
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
    elif not args.dry_run:
        image_alias.symlink_to(image_dir, target_is_directory=True)
    return annotation_dir, image_alias


def common_cfg_options(args: argparse.Namespace, annotation_dir: Path,
                       image_alias: Path) -> list[str]:
    persistent = args.num_workers > 0
    return [
        f'train_dataloader.batch_size={args.batch_size}',
        f'train_dataloader.num_workers={args.num_workers}',
        f'train_dataloader.persistent_workers={persistent}',
        f'train_dataloader.dataset.ann_file={annotation_dir / "train.json"}',
        f'train_dataloader.dataset.data_prefix.img={image_alias}/',
        f'val_dataloader.dataset.ann_file={annotation_dir / "val.json"}',
        f'val_dataloader.dataset.data_prefix.img={image_alias}/',
        f'test_dataloader.dataset.ann_file={annotation_dir / "test.json"}',
        f'test_dataloader.dataset.data_prefix.img={image_alias}/',
        f'val_evaluator.ann_file={annotation_dir / "val.json"}',
        f'test_evaluator.ann_file={annotation_dir / "test.json"}',
        *args.cfg_options,
    ]


def train_stage(args: argparse.Namespace, name: str, config: Path,
                work_dir: Path, checkpoint: Path, epochs: int,
                cfg_options: list[str]) -> Path:
    print(f'\n=== TRAIN {name} ===')
    command = [
        sys.executable,
        str(MMDET_ROOT / 'tools/train.py'),
        str(config),
        '--work-dir',
        str(work_dir),
        '--cfg-options',
        f'load_from={checkpoint}',
        f'train_cfg.max_epochs={epochs}',
        *cfg_options,
    ]
    if args.amp:
        command.append('--amp')
    if args.auto_scale_lr:
        command.append('--auto-scale-lr')
    if args.resume and not args.dry_run:
        try:
            resume_checkpoint = find_checkpoint(work_dir)
        except FileNotFoundError:
            resume_checkpoint = None
        if resume_checkpoint is not None:
            command.extend(['--resume', str(resume_checkpoint)])
    run(command, args.dry_run)
    return checkpoint_after(work_dir, args.dry_run)


def parse_map(output: str) -> float:
    matches = MAP_PATTERN.findall(output)
    if not matches:
        raise RuntimeError('Could not find coco/bbox_mAP in test output.')
    return float(matches[-1])


def evaluate_stage(args: argparse.Namespace, name: str, config: Path,
                   checkpoint: Path, work_dir: Path, split: str,
                   annotation_dir: Path,
                   image_alias: Path) -> float | None:
    print(f'\n=== EVALUATE {name} [{split}] ===')
    result_dir = work_dir / f'{split}_results'
    annotation = annotation_dir / f'{split}.json'
    command = [
        sys.executable,
        str(MMDET_ROOT / 'tools/test.py'),
        str(config),
        str(checkpoint),
        '--work-dir',
        str(result_dir),
        '--out',
        str(result_dir / 'predictions.pkl'),
        '--cfg-options',
        f'test_dataloader.dataset.ann_file={annotation}',
        f'test_dataloader.dataset.data_prefix.img={image_alias}/',
        f'test_evaluator.ann_file={annotation}',
        *args.cfg_options,
    ]
    output = run(command, args.dry_run)
    return None if args.dry_run else parse_map(output)


def run_probe(args: argparse.Namespace, probe_dir: Path,
              cfg_options: list[str]) -> bool:
    print('\n=== PROBE H ===')
    command = [
        sys.executable,
        str(PROBE_SCRIPT),
        str(PROBE_CONFIG),
        str(args.baseline_checkpoint.resolve()),
        '--output-dir',
        str(probe_dir),
        '--epochs',
        str(args.probe_epochs),
        '--seed',
        str(args.dataset_seed),
        '--cfg-options',
        *cfg_options,
    ]
    run(command, args.dry_run)
    if args.dry_run:
        return True
    report = json.loads(
        (probe_dir / 'metrics.json').read_text(encoding='utf-8'))
    return bool(report['gate_passed'])


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


def require_gate(passed: bool, message: str,
                 force_continue: bool) -> None:
    if passed:
        print(f'PASS: {message}')
        return
    print(f'FAIL: {message}')
    if not force_continue:
        raise RuntimeError(
            f'{message}. Pass --force-continue to override this gate.')
    print('Continuing because --force-continue was supplied.')


def main() -> None:
    args = parse_args()
    positive = (
        args.probe_epochs, args.warmup_epochs, args.epochs, args.batch_size)
    if any(value < 1 for value in positive) or args.num_workers < 0:
        raise ValueError(
            'epochs and batch-size must be positive; workers must be >= 0.')
    if args.max_ap_drop < 0:
        raise ValueError('max-ap-drop must be non-negative.')

    annotation_dir, image_alias = prepare_inputs(args)
    cfg_options = common_cfg_options(
        args, annotation_dir, image_alias)
    pipeline_dir = args.work_dir.resolve()
    if not args.dry_run:
        pipeline_dir.mkdir(parents=True, exist_ok=True)

    probe_passed = run_probe(args, pipeline_dir / 'probe', cfg_options)
    require_gate(
        probe_passed, 'H localization gate', args.force_continue)

    baseline = args.baseline_checkpoint.resolve()
    warmup_checkpoint = train_stage(
        args,
        'sparse warm-up',
        WARMUP_CONFIG,
        pipeline_dir / 'warmup',
        baseline,
        args.warmup_epochs,
        cfg_options)
    detached_checkpoint = train_stage(
        args,
        'sparse detached seed 42',
        SPARSE_CONFIG,
        pipeline_dir / 'detached_seed42',
        warmup_checkpoint,
        args.epochs,
        cfg_options)

    baseline_val = evaluate_stage(
        args, 'baseline', SPARSE_CONFIG, baseline,
        pipeline_dir / 'baseline', 'val', annotation_dir, image_alias)
    detached_val = evaluate_stage(
        args, 'sparse detached seed 42', SPARSE_CONFIG, detached_checkpoint,
        pipeline_dir / 'detached_seed42', 'val', annotation_dir, image_alias)
    ap_gate = (args.dry_run or
               detached_val >= baseline_val - args.max_ap_drop)
    require_gate(
        ap_gate,
        f'detached validation mAP drop <= {args.max_ap_drop}',
        args.force_continue)

    joint_checkpoints = {}
    for seed, config in JOINT_CONFIGS.items():
        joint_checkpoints[seed] = train_stage(
            args,
            f'joint seed {seed}',
            config,
            pipeline_dir / f'joint_seed{seed}',
            detached_checkpoint,
            args.epochs,
            cfg_options)

    summary = {
        'baseline_checkpoint': str(baseline),
        'probe_gate_passed': probe_passed,
        'validation': {
            'baseline_bbox_mAP': baseline_val,
            'detached_bbox_mAP': detached_val,
            'max_ap_drop': args.max_ap_drop,
            'gate_passed': ap_gate,
        },
        'checkpoints': {
            'warmup': str(warmup_checkpoint),
            'detached_seed42': str(detached_checkpoint),
            **{
                f'joint_seed{seed}': str(checkpoint)
                for seed, checkpoint in joint_checkpoints.items()
            },
        },
    }

    if not args.no_test:
        test_scores = {
            'baseline': evaluate_stage(
                args, 'baseline', SPARSE_CONFIG, baseline,
                pipeline_dir / 'baseline', 'test', annotation_dir,
                image_alias),
            'detached_seed42': evaluate_stage(
                args, 'sparse detached seed 42', SPARSE_CONFIG,
                detached_checkpoint, pipeline_dir / 'detached_seed42',
                'test', annotation_dir, image_alias),
        }
        for seed, checkpoint in joint_checkpoints.items():
            test_scores[f'joint_seed{seed}'] = evaluate_stage(
                args, f'joint seed {seed}', JOINT_CONFIGS[seed], checkpoint,
                pipeline_dir / f'joint_seed{seed}', 'test', annotation_dir,
                image_alias)
        if not args.dry_run:
            joint_scores = [
                test_scores['joint_seed42'],
                test_scores['joint_seed43'],
            ]
            test_success = (
                sum(joint_scores) / len(joint_scores) >
                test_scores['baseline'] and
                all(score >= test_scores['baseline'] - args.max_ap_drop
                    for score in joint_scores))
            test_scores['success'] = test_success
        summary['test'] = test_scores

    if not args.dry_run:
        (pipeline_dir / 'pipeline_summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8')
        print('\n=== PIPELINE SUMMARY ===')
        print(json.dumps(summary, indent=2))
    upload_to_huggingface(args, pipeline_dir)


if __name__ == '__main__':
    main()
