#!/usr/bin/env python3
"""Train and compare Faster R-CNN P2 baseline and PAHR-P2."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import train_all_levir_baseline as levir


VARIANTS = {
    'faster_rcnn_p2_768': dict(
        config='configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_1x_coco.py',
        pahr=False),
    'faster_rcnn_pahr_p2_768': dict(
        config='projects/pahr/configs/faster_rcnn_pahr_p2.py',
        pahr=True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='LevirShipData')
    parser.add_argument(
        '--dataset-out', default='mmdetection/data/levir_ship_coco')
    parser.add_argument(
        '--work-dir', default='mmdetection/work_dirs/levir_faster_rcnn_p2')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=768, choices=(512, 768))
    parser.add_argument(
        '--variants',
        default=','.join(VARIANTS),
        help=f"Comma-separated variants: {', '.join(VARIANTS)}.")
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--hf-repo-id', default='duyle2408/fcos_test_haar')
    parser.add_argument('--hf-repo-type', default='dataset')
    parser.add_argument('--hf-token', default='')
    parser.add_argument(
        '--skip-upload', '--no-hf-upload', dest='no_hf_upload',
        action='store_true')
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


def write_config(variant: str, args: argparse.Namespace,
                 dataset_out: Path, image_dir: Path) -> Path:
    root = str(levir.mmdet_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from mmengine.config import Config
    from mmdet.utils import register_all_modules

    register_all_modules()
    settings = VARIANTS[variant]
    cfg = Config.fromfile(str(levir.mmdet_root() / settings['config']))
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
    cfg.default_hooks.checkpoint.update(
        save_best='coco/bbox_mAP_75', rule='greater')
    if settings['pahr']:
        paramwise = cfg.optim_wrapper.get('paramwise_cfg', {})
        custom_keys = dict(paramwise.get('custom_keys', {}))
        custom_keys['neck.detail_mixer.3'] = dict(
            lr_mult=10.0, decay_mult=0.0)
        paramwise['custom_keys'] = custom_keys
        cfg.optim_wrapper.paramwise_cfg = paramwise
    cfg.p2_variant = dict(
        name=variant,
        image_size=args.image_size,
        feature_strides=[4, 8, 16, 32, 64],
        pahr=settings['pahr'],
        detail_lr_mult=10.0 if settings['pahr'] else 1.0)
    output = Path(cfg.work_dir) / 'patched_config.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    return output


def tiny_summary(config: Path, predictions: Path) -> dict:
    from mmengine.config import Config
    from projects.pahr.tools.summarize_pahr import tiny_statistics

    cfg = Config.fromfile(str(config))
    return tiny_statistics(
        Path(cfg.test_dataloader.dataset.ann_file), predictions, 16.0)


def correction_summary(config: Path, checkpoint: Path) -> dict | None:
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules
    from projects.pahr import PAHRFPN

    # MMEngine checkpoints contain trusted HistoryBuffer objects, while
    # PyTorch 2.6 defaults torch.load() to weights_only=True.
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    cfg = Config.fromfile(str(config))
    if cfg.model.neck.type != 'PAHRFPN':
        return None
    register_all_modules()
    model = init_detector(cfg, str(checkpoint), device='cuda:0')
    loader = Runner.build_dataloader(cfg.val_dataloader)
    data = next(iter(loader))
    batch = model.data_preprocessor(data, training=False)
    import torch
    with torch.inference_mode():
        backbone = model.backbone(batch['inputs'])
        baseline = super(PAHRFPN, model.neck).forward(backbone)
        enhanced, aux = model.neck.forward_with_aux(backbone)
        correction = enhanced[0] - baseline[0]
    return dict(
        p2_correction_ratio=float(
            correction.norm() / baseline[0].norm().clamp_min(1e-12)),
        p2_correction_abs_mean=float(correction.abs().mean()),
        raw_correction_rms=float(aux['raw_correction_rms']),
        applied_correction_rms=float(aux['applied_correction_rms']))


def write_summary(variant: str, config: Path, checkpoint: Path,
                  predictions: Path, elapsed: float) -> None:
    from projects.pahr.tools.summarize_pahr import load_coco_metrics

    work_dir = config.parent
    scalars = work_dir / 'vis_data/scalars.json'
    metrics = []
    if scalars.exists():
        metrics = [
            json.loads(line) for line in scalars.read_text().splitlines()
            if 'coco/bbox_mAP_75' in line]
    best = max(
        metrics, key=lambda row: row.get('coco/bbox_mAP_75', -1),
        default={})
    with predictions.open('rb') as stream:
        image_count = len(pickle.load(stream))
    summary = dict(
        variant=variant,
        validation={
            key: best.get(key) for key in (
                'coco/bbox_mAP', 'coco/bbox_mAP_50',
                'coco/bbox_mAP_75')},
        test=load_coco_metrics(predictions),
        tiny=tiny_summary(config, predictions),
        correction=correction_summary(config, checkpoint),
        test_latency_seconds=elapsed,
        test_latency_ms_per_image=elapsed * 1000 / max(image_count, 1))
    output = work_dir / 'test_results/p2_summary.json'
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')


def upload(variant: str, args: argparse.Namespace) -> None:
    if args.no_hf_upload:
        return
    token = args.hf_token or os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError('upload requires --hf-token or HF_TOKEN')
    from huggingface_hub import HfApi
    work_dir = levir.resolve_path(args.work_dir) / variant
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.hf_repo_id, repo_type=args.hf_repo_type,
        exist_ok=True)
    api.upload_folder(
        folder_path=str(work_dir), path_in_repo=variant,
        repo_id=args.hf_repo_id, repo_type=args.hf_repo_type)


def run_variant(variant: str, args: argparse.Namespace,
                dataset_out: Path, image_dir: Path) -> None:
    config = write_config(variant, args, dataset_out, image_dir)
    work_dir = levir.resolve_path(args.work_dir) / variant
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
    started = time.perf_counter()
    levir.run([
        sys.executable, str(levir.mmdet_root() / 'tools/test.py'),
        str(config), str(checkpoint), '--work-dir',
        str(predictions.parent), '--out', str(predictions)])
    write_summary(
        variant, config, checkpoint, predictions,
        time.perf_counter() - started)
    upload(variant, args)


def main() -> None:
    args = parse_args()
    variants = levir.comma_list(args.variants)
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    dataset_out, image_dir = levir.prepare_coco_dataset(args)
    for variant in variants:
        if args.dry_run:
            print('CONFIG', variant, write_config(
                variant, args, dataset_out, image_dir))
        else:
            run_variant(variant, args, dataset_out, image_dir)


if __name__ == '__main__':
    main()
