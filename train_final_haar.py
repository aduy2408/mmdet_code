#!/usr/bin/env python3
"""Run the final C2-to-P3 Haar fusion experiment on LEVIR-Ship."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import train_all_levir_baseline as levir


VARIANT = 'haar_c2_wavelet_fusion_768'
MODEL_CONFIG = 'projects/pahr/configs/fcos_haar_c2_fusion.py'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='LevirShipData')
    parser.add_argument(
        '--dataset-out', default='mmdetection/data/levir_ship_coco')
    parser.add_argument(
        '--work-dir',
        default='mmdetection/work_dirs/levir_final_haar')
    parser.add_argument('--image-size', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--hf-repo-id', default='duyle2408/fcos_test_haar')
    parser.add_argument('--hf-repo-type', default='dataset')
    parser.add_argument('--hf-token', default='')
    parser.add_argument('--skip-upload', action='store_true')
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


def write_config(args: argparse.Namespace, dataset_out: Path,
                 image_dir: Path) -> Path:
    root = str(levir.mmdet_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from mmengine.config import Config
    from mmdet.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(str(levir.mmdet_root() / MODEL_CONFIG))
    cfg = levir.patch_config(cfg, VARIANT, args, dataset_out, image_dir)
    for dataloader in (
            cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        set_resize_scale(dataloader.dataset.pipeline, args.image_size)
    for scheduler in cfg.param_scheduler:
        if scheduler.type == 'MultiStepLR':
            scheduler.end = args.epochs
            scheduler.milestones = [
                round(args.epochs * 2 / 3),
                round(args.epochs * 11 / 12)]
    paramwise = cfg.optim_wrapper.get('paramwise_cfg', {})
    custom_keys = dict(paramwise.get('custom_keys', {}))
    custom_keys['neck.fusion_mixer.3'] = dict(
        lr_mult=10.0, decay_mult=0.0)
    paramwise['custom_keys'] = custom_keys
    cfg.optim_wrapper.paramwise_cfg = paramwise
    cfg.default_hooks.checkpoint.update(
        save_best='coco/bbox_mAP_75', rule='greater')
    cfg.final_haar = dict(
        variant=VARIANT,
        image_size=args.image_size,
        feature_strides=[8, 16, 32, 64, 128],
        fusion='C2 Haar bands -> P3',
        fusion_output_lr_mult=10.0)
    output = Path(cfg.work_dir) / 'patched_config.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    return output


def correction_summary(config: Path, checkpoint: Path) -> dict:
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(str(config))
    model = init_detector(cfg, str(checkpoint), device='cuda:0')
    loader = Runner.build_dataloader(cfg.val_dataloader)
    batch = model.data_preprocessor(next(iter(loader)), training=False)
    with torch.inference_mode():
        backbone = model.backbone(batch['inputs'])
        _, aux = model.neck.forward_with_aux(backbone)
    return dict(
        band_rms=[float(value) for value in aux['band_rms']],
        correction_rms=float(aux['correction_rms']),
        correction_ratio=float(aux['correction_ratio']),
        baseline_p3_rms=float(aux['baseline_p3_rms']))


def write_summary(config: Path, checkpoint: Path, predictions: Path,
                  elapsed: float) -> Path:
    from projects.pahr.tools.summarize_pahr import (
        load_coco_metrics, tiny_statistics)
    from mmengine.config import Config

    cfg = Config.fromfile(str(config))
    with predictions.open('rb') as stream:
        image_count = len(pickle.load(stream))
    summary = dict(
        variant=VARIANT,
        test=load_coco_metrics(predictions),
        tiny=tiny_statistics(
            Path(cfg.test_dataloader.dataset.ann_file), predictions, 16.0),
        correction=correction_summary(config, checkpoint),
        test_latency_seconds=elapsed,
        test_latency_ms_per_image=elapsed * 1000 / max(image_count, 1),
        baseline=dict(mAP=0.272, AP50=0.735, AP75=0.100))
    output = config.parent / 'test_results/final_haar_summary.json'
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return output


def upload(args: argparse.Namespace) -> None:
    if args.skip_upload:
        return
    token = args.hf_token or os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError('upload requires --hf-token or HF_TOKEN')
    from huggingface_hub import HfApi
    folder = levir.resolve_path(args.work_dir) / VARIANT
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.hf_repo_id, repo_type=args.hf_repo_type,
        exist_ok=True)
    api.upload_folder(
        folder_path=str(folder), path_in_repo=VARIANT,
        repo_id=args.hf_repo_id, repo_type=args.hf_repo_type)


def main() -> None:
    args = parse_args()
    dataset_out, image_dir = levir.prepare_coco_dataset(args)
    config = write_config(args, dataset_out, image_dir)
    print('CONFIG', config)
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
    started = time.perf_counter()
    levir.run([
        sys.executable, str(levir.mmdet_root() / 'tools/test.py'),
        str(config), str(checkpoint), '--work-dir', str(predictions.parent),
        '--out', str(predictions)])
    print('SUMMARY', write_summary(
        config, checkpoint, predictions, time.perf_counter() - started))
    upload(args)


if __name__ == '__main__':
    main()
