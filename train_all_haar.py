#!/usr/bin/env python3
"""Prepare LEVIR-Ship, train/test PAHR, and upload the run to Hugging Face."""

from __future__ import annotations

import argparse

import train_all_levir_baseline as levir


MODEL_NAME = 'haar'
MODEL_CONFIG = 'projects/pahr/configs/fcos_pahr.py'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='LevirShipData')
    parser.add_argument(
        '--dataset-out', default='mmdetection/data/levir_ship_coco')
    parser.add_argument(
        '--work-dir', default='mmdetection/work_dirs/levir_haar')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
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
        '--hf-repo-id', default='duyle2408/levir_ship_mmdet_runs')
    parser.add_argument('--hf-repo-type', default='dataset')
    parser.add_argument(
        '--hf-token',
        default='',
        help='Hugging Face token; defaults to the HF_TOKEN environment variable.')
    parser.add_argument(
        '--no-hf-upload',
        action='store_true',
        help='Skip Hugging Face upload.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levir.MODEL_CONFIGS[MODEL_NAME] = MODEL_CONFIG
    dataset_out, image_dir = levir.prepare_coco_dataset(args)
    config_path = levir.write_config(
        MODEL_NAME, args, dataset_out, image_dir)
    if args.dry_run:
        print(f'CONFIG {MODEL_NAME}: {config_path}')
        return
    levir.run_job(MODEL_NAME, args, dataset_out, image_dir)


if __name__ == '__main__':
    main()
