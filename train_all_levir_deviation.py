#!/usr/bin/env python3
"""Train the four FCOS downsampling ablations on LEVIR-Ship."""

from __future__ import annotations

import argparse

import train_all_levir_baseline as baseline


MODEL_CONFIGS = {
    "baseline": "configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
    "pixel_unshuffle":
        "configs/fcos/"
        "fcos_r50-caffe_fpn_gn-head_1x_levir_pixel-unshuffle.py",
    "context": "configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_levir_context.py",
    "deviation": "configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_levir_deviation.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="LevirShipData")
    parser.add_argument("--dataset-out", default="mmdetection/data/levir_ship_coco")
    parser.add_argument(
        "--work-dir", default="mmdetection/work_dirs/levir_deviation_ablation"
    )
    parser.add_argument(
        "--models",
        default="baseline,pixel_unshuffle,context,deviation",
        help="Comma-separated ablations to train.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument(
        "--hf-repo-id", default="duyle2408/levir_ship_dpd_ablation"
    )
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--no-hf-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_machines < 1:
        raise ValueError("--num-machines must be >= 1")
    if not 0 <= args.machine_index < args.num_machines:
        raise ValueError("--machine-index must be in [0, num_machines)")

    models = baseline.comma_list(args.models)
    unknown = sorted(set(models) - set(MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")

    # Reuse the mature dataset/config/train/test/upload path from the baseline
    # runner while swapping only its model table.
    baseline.MODEL_CONFIGS = MODEL_CONFIGS
    dataset_out, image_dir = baseline.prepare_coco_dataset(args)
    assigned = [
        model
        for index, model in enumerate(models)
        if index % args.num_machines == args.machine_index
    ]
    print(f"Assigned models ({args.machine_index}/{args.num_machines}): {assigned}")
    if args.dry_run:
        for model_name in assigned:
            config_path = baseline.write_config(
                model_name, args, dataset_out, image_dir
            )
            print(f"CONFIG {model_name}: {config_path}")
        return

    for model_name in assigned:
        baseline.run_job(model_name, args, dataset_out, image_dir)


if __name__ == "__main__":
    main()
