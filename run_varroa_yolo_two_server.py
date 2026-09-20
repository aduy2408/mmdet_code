#!/usr/bin/env python3
"""Run the Varroa MMDetection YOLO-protocol matrix across two Marimo servers.

The matrix is four explicit MMDetection baselines x two augmentation protocols x
training seeds 42 and 43. Split seed remains fixed at 42. This runner only
starts when launched through utils.marimo_ops.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

MODELS = ("fcos", "faster_rcnn", "cascade_rcnn", "rtmdet")
PROTOCOLS = ("no_mosaic", "mosaic")
SEEDS = (42, 43)
SPLIT_SEED = 42


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=int, choices=(1, 2), required=True)
    parser.add_argument("--python", default="/marimo/mmdet-venv/bin/python")
    parser.add_argument("--code-root", type=Path, default=root)
    parser.add_argument("--data-root", default="/marimo/Varroa")
    parser.add_argument("--dataset-out", default="/marimo/mmdet_code/work_dirs/varroa_yolo_protocol/data/varroa_coco")
    parser.add_argument("--work-root", default="/marimo/mmdet_code/work_dirs/varroa_yolo_protocol")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", default="matrix", help="Matrix contract marker; per-job seeds are fixed to 42 and 43.")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-yaml", default="matrix")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--hf-repo-id",
        default="duyle2408/varroa_mmdet_yolo_protocol_runs",
    )
    parser.add_argument("--remote-prefix", default="varroa_yolo_protocol")
    parser.add_argument("--upload-interval-hours", type=float, default=1.0)
    parser.add_argument(
        "--exclude-job",
        action="append",
        default=[],
        metavar="PROTOCOL:SEED:MODEL",
        help="Skip a previously verified job, for example no_mosaic:42:fcos.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def jobs() -> list[tuple[str, str, int]]:
    return [
        (model, protocol, seed)
        for protocol in PROTOCOLS
        for model in MODELS
        for seed in SEEDS
    ]


def main() -> None:
    args = parse_args()
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Launch through python -m utils.marimo_ops launch")

    all_jobs = jobs()
    excluded = set(args.exclude_job)
    invalid = excluded - {f"{protocol}:{seed}:{model}" for model, protocol, seed in all_jobs}
    if invalid:
        raise ValueError(f"unknown --exclude-job values: {sorted(invalid)}")
    all_jobs = [
        job for job in all_jobs
        if f"{job[1]}:{job[2]}:{job[0]}" not in excluded
    ]
    assigned = [job for index, job in enumerate(all_jobs) if index % 2 == args.machine - 1]
    print(f"machine={args.machine} assigned={len(assigned)}/{len(all_jobs)} jobs")
    for model, protocol, seed in assigned:
        print(f"  {protocol}/seed{seed}/{model}")
    if args.list or args.dry_run:
        return

    code_root = args.code_root.resolve()
    runner = code_root / "train_all_mmdet.py"
    python = str(Path(args.python))
    env = os.environ.copy()
    env["MMDET_PYTHON"] = python
    venv_site = Path(python).resolve().parent.parent / "lib/python3.11/site-packages"
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(venv_site), str(code_root), str(code_root / "mmdetection")) if value
    )

    for model, protocol, seed in assigned:
        work_dir = Path(args.work_root) / f"{protocol}_seed{seed}"
        command = [
            python,
            str(runner),
            "--python", python,
            "--data-root", args.data_root,
            "--dataset-out", args.dataset_out,
            "--work-dir", str(work_dir),
            "--models", model,
            "--variants", "base",
            "--yolo-protocol", protocol,
            "--epochs", str(args.epochs),
            "--early-stop-patience", str(args.patience),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.num_workers),
            "--model-yaml", args.model_yaml,
            "--nms-iou", str(args.nms_iou),
            "--img-scale", str(args.image_size), str(args.image_size),
            "--split-seed", str(args.split_seed),
            "--seed", str(seed),
            "--hf-repo-id", args.hf_repo_id,
            "--remote-prefix", args.remote_prefix,
            "--upload-interval-hours", str(args.upload_interval_hours),
        ]
        if args.amp:
            command.append("--amp")
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, cwd=code_root, env=env, check=True)


if __name__ == "__main__":
    main()
