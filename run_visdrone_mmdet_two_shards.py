#!/usr/bin/env python3
"""Launch the VisDrone MMDetection matrix as two deterministic Molab shards.

This supervisor prepares one immutable Marimo contract per model/seed and
launches each job through ``utils.marimo_ops``. It never starts training when
HF upload context or preflight is unavailable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "train_visdrone_mmdet_baselines.py"
PYTHON = "/marimo/mmdet-venv/bin/python"
MODELS = ("fcos", "faster_rcnn", "atss", "cascade_rcnn", "rtmdet", "retinanet")
CONFIGS = {
    "fcos": "configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
    "faster_rcnn": "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    "atss": "configs/atss/atss_r50_fpn_1x_coco.py",
    "cascade_rcnn": "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
    "rtmdet": "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
    "retinanet": "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--data-root", default="/marimo/VisDrone2019")
    parser.add_argument("--dataset-out", default="/marimo/mmdet_code/work_dirs/visdrone2019_matrix/data")
    parser.add_argument("--work-dir", default="/marimo/mmdet_code/work_dirs/visdrone2019_matrix/runs")
    parser.add_argument("--control-root", default="/marimo/mmdet_code/work_dirs/visdrone2019_matrix/control")
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--remote-prefix", default="visdrone2019_mmdet_baselines")
    parser.add_argument("--seed-list", default="42,43")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def jobs(args: argparse.Namespace) -> list[tuple[str, int]]:
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seed_list.split(",") if item.strip()]
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    return [(model, seed) for seed in seeds for model in models]


def contract(args: argparse.Namespace, model: str, seed: int, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = Path(args.dataset_out) / "visdrone2019.yaml"
    payload = {
        "dataset": "visdrone2019-det",
        "data_root": args.data_root,
        "dataset_yaml": str(data_yaml),
        "model_yaml": str(ROOT / "mmdetection" / CONFIGS[model]),
        "seed": seed,
        "split_seed": args.split_seed,
        "workers": args.workers,
        "epochs": args.epochs,
        "patience": args.patience,
        "nms_iou": args.nms_iou,
        "hf_repo_id": args.hf_repo_id,
        "runner": str(RUNNER),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "amp": args.amp,
        "remote_prefix": f"{args.remote_prefix}/{model}/seed{seed}",
    }
    path = run_dir / "contract_input.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def launch(args: argparse.Namespace, model: str, seed: int) -> None:
    job_dir = Path(args.control_root) / f"{model}_seed{seed}"
    contract_path = contract(args, model, seed, job_dir)
    subprocess.run([
        PYTHON, "-m", "utils.marimo_ops", "contract",
        "--run-dir", str(job_dir), "--contract-json", str(contract_path),
    ], cwd=ROOT, check=True)
    subprocess.run([
        PYTHON, "-m", "utils.marimo_ops", "preflight",
        "--repo", str(ROOT), "--python", PYTHON,
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--upload-required", "--hf-repo-id", args.hf_repo_id,
        "--data-root", args.data_root, "--dataset-yaml", str(Path(args.dataset_out) / "visdrone2019.yaml"),
        "--contract", str(job_dir / "run_contract.json"),
        "--required-path", "train_visdrone_mmdet_baselines.py",
    ], cwd=ROOT, check=True)
    command = [
        PYTHON, str(RUNNER), "--python", PYTHON,
        "--data-root", args.data_root, "--dataset-out", args.dataset_out,
        "--work-dir", args.work_dir, "--models", model,
        "--model-yaml", str(ROOT / "mmdetection" / CONFIGS[model]),
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--lr", str(args.lr), "--batch-size", str(args.batch_size),
        "--workers", str(args.workers), "--image-size", *map(str, args.image_size),
        "--nms-iou", str(args.nms_iou), "--seed", str(seed),
        "--split-seed", str(args.split_seed), "--hf-repo-id", args.hf_repo_id,
        "--remote-prefix", args.remote_prefix,
    ]
    if args.amp:
        command.append("--amp")
    subprocess.run([
        PYTHON, "-m", "utils.marimo_ops", "launch",
        "--cwd", str(ROOT), "--run-dir", str(job_dir),
        "--artifact-root", str(Path(args.work_dir) / model / f"seed{seed}"),
        "--", *command,
    ], cwd=ROOT, check=True)
    print(f"LAUNCHED shard={args.shard} model={model} seed={seed}", flush=True)


def prepare_dataset(args: argparse.Namespace) -> None:
    subprocess.run([
        PYTHON, str(RUNNER), "--python", PYTHON,
        "--data-root", args.data_root, "--dataset-out", args.dataset_out,
        "--work-dir", args.work_dir, "--models", "fcos", "--seed", "42",
        "--split-seed", str(args.split_seed), "--no-copy-images", "--dry-run",
    ], cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    all_jobs = jobs(args)
    assigned = [job for index, job in enumerate(all_jobs) if index % 2 == args.shard]
    print(f"SHARD {args.shard}: {len(assigned)}/{len(all_jobs)} jobs", flush=True)
    prepare_dataset(args)
    for model, seed in assigned:
        launch(args, model, seed)


if __name__ == "__main__":
    main()
