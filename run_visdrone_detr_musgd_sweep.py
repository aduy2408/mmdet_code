#!/usr/bin/env python3
"""Run a bounded MuSGD hyperparameter sweep for DETR-R18 and RT-DETR-R18."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
PYTHON = "/marimo/mmdet-venv/bin/python"
DATA_ROOT = "/marimo/VisDrone2019"
RTDETR_ROOT = "/marimo/rtdetr-mmdet"
HF_REPO = "duyle2408/visdrone-detr-r18-musgd-sweep-runs"
SPLIT_SEED = 42
TRAINING_SEED = 42
EPOCHS = 30
PATIENCE = 0
IMAGE_SIZE = (1536, 1536)
BATCH_SIZE = 8
WORKERS = 8
NMS_IOU = 0.5
MODELS = ("detr_r18", "rtdetr_r18")


@dataclass(frozen=True)
class Variant:
    name: str
    lr: float
    clip_grad_max_norm: float


VARIANTS = (
    Variant("lr1e-3_clip1", 1e-3, 1.0),
    Variant("lr3e-4_clip1", 3e-4, 1.0),
    Variant("lr1e-3_clip0p1", 1e-3, 0.1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--rtdetr-root", default=RTDETR_ROOT)
    parser.add_argument("--work-root", default="/marimo/mmdet_runs/visdrone_detr_musgd_sweep_30e")
    parser.add_argument("--dataset-out", default="/marimo/mmdet_runs/visdrone_detr_musgd_sweep_30e/data/visdrone2019_coco")
    parser.add_argument("--queue-manifest", default="/marimo/mmdet_runs/visdrone_detr_musgd_sweep_30e/queue_manifest.json")
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variant", action="append", choices=[v.name for v in VARIANTS])
    parser.add_argument("--model", action="append", choices=list(MODELS))
    return parser.parse_args()


def selected_variants(args: argparse.Namespace) -> tuple[Variant, ...]:
    names = set(args.variant or [v.name for v in VARIANTS])
    return tuple(v for v in VARIANTS if v.name in names)


def selected_models(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.model or MODELS)


def command_for(args: argparse.Namespace, variant: Variant, model: str) -> list[str]:
    work_dir = Path(args.work_root) / variant.name
    remote_prefix = f"sweep30e/{variant.name}"
    return [
        PYTHON,
        str(CODE_ROOT / "train_visdrone_mmdet_baselines.py"),
        "--python", PYTHON,
        "--data-root", args.data_root,
        "--rtdetr-root", args.rtdetr_root,
        "--dataset-out", args.dataset_out,
        "--work-dir", str(work_dir),
        "--models", model,
        "--epochs", str(EPOCHS),
        "--early-stop-patience", str(PATIENCE),
        "--lr", str(variant.lr),
        "--optimizer", "musgd",
        "--clip-grad-max-norm", str(variant.clip_grad_max_norm),
        "--variant-name", variant.name,
        "--batch-size", str(BATCH_SIZE),
        "--workers", str(WORKERS),
        "--image-size", str(IMAGE_SIZE[0]), str(IMAGE_SIZE[1]),
        "--split-seed", str(SPLIT_SEED),
        "--seed", str(TRAINING_SEED),
        "--nms-iou", str(NMS_IOU),
        "--hf-repo-id", args.hf_repo_id,
        "--remote-prefix", remote_prefix,
        "--upload-interval-hours", "1.0",
    ]


def main() -> None:
    args = parse_args()
    variants = selected_variants(args)
    models = selected_models(args)
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1" and not args.dry_run:
        raise RuntimeError("Launch this sweep through python -m utils.marimo_ops launch")
    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be passed from the live Marimo global kernel")

    jobs = []
    for variant in variants:
        for model in models:
            jobs.append({
                "variant": asdict(variant),
                "model": model,
                "command": command_for(args, variant, model),
            })
    manifest = {
        "experiment_id": "visdrone_detr_r18_musgd_sweep_30e",
        "control_config_or_baseline_config": {
            "detr_r18": "configs/detr/detr_r18_8xb2-500e_coco.py",
            "rtdetr_r18": "configs/rtdetr/rtdetr_r18vd_8xb2-72e_coco.py",
        },
        "variant_config_or_explicit_change": "MuSGD sweep over LR and gradient clip, 30 epochs, no early stopping",
        "dataset_root": args.data_root,
        "split_seed": SPLIT_SEED,
        "training_seed": TRAINING_SEED,
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "amp": False,
        "nms_iou": NMS_IOU,
        "hf_repo": args.hf_repo_id,
        "upload_required": True,
        "jobs": jobs,
    }
    manifest_path = Path(args.queue_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "jobs": len(jobs), "upload_required": True}, indent=2), flush=True)
    for job in jobs:
        command = job["command"]
        print("RUN", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=CODE_ROOT, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
