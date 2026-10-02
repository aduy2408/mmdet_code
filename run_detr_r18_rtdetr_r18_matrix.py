#!/usr/bin/env python3
"""Launch the canonical DETR-R18/RT-DETR-R18 matrix on two Marimo servers.

This is the single tracked launcher to sync to each Marimo checkout. It keeps
all dataset/model/augmentation settings explicit, writes one immutable manifest
per job, and shards the matrix deterministically across machine 1 and 2.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = os.environ.get(
    "MMDET_PYTHON",
    "/marimo/mmdet-venv/bin/python"
    if Path("/marimo/mmdet-venv/bin/python").is_file()
    else sys.executable,
)
DEFAULT_RTDETR_ROOT = "/marimo/rtdetr-mmdet"
DEFAULT_HF_REPO = "duyle2408/detr_r18_rtdetr_r18_matrix_runs"
MODELS = ("rtdetr_r18", "detr_r18")
DATASETS = ("varroa", "levirship", "tinyperson", "visdrone")
SEEDS = (42, 43)
SPLIT_SEED = 42
EPOCHS = 100
PATIENCE = 15
UPLOAD_INTERVAL_HOURS = 1.0


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    runner: str
    data_root: str
    batch_size: int
    workers: int
    image_size: tuple[int, int]
    variant: str
    protocol: str
    variants: tuple[str, ...]
    window_images: bool = False
    nms_iou: float = 0.5


DATASET_SPECS = {
    "varroa": DatasetSpec(
        name="varroa",
        runner="train_all_mmdet.py",
        data_root="/marimo/Varroa",
        batch_size=8,
        workers=8,
        image_size=(640, 640),
        variant="no_mosaic",
        protocol="YOLO-matched no_mosaic with MuSGD",
        variants=("no_mosaic", "mosaic"),
    ),
    "levirship": DatasetSpec(
        name="levirship",
        runner="train_all_levir_baseline.py",
        data_root="/marimo/LevirShip/LevirShipData",
        batch_size=4,
        workers=4,
        image_size=(512, 512),
        variant="mosaic",
        protocol="scene-safe LEVIR-Ship baseline with Mosaic then no-Mosaic switch",
        variants=("no_mosaic", "mosaic"),
    ),
    "tinyperson": DatasetSpec(
        name="tinyperson",
        runner="train_all_tinyperson_baseline.py",
        data_root="/marimo/TinyPerson",
        batch_size=8,
        workers=8,
        image_size=(640, 640),
        variant="mosaic",
        protocol="source-image split with erased training window images",
        variants=("no_mosaic", "mosaic"),
        window_images=True,
    ),
    "visdrone": DatasetSpec(
        name="visdrone",
        runner="train_visdrone_mmdet_baselines.py",
        data_root="/marimo/VisDrone2019",
        batch_size=8,
        workers=8,
        image_size=(1536, 1536),
        variant="musgd_default",
        protocol="official VisDrone train/val/test-dev COCO conversion",
        variants=("no_mosaic",),
    ),
}


@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    seed: int
    variant: str



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=int, choices=(1, 2), required=True)
    parser.add_argument("--all-jobs", action="store_true")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--rtdetr-root", default=DEFAULT_RTDETR_ROOT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO)
    parser.add_argument("--work-root", default="/marimo/mmdet_code/work_dirs/detr_r18_matrix")
    parser.add_argument("--queue-state", default="/marimo/mmdet_code/work_dirs/detr_r18_matrix/queue_state.json")
    parser.add_argument("--manifest-root", default="/marimo/mmdet_code/work_dirs/detr_r18_matrix/manifests")
    parser.add_argument("--remote-prefix", default="detr_matrix")
    parser.add_argument("--upload-interval-hours", type=float, default=UPLOAD_INTERVAL_HOURS)
    parser.add_argument("--exclude-job", action="append", default=[], metavar="DATASET:MODEL:SEED")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()



def jobs() -> list[Job]:
    return [
        Job(dataset, model, seed, variant)
        for dataset in DATASETS
        for variant in DATASET_SPECS[dataset].variants
        for model in MODELS
        for seed in SEEDS
    ]



def source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"



def worktree_dirty() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=CODE_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"



def validate_args(args: argparse.Namespace) -> None:
    if args.epochs != EPOCHS or args.patience != PATIENCE or args.split_seed != SPLIT_SEED:
        raise ValueError("Matrix contract requires epochs=100, patience=15, split_seed=42")
    if args.hf_repo_id != DEFAULT_HF_REPO:
        raise ValueError(f"Unexpected task-specific HF repository: {args.hf_repo_id}")
    if args.upload_interval_hours <= 0:
        raise ValueError("upload_interval_hours must be positive for required snapshots")
    if not args.dry_run and os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Launch the matrix through utils.marimo_ops.launch")
    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must come from the live Marimo global namespace")
    dirty = worktree_dirty()
    if not args.dry_run and dirty not in ("", "unknown") and not args.allow_dirty:
        raise RuntimeError("Refusing a dirty checkout. Commit/sync this launcher first or pass --allow-dirty.")



def validate_job(job: Job, args: argparse.Namespace) -> DatasetSpec:
    if job.dataset not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {job.dataset}")
    if job.model not in MODELS:
        raise ValueError(f"Unknown model: {job.model}")
    spec = DATASET_SPECS[job.dataset]
    if job.variant not in spec.variants:
        raise ValueError(
            f"Variant {job.variant!r} is not supported for {job.dataset}; "
            f"supported={spec.variants}"
        )
    if spec.window_images and job.dataset != "tinyperson":
        raise AssertionError("Only TinyPerson may use window_images")
    if job.dataset == "tinyperson" and not spec.window_images:
        raise AssertionError("TinyPerson must use window_images=true")
    return spec



def common_args(job: Job, args: argparse.Namespace) -> list[str]:
    return [
        "--python", args.python,
        "--epochs", str(args.epochs),
        "--split-seed", str(args.split_seed),
        "--seed", str(job.seed),
        "--rtdetr-root", args.rtdetr_root,
        "--hf-repo-id", args.hf_repo_id,
        "--upload-interval-hours", str(args.upload_interval_hours),
    ] + (["--amp"] if args.amp else [])



def command_for(job: Job, args: argparse.Namespace) -> list[str]:
    spec = validate_job(job, args)
    work_root = Path(args.work_root)
    work_dir = work_root / job.dataset / job.variant / f"{job.model}_seed{job.seed}"
    dataset_out = work_root / "data" / job.dataset
    command = [args.python, str(CODE_ROOT / spec.runner)]
    common = common_args(job, args)

    if job.dataset == "varroa":
        command += [
            "--data-root", spec.data_root,
            "--dataset-out", str(dataset_out),
            "--work-dir", str(work_dir),
            "--models", job.model,
            "--variants", "base",
            "--yolo-protocol", job.variant,
            "--early-stop-patience", str(args.patience),
            "--batch-size", str(spec.batch_size),
            "--num-workers", str(spec.workers),
            "--img-scale", *map(str, spec.image_size),
            "--remote-prefix", f"{args.remote_prefix}/{job.dataset}",
        ]
    elif job.dataset == "levirship":
        command += [
            "--data-root", spec.data_root,
            "--work-dir", str(work_dir),
            "--models", job.model,
            "--variant", job.variant,
            "--batch-size", str(spec.batch_size),
            "--num-workers", str(spec.workers),
            "--image-size", str(spec.image_size[0]),
            "--patience", str(args.patience),
            "--remote-prefix", f"{args.remote_prefix}/{job.dataset}",
        ]
    elif job.dataset == "tinyperson":
        command += [
            "--data-root", spec.data_root,
            "--work-dir", str(work_dir),
            "--prepared-ann-dir", str(dataset_out / f"seed{job.seed}"),
            "--models", job.model,
            "--variant", job.variant,
            "--batch-size", str(spec.batch_size),
            "--num-workers", str(spec.workers),
            "--image-size", str(spec.image_size[0]),
            "--patience", str(args.patience),
            "--remote-prefix", f"{args.remote_prefix}/{job.dataset}",
        ]
    elif job.dataset == "visdrone":
        command += [
            "--data-root", spec.data_root,
            "--dataset-out", str(dataset_out),
            "--work-dir", str(work_dir),
            "--models", job.model,
            "--epochs", str(args.epochs),
            "--early-stop-patience", str(args.patience),
            "--lr", "0.01",
            "--batch-size", str(spec.batch_size),
            "--workers", str(spec.workers),
            "--image-size", *map(str, spec.image_size),
            "--remote-prefix", f"{args.remote_prefix}/{job.dataset}",
            "--variant-name", "musgd_default",
        ]
    else:
        raise ValueError(job.dataset)
    return command + common



def manifest_for(job: Job, args: argparse.Namespace) -> dict[str, Any]:
    spec = validate_job(job, args)
    command = command_for(job, args)
    config = {
        "detr_r18": "mmdetection/configs/detr/detr_r18_8xb2-500e_coco.py",
        "rtdetr_r18": "external:flytocc/rtdetr-mmdet/configs/rtdetr/rtdetr_r18vd_8xb2-72e_coco.py",
    }[job.model]
    work_dir = str(
        Path(args.work_root) / job.dataset / job.variant / f"{job.model}_seed{job.seed}"
    )
    return {
        "experiment_id": f"detr_r18_rtdetr_r18_{job.dataset}_{job.variant}_{job.model}_seed{job.seed}",
        "baseline": {
            "control_config_or_baseline_config": config,
            "model": job.model,
            "backbone": "ResNet-18",
        },
        "variant": {
            "variant_config_or_explicit_change": "dataset adapter + fixed baseline protocol",
            "protocol": spec.protocol,
        },
        "source_commit": source_commit(),
        "runner": str(CODE_ROOT / spec.runner),
        "python_executable": args.python,
        "dataset": job.dataset,
        "dataset_root": spec.data_root,
        "split_seed": args.split_seed,
        "training_seed": job.seed,
        "model_backbone_pretrained_source": config,
        "image_size": list(spec.image_size),
        "batch_size": spec.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "workers": spec.workers,
        "amp": args.amp,
        "optimizer": {
            "type": "MuSGD",
            "lr": 0.01,
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay": 0.0005,
            "muon": 0.2,
            "sgd": 1.0,
        },
        "augmentation_variant": job.variant,
        "window_images": spec.window_images,
        "nms_iou": spec.nms_iou,
        "hf_repo": args.hf_repo_id,
        "remote_prefix": f"{args.remote_prefix}/{job.dataset}/{job.variant}/{job.model}/seed{job.seed}",
        "work_dir": work_dir,
        "required_artifacts": [
            "experiment_manifest.json",
            "patched_config.py",
            "checkpoint",
            "test_results",
            "final_results.json",
        ],
        "upload_required": True,
        "command": command,
    }



def write_manifests(jobs_to_write: list[Job], args: argparse.Namespace) -> None:
    root = Path(args.manifest_root)
    root.mkdir(parents=True, exist_ok=True)
    for job in jobs_to_write:
        path = root / f"{job.dataset}__{job.variant}__{job.model}__seed{job.seed}.json"
        path.write_text(json.dumps(manifest_for(job, args), indent=2) + "\n", encoding="utf-8")



def write_state(
    path: Path,
    assigned: list[Job],
    args: argparse.Namespace,
    statuses: list[str] | None = None,
    queue_status: str = "running",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    statuses = statuses or ["pending"] * len(assigned)
    payload = {
        "experiment": "detr_r18_rtdetr_r18_matrix",
        "source_commit": source_commit(),
        "machine": "all" if args.all_jobs else args.machine,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "optimizer": "MuSGD(lr=0.01, momentum=0.9, weight_decay=0.0005, muon=0.2, sgd=1.0)",
        "status": queue_status,
        "jobs": [asdict(job) | {"status": statuses[index]} for index, job in enumerate(assigned)],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")



def main() -> None:
    args = parse_args()
    validate_args(args)
    excluded = set(args.exclude_job)
    all_jobs = jobs()
    valid = {f"{job.dataset}:{job.variant}:{job.model}:{job.seed}" for job in all_jobs}
    unknown = excluded - valid
    if unknown:
        raise ValueError(f"Unknown --exclude-job values: {sorted(unknown)}")
    assigned = [
        job for index, job in enumerate(all_jobs)
        if (args.all_jobs or index % 2 == args.machine - 1)
        and f"{job.dataset}:{job.variant}:{job.model}:{job.seed}" not in excluded
    ]
    print(f"machine={'all' if args.all_jobs else args.machine} assigned={len(assigned)} jobs")
    for job in assigned:
        spec = validate_job(job, args)
        print(f"  {job.dataset}/{job.variant}/{job.model}/seed{job.seed} size={spec.image_size} batch={spec.batch_size} workers={spec.workers} window_images={spec.window_images}")
    write_manifests(assigned, args)
    if args.dry_run:
        for job in assigned:
            print("RUN", " ".join(command_for(job, args)))
        return
    state_path = Path(args.queue_state)
    statuses = ["pending"] * len(assigned)
    write_state(state_path, assigned, args, statuses)
    env = os.environ.copy()
    env["MMDET_PYTHON"] = args.python
    env["PYTHONPATH"] = os.pathsep.join((str(CODE_ROOT), str(CODE_ROOT / "mmdetection")))
    for index, job in enumerate(assigned):
        command = command_for(job, args)
        print("RUN", " ".join(command), flush=True)
        statuses[index] = "running"
        write_state(state_path, assigned, args, statuses)
        try:
            subprocess.run(command, cwd=CODE_ROOT, env=env, check=True)
        except Exception:
            statuses[index] = "failed"
            write_state(state_path, assigned, args, statuses, queue_status="failed")
            raise
        statuses[index] = "completed"
        write_state(state_path, assigned, args, statuses)
        print(f"COMPLETED {index + 1}/{len(assigned)} {job.dataset}/{job.variant}/{job.model}/seed{job.seed}", flush=True)
    write_state(state_path, assigned, args, statuses, queue_status="completed")


if __name__ == "__main__":
    main()
