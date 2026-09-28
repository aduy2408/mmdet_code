#!/usr/bin/env python3
"""Run the DETR-R18 and RT-DETR-R18 matrix across two Marimo servers.

The runner is itself launched detached through ``utils.marimo_ops launch``.
Each server receives a deterministic shard of the 4 datasets x 2 models x 2
training seeds matrix. The dataset launchers perform hourly snapshots and final
Hugging Face verification for each job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
PYTHON = "/marimo/mmdet-venv/bin/python"
RTDETR_ROOT = "/marimo/rtdetr-mmdet"
# Recovery order: validate RT-DETR first, then run the DETR jobs that follow it.
MODELS = ("rtdetr_r18", "detr_r18")
SEEDS = (42, 43)
SPLIT_SEED = 42
EPOCHS = 100
PATIENCE = 15
UPLOAD_INTERVAL_HOURS = 1.0


@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    seed: int


DATASETS = ("varroa", "levirship", "tinyperson", "visdrone")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=int, choices=(1, 2), required=True)
    parser.add_argument("--all-jobs", action="store_true", help="Run every non-excluded matrix job sequentially on this host.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--hf-repo-id", default="duyle2408/detr_r18_rtdetr_r18_matrix_runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exclude-job", action="append", default=[], metavar="DATASET:MODEL:SEED")
    parser.add_argument("--work-root", default="/marimo/mmdet_code/work_dirs/detr_r18_matrix")
    parser.add_argument("--queue-state", default="/marimo/mmdet_code/work_dirs/detr_r18_matrix/queue_state.json")
    return parser.parse_args()


def jobs() -> list[Job]:
    return [Job(dataset, model, seed) for dataset in DATASETS for model in MODELS for seed in SEEDS]


def repo_ids() -> dict[str, str]:
    repo = "duyle2408/detr_r18_rtdetr_r18_matrix_runs"
    return {
        "varroa": repo,
        "levirship": repo,
        "tinyperson": repo,
        "visdrone": repo,
    }


def command_for(job: Job, args: argparse.Namespace) -> list[str]:
    base = [PYTHON]
    work_root = Path(args.work_root)
    repo = repo_ids()[job.dataset]
    common = [
        "--python", PYTHON,
        "--epochs", str(EPOCHS),
        "--split-seed", str(SPLIT_SEED),
        "--seed", str(job.seed),
        "--rtdetr-root", RTDETR_ROOT,
        "--hf-repo-id", repo,
        "--upload-interval-hours", str(UPLOAD_INTERVAL_HOURS),
    ]
    if job.dataset == "varroa":
        return base + [
            str(CODE_ROOT / "train_all_mmdet.py"),
            "--data-root", "/marimo/Varroa",
            "--dataset-out", str(work_root / "data" / "varroa_coco"),
            "--work-dir", str(work_root / "varroa" / f"seed{job.seed}"),
            "--models", job.model,
            "--variants", "base",
            "--yolo-protocol", "no_mosaic",
            "--early-stop-patience", str(PATIENCE),
            "--batch-size", "8",
            "--num-workers", "8",
            "--img-scale", "640", "640",
            "--remote-prefix", "detr_matrix/varroa",
            *common,
        ]
    if job.dataset == "levirship":
        return base + [
            str(CODE_ROOT / "train_all_levir_baseline.py"),
            "--data-root", "/marimo/LevirShip/LevirShipData",
            "--work-dir", str(work_root / "levirship" / f"seed{job.seed}"),
            "--models", job.model,
            "--epochs", str(EPOCHS),
            "--patience", str(PATIENCE),
            "--variant", "mosaic",
            "--batch-size", "4",
            "--num-workers", "4",
            "--image-size", "512",
            "--remote-prefix", "detr_matrix/levirship",
            *common,
        ]
    if job.dataset == "tinyperson":
        return base + [
            str(CODE_ROOT / "train_all_tinyperson_baseline.py"),
            "--data-root", "/marimo/TinyPerson",
            "--work-dir", str(work_root / "tinyperson" / f"seed{job.seed}"),
            "--prepared-ann-dir", str(work_root / "data" / "tinyperson" / f"seed{job.seed}"),
            "--models", job.model,
            "--epochs", str(EPOCHS),
            "--patience", str(PATIENCE),
            "--variant", "mosaic",
            "--batch-size", "2",
            "--num-workers", "4",
            "--image-size", "640",
            "--remote-prefix", "detr_matrix/tinyperson",
            *common,
        ]
    if job.dataset == "visdrone":
        return base + [
            str(CODE_ROOT / "train_visdrone_mmdet_baselines.py"),
            "--data-root", "/marimo/VisDrone2019",
            "--dataset-out", str(work_root / "data" / "visdrone2019_coco"),
            "--work-dir", str(work_root / "visdrone" / f"seed{job.seed}"),
            "--models", job.model,
            "--early-stop-patience", str(PATIENCE),
            "--lr", "0.01",
            "--batch-size", "8",
            "--workers", "8",
            "--image-size", "1536", "1536",
            "--remote-prefix", "detr_matrix/visdrone",
            *common,
        ]
    raise ValueError(job.dataset)


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    if args.epochs != EPOCHS or args.patience != PATIENCE or args.split_seed != SPLIT_SEED:
        raise ValueError("Queue settings must match the recorded matrix contract")
    if args.hf_repo_id != repo_ids()[DATASETS[0]]:
        raise ValueError(f"Unexpected HF repository: {args.hf_repo_id}")
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Launch this queue through python -m utils.marimo_ops launch")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be passed from the live Marimo global kernel")

    excluded = set(args.exclude_job)
    all_jobs = jobs()
    valid = {f"{job.dataset}:{job.model}:{job.seed}" for job in all_jobs}
    unknown = excluded - valid
    if unknown:
        raise ValueError(f"Unknown --exclude-job values: {sorted(unknown)}")
    assigned = [
        job for index, job in enumerate(all_jobs)
        if (args.all_jobs or index % 2 == args.machine - 1)
        and f"{job.dataset}:{job.model}:{job.seed}" not in excluded
    ]
    shard_label = "all" if args.all_jobs else str(args.machine)
    print(f"machine={shard_label} assigned={len(assigned)} jobs", flush=True)
    for job in assigned:
        print(f"  {job.dataset}/{job.model}/seed{job.seed}", flush=True)
    if args.dry_run:
        for job in assigned:
            print("RUN", " ".join(command_for(job, args)), flush=True)
        return

    state_path = Path(args.queue_state)
    state = {
        "experiment": "detr_r18_rtdetr_r18_matrix",
        "machine": "all" if args.all_jobs else args.machine,
        "split_seed": SPLIT_SEED,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "upload_interval_hours": UPLOAD_INTERVAL_HOURS,
        "jobs": [
            {"dataset": job.dataset, "model": job.model, "seed": job.seed, "status": "pending"}
            for job in assigned
        ],
    }
    write_state(state_path, state)
    env = os.environ.copy()
    env["MMDET_PYTHON"] = PYTHON
    env["PYTHONPATH"] = os.pathsep.join((str(CODE_ROOT), str(CODE_ROOT / "mmdetection")))
    for index, job in enumerate(assigned):
        state["jobs"][index]["status"] = "running"
        write_state(state_path, state)
        command = command_for(job, args)
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, cwd=CODE_ROOT, env=env, check=True)
        state["jobs"][index]["status"] = "completed"
        write_state(state_path, state)
    state["status"] = "completed"
    write_state(state_path, state)


if __name__ == "__main__":
    main()
