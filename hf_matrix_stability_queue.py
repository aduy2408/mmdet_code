#!/usr/bin/env python3
"""Recover the HF matrix with an explicit DETR stability learning-rate variant."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.marimo_ops import launch_detached, status, write_run_contract


@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    variant: str
    seed: int


DATASETS = {
    "varroa": {"root": "/marimo/Varroa", "image_size": 640, "batch_size": 8, "workers": 8},
    "levirship": {"root": "/marimo/LevirShip/LevirShipData", "image_size": 512, "batch_size": 4, "workers": 4},
    "tinyperson": {"root": "/marimo/TinyPerson", "image_size": 640, "batch_size": 8, "workers": 8},
    "visdrone": {"root": "/marimo/VisDrone2019", "image_size": 1536, "batch_size": 8, "workers": 8},
}
MODELS = {
    "detr_r50": {"model_id": "facebook/detr-resnet-50", "backbone": "ResNet-50", "lr": 0.001},
    "rtdetr_r18": {"model_id": "PekingU/rtdetr_r18vd", "backbone": "ResNet-18", "lr": 0.01},
}
REPO_ID = "duyle2408/detr_r18_rtdetr_r18_matrix_runs"


def all_jobs() -> list[Job]:
    return [
        Job(dataset, model, variant, seed)
        for dataset in DATASETS
        for variant in ("no_mosaic", "mosaic")
        for model in MODELS
        for seed in (42, 43)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--work-root", default="/marimo/hf_runs/stability_matrix")
    parser.add_argument("--state-file", default="/marimo/hf_runs/stability_matrix/queue_state.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def baseline_evidence(server: int, job: Job) -> tuple[bool, str]:
    """Return true only for a completed upload with no skipped training batches."""
    relative = Path(job.dataset) / job.variant / f"{job.model}_seed{job.seed}"
    for root_name in (f"server{server}", f"server{server}_retry"):
        run = Path("/marimo/hf_runs/full_matrix") / root_name / relative
        marker = run / "upload_complete.json"
        log = run / "train.log"
        if marker.is_file() and log.is_file():
            text = log.read_text(errors="replace")
            if "skip_batch" not in text:
                return True, str(run)
    return False, ""


def write_state(path: Path, server: int, jobs: list[Job], statuses: list[str], active: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "experiment": "hf_detr_r50_rtdetr_r18_matrix_stability_lr",
        "server": server,
        "epochs": 100,
        "patience": 15,
        "split_seed": 42,
        "hf_repo_id": REPO_ID,
        "upload_required": True,
        "stability_variant": "detr_r50_lr=0.001; rtdetr_r18_lr=0.01",
        "active": active,
        "jobs": [asdict(job) | {"status": statuses[i]} for i, job in enumerate(jobs)],
    }, indent=2) + "\n", encoding="utf-8")


def run_job(job: Job, args, repo: Path, run_root: Path):
    spec = DATASETS[job.dataset]
    model = MODELS[job.model]
    run = run_root / job.dataset / job.variant / f"{job.model}_seed{job.seed}"
    run.mkdir(parents=True, exist_ok=True)
    contract = run / "run_contract.json"
    lr = model["lr"]
    if not contract.exists():
        write_run_contract(run, {
            "dataset": job.dataset,
            "data_root": spec["root"],
            "dataset_yaml": "matrix",
            "model_yaml": model["model_id"],
            "seed": job.seed,
            "split_seed": args.split_seed,
            "workers": spec["workers"],
            "epochs": args.epochs,
            "patience": args.patience,
            "nms_iou": None,
            "hf_repo_id": REPO_ID,
            "model": job.model,
            "backbone": model["backbone"],
            "variant": job.variant,
            "optimizer": f"MuSGD(lr={lr},momentum=0.9,weight_decay=0.0005,muon=0.2,sgd=1.0)",
            "stability_variant": "explicit_lr_override_due_nonfinite_GIoU",
            "window_images": job.dataset == "tinyperson",
            "upload_required": True,
        })
    command = [
        sys.executable, str(repo / "train_hf_detection_matrix.py"),
        "--dataset", job.dataset,
        "--data-root", spec["root"],
        "--model", job.model,
        "--variant", job.variant,
        "--split-seed", str(args.split_seed),
        "--seed", str(job.seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--image-size", str(spec["image_size"]),
        "--batch-size", str(spec["batch_size"]),
        "--workers", str(spec["workers"]),
        "--learning-rate", str(lr),
        "--weight-decay", "0.0005",
        "--output-dir", str(run),
        "--hf-repo-id", REPO_ID,
        "--remote-prefix", f"stability/server{args.server_index}/{job.dataset}/{job.variant}/{job.model}/seed{job.seed}",
        "--upload-required",
    ]
    env = {
        "PYTHONPATH": str(repo),
        "MARIMO_TRAIN_WORKFLOW": "1",
        "MARIMO_HF_REPO_ID": REPO_ID,
        "HF_TOKEN": os.environ["HF_TOKEN"],
    }
    launched = launch_detached(command, cwd=repo, log_path=run / "train.log", pid_path=run / "train.pid", state_path=run / "state.json", artifact_root=run, contract_path=contract, env=env)
    print(json.dumps({"job": asdict(job), "pid": launched.pid, "learning_rate": lr}), flush=True)
    while True:
        observed = status(run, state_file="state.json", emit=False)
        if not observed.get("process_alive"):
            state = json.loads((run / "state.json").read_text(encoding="utf-8"))
            if state.get("returncode") != 0:
                raise RuntimeError(f"Job failed: {job}, state={state}")
            required = (run / "experiment_manifest.json", run / "final_results.json", run / "upload_complete.json")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError(f"Job exited without required artifacts: {missing}")
            return
        time.sleep(30)


def main():
    args = parse_args()
    if args.epochs != 100 or args.patience != 15 or args.split_seed != 42:
        raise ValueError("Stability matrix requires epochs=100, patience=15, split_seed=42")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must come from live Marimo globals")
    repo = Path(__file__).resolve().parent
    jobs = [job for index, job in enumerate(all_jobs()) if index % 3 == args.server_index - 1]
    root = Path(args.work_root)
    state_path = Path(args.state_file)
    statuses = ["pending"] * len(jobs)
    write_state(state_path, args.server_index, jobs, statuses)
    for index, job in enumerate(jobs):
        already_done, evidence = baseline_evidence(args.server_index, job)
        if already_done:
            statuses[index] = "completed_baseline_verified"
            print(json.dumps({"job": asdict(job), "status": statuses[index], "evidence": evidence}), flush=True)
            write_state(state_path, args.server_index, jobs, statuses)
            continue
        statuses[index] = "running"
        write_state(state_path, args.server_index, jobs, statuses, active=f"{job.dataset}/{job.variant}/{job.model}/seed{job.seed}")
        run_job(job, args, repo, root / f"server{args.server_index}")
        statuses[index] = "completed_stability"
        write_state(state_path, args.server_index, jobs, statuses)
    write_state(state_path, args.server_index, jobs, statuses, active=None)
    print(json.dumps({"server": args.server_index, "completed": len(jobs), "total": len(all_jobs())}, indent=2))


if __name__ == "__main__":
    main()
