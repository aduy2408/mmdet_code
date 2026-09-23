#!/usr/bin/env python3
"""Queue the requested STW-matched FCOS-SET protocol for LEVIR-Ship and TinyPerson.

This file only runs through the Marimo training workflow. It deliberately keeps
both variants explicit and does not mutate the canonical FCOS-SET config.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = "/marimo/mmdet-venv/bin/python"
WORK = ROOT / "work_dirs/set_stw_protocol"
HF_REPO = "duyle2408/set-fcos-stw-protocol-runs"


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()

JOBS = (
    {
        "name": "levirship_no_mosaic",
        "dataset": "LEVIR-Ship",
        "launcher": "train_all_levir_baseline.py",
        "data_root": "/marimo/LevirShip/LevirShipData",
        "dataset_out": WORK / "data/levir_ship",
        "work_dir": WORK / "levirship/no_mosaic",
        "variant": "no_mosaic",
        "image_size": 512,
        "remote_prefix": "levirship/no_mosaic/seed42",
    },
    {
        "name": "tinyperson_mosaic",
        "dataset": "TinyPerson",
        "launcher": "train_all_tinyperson_baseline.py",
        "data_root": "/marimo/TinyPerson",
        "dataset_out": WORK / "data/tinyperson",
        "work_dir": WORK / "tinyperson/mosaic",
        "variant": "mosaic",
        "image_size": 640,
        "remote_prefix": "tinyperson/mosaic/seed42",
    },
)


def write_manifest(job: dict[str, object]) -> Path:
    out = Path(job["work_dir"]) / "setup_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": f"set-stw-protocol-{job['name']}-seed42",
        "baseline": {
            "control_config_or_baseline_config": str(
                ROOT / "mmdetection/configs/set/fcos_r50_set.py"
            ),
            "model": "FCOS_set",
        },
        "variant": {
            "variant_config_or_explicit_change": (
                f"normal YOLO-matched pipeline; mosaic={job['variant'] == 'mosaic'}"
            ),
            "protocol": job["variant"],
        },
        "source_commit": source_commit(),
        "runner": str(ROOT / "run_set_stw_protocol_queue.py"),
        "dataset": job["dataset"],
        "dataset_root": job["data_root"],
        "python_executable": PYTHON,
        "split_seed": 42,
        "training_seed": 42,
        "model_backbone_pretrained_source": (
            "FCOS_set / ResNet-50 / torchvision://resnet50"
        ),
        "image_size": [job["image_size"], job["image_size"]],
        "batch_size": 8,
        "epochs": 100,
        "patience": 0,
        "workers": 8,
        "amp": False,
        "optimizer": {
            "type": "MuSGD",
            "lr": 0.01,
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay": 0.0005,
            "muon": 0.2,
            "sgd": 1.0,
        },
        "nms_iou": 0.5,
        "hf_repo": HF_REPO,
        "remote_prefix": job["remote_prefix"],
        "required_artifacts": [
            "patched_config.py",
            "checkpoint",
            "test_results",
            "job_summary.json",
        ],
        "upload_required": True,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def command_for(job: dict[str, object]) -> list[str]:
    command = [
        PYTHON,
        str(ROOT / str(job["launcher"])),
        "--python",
        PYTHON,
        "--data-root",
        str(job["data_root"]),
        "--work-dir",
        str(job["work_dir"]),
        "--models",
        "fcos_set",
        "--variant",
        str(job["variant"]),
        "--epochs",
        "100",
        "--patience",
        "0",
        "--batch-size",
        "8",
        "--num-workers",
        "8",
        "--image-size",
        str(job["image_size"]),
        "--seed",
        "42",
        "--split-seed",
        "42",
        "--hf-repo-id",
        HF_REPO,
        "--remote-prefix",
        str(job["remote_prefix"]),
    ]
    if job["name"] == "levirship_no_mosaic":
        command.extend(["--dataset-out", str(job["dataset_out"])])
    else:
        command.extend(["--prepared-ann-dir", str(job["dataset_out"])])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1" and not (args.list or args.dry_run):
        raise RuntimeError("Launch through python -m utils.marimo_ops launch")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "mmdetection"), env.get("PYTHONPATH", "")]
    )
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    for job in JOBS:
        manifest = write_manifest(job)
        command = command_for(job)
        print(f"MANIFEST {manifest}")
        print("RUN", " ".join(command), flush=True)
        if not args.list and not args.dry_run:
            subprocess.run(command, cwd=str(ROOT), env=env, check=True)
            print(f"COMPLETED {job['name']}", flush=True)


if __name__ == "__main__":
    main()
