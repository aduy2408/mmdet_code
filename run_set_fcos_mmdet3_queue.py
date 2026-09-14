#!/usr/bin/env python3
"""Sequential MMDetection 3.x FCOS_set queue for the three datasets."""
from __future__ import annotations

import json
import os
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = "/marimo/mmdet-venv/bin/python"
WORK = ROOT / "work_dirs/set_fcos3"
HF_REPO = "duyle2408/set_fcos_runs"


def write_manifest(name: str, root: str, variant: str = "base") -> None:
    out = WORK / name / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": f"set-fcos-mmdet3-{name}-seed42",
        "control_config_or_baseline_config": str(ROOT / "mmdetection/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py"),
        "variant_config_or_explicit_change": str(ROOT / "mmdetection/configs/set/fcos_r50_set.py"),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "runner": str(ROOT / "run_set_fcos_mmdet3_queue.py"),
        "python_executable": PYTHON,
        "dataset_root": root,
        "split_seed": 42,
        "training_seed": 42,
        "model/backbone/pretrained source": "FCOS_set 3.x port / ResNet-50 / torchvision://resnet50",
        "image_size, batch_size, epochs, patience, AMP": "1333x800 keep_ratio, 2, 12, none, false",
        "NMS IoU": 0.5,
        "HF repo and remote prefix": f"{HF_REPO}:set_fcos/{name}/seed42",
        "required artifacts": ["manifest.json", "checkpoint", "test results", "upload"],
        "upload_required": True,
        "variant": variant,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", default="varroa,levir_ship,tinyperson",
        help="Comma-separated dataset stages to run.")
    selected = {item.strip() for item in parser.parse_args().datasets.split(',')}
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT), str(ROOT / "mmdetection"), env.get("PYTHONPATH", "")
    ])
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    commands = [
        ("varroa", "/marimo/Varroa", [
            PYTHON, str(ROOT / "train_all_mmdet.py"),
            "--data-root", "/marimo/Varroa",
            "--dataset-out", str(WORK / "data/varroa_coco"),
            "--work-dir", str(WORK / "varroa"),
            "--models", "fcos_set", "--variants", "base",
            "--epochs", "12", "--batch-size", "2", "--num-workers", "4",
            "--hf-repo-id", HF_REPO,
        ]),
        ("levir_ship", "/marimo/LevirShip/LevirShipData", [
            PYTHON, str(ROOT / "train_all_levir_baseline.py"),
            "--data-root", "/marimo/LevirShip/LevirShipData",
            "--dataset-out", str(WORK / "data/levir_ship"),
            "--work-dir", str(WORK / "levir_ship"),
            "--models", "fcos_set",
            "--epochs", "12", "--batch-size", "2", "--num-workers", "4",
            "--hf-repo-id", HF_REPO,
        ]),
        ("tinyperson", "/marimo/TinyPerson", [
            PYTHON, str(ROOT / "train_all_tinyperson_baseline.py"),
            "--data-root", "/marimo/TinyPerson",
            "--prepared-ann-dir", str(WORK / "data/tinyperson_seed42"),
            "--work-dir", str(WORK / "tinyperson"),
            "--models", "fcos_set",
            "--epochs", "12", "--batch-size", "2", "--num-workers", "4",
            "--python", PYTHON, "--hf-repo-id", HF_REPO,
        ]),
    ]
    for name, dataset_root, command in commands:
        if name not in selected:
            continue
        write_manifest(name, dataset_root)
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)
        print("COMPLETED", name, flush=True)


if __name__ == "__main__":
    main()
