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


def write_manifest(name: str, root: str, seed: int, work_dir: Path, variant: str = "base") -> None:
    out = work_dir / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": f"set-fcos-mmdet3-{name}-seed{seed}",
        "control_config_or_baseline_config": str(ROOT / "mmdetection/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py"),
        "variant_config_or_explicit_change": str(ROOT / "mmdetection/configs/set/fcos_r50_set.py"),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "runner": str(ROOT / "run_set_fcos_mmdet3_queue.py"),
        "python_executable": PYTHON,
        "dataset_root": root,
        "split_seed": 42,
        "training_seed": seed,
        "model/backbone/pretrained source": "FCOS_set 3.x port / ResNet-50 / torchvision://resnet50",
        "image_size, batch_size, epochs, patience, AMP": "640x640, 8, 100, 0, false",
        "NMS IoU": 0.5,
        "HF repo and remote prefix": f"{HF_REPO}:set_fcos/{name}/seed{seed}",
        "required artifacts": ["manifest.json", "checkpoint", "test results", "upload"],
        "upload_required": True,
        "variant": variant,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", default="varroa,levir_ship,tinyperson",
        help="Comma-separated dataset stages to run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    selected = {item.strip() for item in args.datasets.split(",")}
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT), str(ROOT / "mmdetection"), env.get("PYTHONPATH", "")
    ])
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    jobs = (
        ("varroa", "/marimo/Varroa", "train_all_mmdet.py"),
        ("levir_ship", "/marimo/LevirShip/LevirShipData", "train_all_levir_baseline.py"),
        ("tinyperson", "/marimo/TinyPerson", "train_all_tinyperson_baseline.py"),
    )
    for seed in args.seeds:
        for name, dataset_root, launcher in jobs:
            if name not in selected:
                continue
            work_dir = WORK / name / f"seed_{seed}"
            command = [
                PYTHON, str(ROOT / launcher),
                "--data-root", dataset_root,
                "--work-dir", str(work_dir),
                "--models", "fcos_set",
                "--epochs", "100",
                "--patience", "0",
                "--seed", str(seed),
                "--split-seed", "42",
                "--batch-size", "8",
                "--num-workers", "8",
                "--hf-repo-id", HF_REPO,
            ]
            if name == "varroa":
                command += [
                    "--dataset-out", str(work_dir / "data" / "varroa_coco"),
                    "--variants", "base",
                ]
            elif name == "tinyperson":
                command += [
                    "--prepared-ann-dir", str(work_dir / "data" / "tinyperson"),
                    "--python", PYTHON,
                ]
            else:
                command += ["--dataset-out", str(work_dir / "data" / "levir_ship")]
            write_manifest(name, dataset_root, seed, work_dir)
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, cwd=str(ROOT), env=env, check=True)
            print("COMPLETED", f"{name}/seed_{seed}", flush=True)


if __name__ == "__main__":
    main()
