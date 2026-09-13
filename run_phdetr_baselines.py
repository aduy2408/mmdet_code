#!/usr/bin/env python3
"""Run the standalone PH-DETR implementation with the project baseline protocol.

PH-DETR is not an MMDetection model.  This launcher prepares the same COCO
splits used by ``train_all_*_baseline.py``, generates a native PH-DETR YAML
configuration, runs PH-DETR's own trainer/evaluator, and uploads the completed
run directory when ``HF_TOKEN`` is available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

import train_all_levir_baseline as levir
import train_all_tinyperson_baseline as tinyperson


PHDETR_COMMIT = "6817e8d1f0fe7d2b59c373ac98bbd9683bdfe053"
DATASET_SETTINGS = {
    "levirship": {
        "label": "LEVIR-Ship",
        "data_root": "LevirShipData",
        "image_size": 512,
        "batch_size": 4,
        "hf_repo": "duyle2408/levir_ship_mmdet_runs",
        "prepared_dir": "mmdetection/data/phdetr_levir_ship",
        "work_root": "mmdetection/work_dirs/phdetr_levir_ship",
    },
    "tinyperson": {
        "label": "TinyPerson",
        "data_root": "../TinyPerson/tiny_set",
        "image_size": 640,
        "batch_size": 2,
        "hf_repo": "duyle2408/tinyperson_mmdet_runs",
        "prepared_dir": "mmdetection/data/phdetr_tinyperson_seed42",
        "work_root": "mmdetection/work_dirs/phdetr_tinyperson",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root() / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SETTINGS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--python", default="/marimo/mmdet-venv/bin/python")
    parser.add_argument("--phdetr-root", default="/marimo/PH-DETR")
    parser.add_argument("--work-root")
    parser.add_argument("--data-root")
    parser.add_argument("--prepared-dir")
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-prefix", default="phdetr")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def ensure_phdetr_root(path: Path) -> None:
    if not (path / "train.py").is_file():
        raise FileNotFoundError(
            f"PH-DETR checkout missing train.py: {path}; expected commit {PHDETR_COMMIT}"
        )
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True, capture_output=True, check=True
    ).stdout.strip()
    if actual != PHDETR_COMMIT:
        raise RuntimeError(f"PH-DETR commit mismatch: expected {PHDETR_COMMIT}, got {actual}")


def prepare_dataset(args: argparse.Namespace, settings: dict[str, Any]) -> dict[str, Path]:
    if args.dataset == "levirship":
        class Namespace:
            pass

        ns = Namespace()
        ns.data_root = args.data_root
        ns.dataset_out = args.prepared_dir
        ns.split_seed = args.split_seed
        ns.limit = args.limit
        dataset_out, image_dir = levir.prepare_coco_dataset(ns)
        return {
            "train_ann": dataset_out / "annotations/train.json",
            "val_ann": dataset_out / "annotations/val.json",
            "test_ann": dataset_out / "annotations/test.json",
            "train_images": image_dir,
            "val_images": image_dir,
            "test_images": image_dir,
        }

    dataset_root = resolve(args.data_root)
    train_images = tinyperson.ensure_erased_train_images(dataset_root, args.dry_run)
    prepared = tinyperson.prepare_validation_split(
        dataset_root,
        resolve(args.prepared_dir),
        args.split_seed,
        0.15,
    )
    return {
        "train_ann": prepared["train"],
        "val_ann": prepared["val"],
        "test_ann": dataset_root / tinyperson.TEST_ANN,
        "train_images": train_images,
        "val_images": train_images,
        "test_images": dataset_root / "test",
    }


def make_config(
    args: argparse.Namespace,
    settings: dict[str, Any],
    dataset: dict[str, Path],
    work_dir: Path,
    phdetr_root: Path,
    ann_file: Path | None = None,
    image_dir: Path | None = None,
    name: str = "phdetr_train.yml",
) -> Path:
    image_size = args.image_size or settings["image_size"]
    batch_size = args.batch_size or settings["batch_size"]
    active_ann = ann_file or dataset["train_ann"]
    active_images = image_dir or dataset["train_images"]
    validation_ann = ann_file or dataset["val_ann"]
    validation_images = image_dir or dataset["val_images"]
    config = {
        "__include__": [
            str(phdetr_root / "configs/dataset/custom_detection.yml"),
            str(phdetr_root / "configs/runtime.yml"),
            str(phdetr_root / "configs/include/dataloader.yml"),
            str(phdetr_root / "configs/include/optimizer.yml"),
            str(phdetr_root / "configs/include/dfine_hgnetv2.yml"),
        ],
        "output_dir": str(work_dir),
        "num_classes": 1,
        "eval_spatial_size": [image_size, image_size],
        "epoches": args.epochs,
        "checkpoint_freq": 1,
        "use_amp": True,
        "use_ema": True,
        "train_dataloader": {
            "dataset": {
                "img_folder": str(active_images),
                "ann_file": str(active_ann),
                "return_masks": False,
                "transforms": {
                    "type": "Compose",
                    "ops": [
                        {"type": "Resize", "size": [image_size, image_size]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
                    ],
                },
            },
            "total_batch_size": batch_size,
            "num_workers": args.num_workers,
            "collate_fn": {
                "base_size": image_size,
                "stop_epoch": args.epochs,
                "base_size_repeat": 3,
                "ema_restart_decay": 0.9999,
            },
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": str(validation_images),
                "ann_file": str(validation_ann),
                "return_masks": False,
                "transforms": {
                    "type": "Compose",
                    "ops": [
                        {"type": "Resize", "size": [image_size, image_size]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                    ],
                },
            },
            "total_batch_size": 1,
            "num_workers": args.num_workers,
        },
        "HGNetv2": {
            "name": "B2",
            "return_idx": [0, 1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
        },
        "HybridEncoder": {
            "in_channels": [96, 384, 768, 1536],
            "feat_strides": [4, 8, 16, 32],
            "hidden_dim": 128,
            "depth_mult": 0.67,
            "use_encoder_idx": [3],
        },
        "DFINETransformer": {
            "num_layers": 4,
            "eval_idx": -1,
            "feat_channels": [128, 128, 128],
            "num_levels": 3,
        },
        "DFINEPostProcessor": {"num_top_queries": 300},
        "randomness": {"seed": args.seed},
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / name
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def run_command(command: list[str], cwd: Path, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        completed = subprocess.run(
            command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, text=True
        )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return log_path.read_text(encoding="utf-8", errors="replace")


def native_train_command(
    python: str,
    entrypoint: Path,
    config: Path,
    *extra: str,
) -> list[str]:
    """Use one-process torchrun because PH-DETR calls distributed APIs at init."""
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port=29511",
        str(entrypoint),
        "-c",
        str(config),
        *extra,
    ]


def metric_rows(text: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for match in re.finditer(r"coco_eval_bbox['\"]?\s*:\s*(\[[^\]]+\])", text):
        try:
            row = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(row, list) and row and all(isinstance(v, (float, int)) for v in row):
            rows.append([float(v) for v in row])
    return rows


def metric_dict(rows: list[list[float]]) -> dict[str, float | None]:
    if not rows:
        raise RuntimeError("PH-DETR evaluator produced no coco_eval_bbox metrics")
    best = max(rows, key=lambda row: row[0])
    names = [
        "map_50_95",
        "ap50",
        "ap75",
        "ap_small",
        "ap_medium",
        "ap_large",
    ]
    return {name: (best[index] if index < len(best) else None) for index, name in enumerate(names)}


def upload(work_dir: Path, repo_id: str, prefix: str) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required before PH-DETR training/upload")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    remote_path = f"{prefix}/{work_dir.name}"
    api.upload_folder(
        folder_path=str(work_dir),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="dataset",
    )
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision="main")
    if not any(path.startswith(remote_path + "/") for path in files):
        raise RuntimeError(f"HF upload verifier did not find remote prefix {remote_path}")
    return remote_path


def run_one(args: argparse.Namespace) -> Path:
    settings = DATASET_SETTINGS[args.dataset]
    args.data_root = args.data_root or settings["data_root"]
    args.prepared_dir = args.prepared_dir or settings["prepared_dir"]
    args.work_root = args.work_root or settings["work_root"]
    args.hf_repo_id = args.hf_repo_id or settings["hf_repo"]
    phdetr_root = resolve(args.phdetr_root)
    entrypoint = repo_root() / "phdetr_compat_entry.py"
    ensure_phdetr_root(phdetr_root)
    dataset = prepare_dataset(args, settings)
    work_dir = resolve(args.work_root) / f"seed{args.seed}"
    train_config = make_config(args, settings, dataset, work_dir, phdetr_root)
    test_config = make_config(
        args,
        settings,
        dataset,
        work_dir,
        phdetr_root,
        ann_file=dataset["test_ann"],
        image_dir=dataset["test_images"],
        name="phdetr_test.yml",
    )
    manifest = {
        "experiment_id": f"phdetr_{args.dataset}_seed{args.seed}",
        "baseline": "train_all baseline split and resize protocol",
        "variant": "standalone PH-DETR native runner",
        "source_commit": PHDETR_COMMIT,
        "runner": "phdetr_compat_entry.py -> PH-DETR/train.py",
        "python_executable": args.python,
        "dataset_root": str(resolve(args.data_root)),
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "model_backbone_pretrained": "PH-DETR DFINE with HGNetv2-B2 pretrained=True",
        "image_size": args.image_size or settings["image_size"],
        "batch_size": args.batch_size or settings["batch_size"],
        "epochs": args.epochs,
        "patience": "none",
        "amp": True,
        "nms_iou": "not_applicable_native_phdetr",
        "hf_repo": args.hf_repo_id,
        "hf_remote_prefix": f"{args.hf_prefix}/{work_dir.name}",
        "required_artifacts": ["phdetr_train.yml", "phdetr_test.yml", "train.log", "test.log", "final_results.json", "best_stg1.pth"],
        "upload_required": True,
        "train_config": str(train_config),
        "test_config": str(test_config),
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return work_dir
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required before training")

    train_log = run_command(
        native_train_command(
            args.python,
            entrypoint,
            train_config,
            "--use-amp",
            "--seed",
            str(args.seed),
        ),
        phdetr_root,
        work_dir / "train.log",
    )
    checkpoint = work_dir / "best_stg1.pth"
    if not checkpoint.is_file():
        checkpoint = work_dir / "best_stg2.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PH-DETR did not produce a best checkpoint in {work_dir}")
    test_log = run_command(
        native_train_command(
            args.python,
            entrypoint,
            test_config,
            "-r",
            str(checkpoint),
            "--test-only",
            "--use-amp",
            "--seed",
            str(args.seed),
        ),
        phdetr_root,
        work_dir / "test.log",
    )
    results = {
        "dataset": settings["label"],
        "model": "phdetr",
        "seed": args.seed,
        "source_commit": PHDETR_COMMIT,
        "validation": metric_dict(metric_rows(train_log)),
        "test": metric_dict(metric_rows(test_log)),
    }
    (work_dir / "final_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    remote_path = upload(work_dir, args.hf_repo_id, args.hf_prefix)
    results["hf_remote_path"] = remote_path
    (work_dir / "final_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return work_dir


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise SystemExit("--dataset is required; run one dataset/seed per Marimo queue item")
    if args.smoke_test:
        args.epochs = 1
        if args.limit == 0 and args.dataset == "levirship":
            args.limit = 64
    run_one(args)


if __name__ == "__main__":
    main()
