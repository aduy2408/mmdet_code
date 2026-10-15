#!/usr/bin/env python3
"""Re-evaluate uploaded TinyPerson MMDetection predictions with merged metrics.

The source HF repositories contain the original checkpoints, configs, tile-level
predictions, and prior metrics. This runner pulls those artifacts, recomputes
validation and test metrics with the repository evaluator, combines them with
split-qualified native fields, and uploads a task-specific verified result repo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

MODELS = {"fcos", "faster_rcnn", "cascade_rcnn", "rtmdet", "atss", "retinanet"}
PROTOCOLS = {"mosaic", "no_mosaic"}
SOURCE_REPOS = {
    42: "duyle2408/tinyperson_mmdet_yolo_protocol_seed42_runs",
    43: "duyle2408/tinyperson_mmdet_yolo_protocol_seed43_runs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/marimo/tinyperson_mmdet_merged_eval_20261015"))
    parser.add_argument("--python", default="/marimo/mmdet-venv/bin/python")
    parser.add_argument("--code-root", type=Path, default=Path("/marimo/mmdet_code"))
    parser.add_argument("--data-root", type=Path, default=Path("/marimo/TinyPerson"))
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--models", default=",".join(sorted(MODELS)))
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--protocols", default="mosaic,no_mosaic")
    parser.add_argument("--pull-checkpoints", action="store_true")
    return parser.parse_args()


def source_patterns(pull_checkpoints: bool) -> list[str]:
    patterns = [
        "tinyperson_mmdet_yolo_protocol/**/final_results.json",
        "tinyperson_mmdet_yolo_protocol/**/patched_config*.py",
        "tinyperson_mmdet_yolo_protocol/**/experiment_manifest.json",
        "tinyperson_mmdet_yolo_protocol/**/test_results/**/*.json",
    ]
    if pull_checkpoints:
        patterns.append("tinyperson_mmdet_yolo_protocol/**/*.pth")
    return patterns


def discover_runs(snapshot: Path) -> dict[tuple[str, int, str], Path]:
    runs: dict[tuple[str, int, str], Path] = {}
    for final in snapshot.glob("tinyperson_mmdet_yolo_protocol/**/final_results.json"):
        run_dir = final.parent
        test_pred = run_dir / "test_results" / "test" / "tinyperson.bbox.json"
        val_pred = run_dir / "test_results" / "validation" / "tinyperson.bbox.json"
        if not test_pred.is_file() or not val_pred.is_file():
            continue
        parts = run_dir.relative_to(snapshot / "tinyperson_mmdet_yolo_protocol").parts
        if len(parts) < 3 or parts[0] not in PROTOCOLS:
            continue
        protocol = parts[0]
        match = re.fullmatch(r"seed(\d+)", parts[1])
        if not match:
            continue
        seed = int(match.group(1))
        model = next((part for part in parts[2:] if part in MODELS), None)
        if model is None:
            continue
        key = (protocol, seed, model)
        old = runs.get(key)
        if old is None or len(run_dir.parts) < len(old.parts):
            runs[key] = run_dir
    return runs


def score_premerged(result: Path, merged_gt: Path, output: Path) -> dict[str, Any]:
    import numpy as np
    from evaluate_tinyperson_metrics import evaluate, mean_precision

    prepared = evaluate(
        merged_gt,
        result,
        np.array([0.25, 0.50, 0.75]),
        [[1**2, 1e5**2], [1**2, 20**2], [1**2, 8**2], [8**2, 12**2], [12**2, 20**2], [20**2, 32**2], [32**2, 1e5**2]],
        ["all", "tiny", "tiny1", "tiny2", "tiny3", "small", "reasonable"],
    )
    coco = evaluate(
        merged_gt,
        result,
        np.linspace(0.50, 0.95, 10),
        [[0, 1e5**2], [0, 32**2], [32**2, 96**2], [96**2, 1e5**2]],
        ["all", "small", "medium", "large"],
    )
    metrics = {
        "map_50_95": mean_precision(coco, None, "all"),
        "ap50": mean_precision(prepared, 0.50, "all"),
        "ap75": mean_precision(prepared, 0.75, "all"),
        "ap50_tiny1": mean_precision(prepared, 0.50, "tiny1"),
        "ap50_tiny2": mean_precision(prepared, 0.50, "tiny2"),
        "ap50_tiny3": mean_precision(prepared, 0.50, "tiny3"),
        "ap50_small": mean_precision(prepared, 0.50, "small"),
        "ap_small": mean_precision(coco, None, "small"),
        "ap_medium": mean_precision(coco, None, "medium"),
    }
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def find_native_metrics(split_dir: Path) -> dict[str, float]:
    for candidate in split_dir.glob("**/*.json"):
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict) or "coco/bbox_mAP_50" not in raw:
            continue
        mapping = {
            "coco/bbox_mAP_50": "AP50",
            "coco/bbox_mAP_75": "AP75",
            "coco/bbox_mAP": "mAP50-95",
            "coco/bbox_mAP_s": "AP-Small",
            "coco/bbox_mAP_m": "AP-Medium",
            "coco/bbox_mAP_l": "AP-Large",
        }
        return {label: float(raw[key]) for key, label in mapping.items() if key in raw}
    raise FileNotFoundError(f"Native MMDetection metrics JSON not found under {split_dir}")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required in the detached Marimo child environment")
    args.project.mkdir(parents=True, exist_ok=True)
    selected_models = {x for x in args.models.split(",") if x}
    selected_seeds = {int(x) for x in args.seeds.split(",") if x}
    selected_protocols = {x for x in args.protocols.split(",") if x}
    if not selected_models <= MODELS or not selected_protocols <= PROTOCOLS:
        raise ValueError("Unknown model or protocol")
    if not selected_seeds <= SOURCE_REPOS:
        raise ValueError(f"Only source seeds are available: {sorted(SOURCE_REPOS)}")

    snapshots: dict[int, Path] = {}
    for seed in sorted(selected_seeds):
        snapshots[seed] = Path(snapshot_download(
            repo_id=SOURCE_REPOS[seed],
            repo_type="dataset",
            allow_patterns=source_patterns(args.pull_checkpoints),
            local_dir=str(args.project / f"source_seed{seed}"),
            token=token,
        ))

    api = HfApi(token=token)
    api.create_repo(repo_id=args.output_repo_id, repo_type="dataset", exist_ok=True, private=False)
    all_results: dict[str, Any] = {}
    for seed, snapshot in snapshots.items():
        runs = discover_runs(snapshot)
        for protocol in sorted(selected_protocols):
            for model in sorted(selected_models):
                key = (protocol, seed, model)
                if key not in runs:
                    raise FileNotFoundError(f"Missing source run: {key}")
                source = runs[key]
                out_dir = args.project / "results" / protocol / f"seed{seed}" / model
                out_dir.mkdir(parents=True, exist_ok=True)
                split_metrics: dict[str, Any] = {}
                for split, prefix in (("validation", "val"), ("test", "test")):
                    split_dir = source / "test_results" / split
                    result = split_dir / "metrics_merged_detections.json"
                    merged_gt = split_dir / "metrics_prepared_ground_truth.json"
                    if not result.is_file() or not merged_gt.is_file():
                        raise FileNotFoundError(f"Missing merged artifacts under {split_dir}")
                    merged_output = out_dir / f"{split}_merged_metrics.json"
                    merged = score_premerged(result, merged_gt, merged_output)
                    native = find_native_metrics(split_dir)
                    split_metrics.update({f"{prefix}/{k}": v for k, v in native.items()})
                    merged_prefix = "val_merged" if split == "validation" else "test_merged"
                    split_metrics.update({f"{merged_prefix}/{k}": float(v) for k, v in merged.items()})
                manifest = {
                    "dataset": "TinyPerson",
                    "model": model,
                    "protocol": protocol,
                    "split_seed": args.split_seed,
                    "training_seed": seed,
                    "variant": "post_hoc_merged_re_evaluation",
                    "source_repo": SOURCE_REPOS[seed],
                    "source_prefix": str(source.relative_to(snapshot)),
                    "source_commit": "17449f3b0b471510071ff683f2f59868cb25e432",
                    "runner": str(args.code_root / "evaluate_tinyperson_metrics.py"),
                    "python_executable": args.python,
                    "dataset_root": str(args.data_root),
                    "nms_iou": args.nms_iou,
                    "corner_ground_truth": "source metrics_prepared_ground_truth.json",
                    "merged_ground_truth": "source metrics_prepared_ground_truth.json",
                    "required_artifacts": ["validation_metrics", "test_metrics", "val_merged_metrics", "test_merged_metrics"],
                    "upload_required": True,
                }
                (out_dir / "evaluation_metrics.json").write_text(json.dumps(split_metrics, indent=2, sort_keys=True) + "\n")
                (out_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                (out_dir / "source_final_results.json").write_bytes((source / "final_results.json").read_bytes())
                remote = f"tinyperson_mmdet_merged_eval/{protocol}/seed{seed}/{model}"
                api.upload_folder(folder_path=str(out_dir), path_in_repo=remote, repo_id=args.output_repo_id, repo_type="dataset")
                files = api.list_repo_files(repo_id=args.output_repo_id, repo_type="dataset")
                required = {f"{remote}/{name}" for name in ("evaluation_metrics.json", "evaluation_manifest.json", "source_final_results.json")}
                missing = sorted(required - set(files))
                if missing:
                    raise RuntimeError(f"Remote verification failed for {remote}: {missing}")
                all_results[f"{protocol}/seed{seed}/{model}"] = split_metrics
                print(json.dumps({"run": f"{protocol}/seed{seed}/{model}", "metrics": split_metrics}, sort_keys=True), flush=True)
    summary = args.project / "results" / "summary.json"
    summary.write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n")
    api.upload_file(path_or_fileobj=str(summary), path_in_repo="tinyperson_mmdet_merged_eval/summary.json", repo_id=args.output_repo_id, repo_type="dataset")
    print(json.dumps({"complete": len(all_results), "output_repo": args.output_repo_id}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
