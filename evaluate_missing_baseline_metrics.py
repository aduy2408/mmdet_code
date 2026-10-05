#!/usr/bin/env python3
"""Evaluate missing MMDetection baseline metrics without retraining.

This is a setup-only, evaluation-only runner for the gaps recorded in
``yolo_related/docs/reports/mmdetection_baseline_runs_report.md``:

* Varroa: validation AP50, AP75, mAP50-95, and AP50-Small for all 24
  no-mosaic/mosaic x seed42/43 x six-model baseline rows.
* LEVIR-Ship: seed42 no-mosaic RetinaNet, validation and test metrics.
* TinyPerson: seed43 no-mosaic RetinaNet, validation and test metrics.

The runner is deliberately fail-closed. Every job must provide explicit
split-specific configs, checkpoints, and COCO annotation files. It does not
infer a model, dataset, checkpoint, or output location from a filename and it
does not train or upload anything. Launch it through the Marimo workflow only
when the user supplies a server and requests execution.

A job JSON file has this shape::

    {
      "jobs": [
        {
          "id": "varroa_no_mosaic_seed42_fcos",
          "dataset": "varroa",
          "model": "fcos",
          "protocol": "no_mosaic",
          "seed": 42,
          "config_val": "/marimo/.../patched_config.py",
          "config_test": "/marimo/.../patched_config.py",
          "checkpoint": "/marimo/.../best_coco_bbox_mAP_epoch_XX.pth",
          "annotation_val": "/marimo/.../val.json",
          "annotation_test": "/marimo/.../test.json"
        }
      ]
    }

Each split config must already point its dataloader/evaluator at the intended
split. This avoids silently evaluating the wrong split.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


MODELS = ("fcos", "faster_rcnn", "cascade_rcnn", "rtmdet", "atss", "retinanet")
PROTOCOLS = ("no_mosaic", "mosaic")
DATASETS = ("varroa", "levirship", "tinyperson")
REQUIRED_FIELDS = (
    "val/AP50",
    "val/AP75",
    "val/mAP50-95",
    "val/AP50-Small",
    "test/AP50",
    "test/AP75",
    "test/mAP50-95",
    "test/AP50-Small",
)


@dataclass(frozen=True)
class Job:
    job_id: str
    dataset: str
    model: str
    protocol: str
    seed: int
    config_val: Path
    config_test: Path
    checkpoint: Path
    annotation_val: Path
    annotation_test: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--mmdet-root", type=Path, required=True)
    parser.add_argument("--python", default="/marimo/mmdet-venv/bin/python")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-dets", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=0, help="Contract metadata; evaluation does not train")
    parser.add_argument("--patience", type=int, default=0, help="Contract metadata; evaluation does not train")
    parser.add_argument("--workers", type=int, default=8, help="Contract metadata and MMDetection worker count")
    parser.add_argument("--seed", type=int, default=42, help="Contract metadata")
    parser.add_argument("--split-seed", type=int, default=42, help="Contract metadata")
    parser.add_argument("--data-root", default="/marimo/Varroa;/marimo/TinyPerson;/marimo/LevirShip", help="Contract metadata")
    parser.add_argument("--model-yaml", default="remote HF patched configs per job", help="Contract metadata")
    parser.add_argument("--hf-repo-id", default="duyle2408/mmdet-missing-baseline-eval-20261005", help="Contract metadata")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_jobs(path: Path) -> list[Job]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("jobs file must contain a non-empty 'jobs' list")
    jobs: list[Job] = []
    seen: set[str] = set()
    for raw in raw_jobs:
        required = (
            "id", "dataset", "model", "protocol", "seed", "config_val",
            "config_test", "checkpoint", "annotation_val", "annotation_test",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Job is missing fields {missing}: {raw!r}")
        job_id = str(raw["id"])
        if job_id in seen:
            raise ValueError(f"Duplicate job id: {job_id}")
        seen.add(job_id)
        dataset = str(raw["dataset"])
        model = str(raw["model"])
        protocol = str(raw["protocol"])
        if dataset not in DATASETS:
            raise ValueError(f"Unsupported dataset {dataset!r} for {job_id}")
        if model not in MODELS:
            raise ValueError(f"Unsupported model {model!r} for {job_id}")
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unsupported protocol {protocol!r} for {job_id}")
        seed = int(raw["seed"])
        if seed not in (42, 43):
            raise ValueError(f"Expected training seed 42 or 43 for {job_id}, got {seed}")
        jobs.append(
            Job(
                job_id=job_id,
                dataset=dataset,
                model=model,
                protocol=protocol,
                seed=seed,
                config_val=Path(raw["config_val"]).expanduser().resolve(),
                config_test=Path(raw["config_test"]).expanduser().resolve(),
                checkpoint=Path(raw["checkpoint"]).expanduser().resolve(),
                annotation_val=Path(raw["annotation_val"]).expanduser().resolve(),
                annotation_test=Path(raw["annotation_test"]).expanduser().resolve(),
            )
        )
    return jobs


def validate_job(job: Job) -> None:
    for path in (
        job.config_val,
        job.config_test,
        job.checkpoint,
        job.annotation_val,
        job.annotation_test,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{job.job_id}: required file does not exist: {path}")


def run_mmdet_test(
    job: Job,
    split: str,
    config: Path,
    mmdet_root: Path,
    python: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = output_dir / f"{split}_predictions.pkl"
    command = [
        python,
        str(mmdet_root / "tools" / "test.py"),
        str(config),
        str(job.checkpoint),
        "--work-dir",
        str(output_dir),
        "--out",
        str(prediction_file),
    ]
    env = os.environ.copy()
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(mmdet_root), env.get("PYTHONPATH", "")) if item
    )
    subprocess.run(command, cwd=str(mmdet_root), env=env, check=True)
    if not prediction_file.is_file():
        raise FileNotFoundError(f"MMDetection test produced no predictions: {prediction_file}")
    return prediction_file


def precision(evaluator: COCOeval, iou: float | None, area: str) -> float | None:
    values = evaluator.eval["precision"]
    if iou is not None:
        indices = np.where(np.isclose(evaluator.params.iouThrs, iou))[0]
        if not len(indices):
            raise ValueError(f"IoU threshold {iou} missing from evaluator")
        values = values[indices]
    area_index = evaluator.params.areaRngLbl.index(area)
    values = values[:, :, :, area_index, -1]
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else None


def predictions_to_coco(predictions: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in predictions:
        instances = item["pred_instances"]
        boxes = instances["bboxes"]
        scores = instances["scores"]
        labels = instances["labels"]
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().tolist()
            scores = scores.detach().cpu().tolist()
            labels = labels.detach().cpu().tolist()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            results.append({
                "image_id": int(item["img_id"]),
                "category_id": int(label) + 1,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score),
            })
    return results


def evaluate_predictions(annotation: Path, prediction_file: Path, max_dets: int) -> dict[str, float | None]:
    import pickle

    ground_truth = COCO(str(annotation))
    with prediction_file.open("rb") as handle:
        predictions = pickle.load(handle)
    results = predictions_to_coco(predictions)
    detections = ground_truth.loadRes(results) if results else ground_truth.loadRes([])
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.maxDets = [1, 10, max_dets]
    evaluator.evaluate()
    evaluator.accumulate()
    return {
        "AP50": precision(evaluator, 0.50, "all"),
        "AP75": precision(evaluator, 0.75, "all"),
        "mAP50-95": precision(evaluator, None, "all"),
        "AP50-Small": precision(evaluator, 0.50, "small"),
        "AP-Small": precision(evaluator, None, "small"),
        "AP-Medium": precision(evaluator, None, "medium"),
        "prediction_count": len(results),
    }


def evaluate_job(job: Job, args: argparse.Namespace) -> dict[str, Any]:
    validate_job(job)
    job_dir = args.output_root / job.job_id
    result: dict[str, Any] = {
        "job_id": job.job_id,
        "dataset": job.dataset,
        "model": job.model,
        "protocol": job.protocol,
        "seed": job.seed,
        "checkpoint": str(job.checkpoint),
        "split_seed": 42,
        "nms_iou": 0.5,
        "source": "MMDetection tools/test.py + pycocotools",
    }
    for split, config, annotation in (
        ("val", job.config_val, job.annotation_val),
        ("test", job.config_test, job.annotation_test),
    ):
        prediction_file = run_mmdet_test(job, split, config, args.mmdet_root, args.python, job_dir)
        metrics = evaluate_predictions(annotation, prediction_file, args.max_dets)
        for key, value in metrics.items():
            result[f"{split}/{key}"] = value
    result["test_protocol"] = (
        "TinyPerson native tile/test COCO protocol; merged original-image metrics require a separate evaluator"
        if job.dataset == "tinyperson"
        else "Native held-out COCO test split"
    )
    return result


def main() -> None:
    args = parse_args()
    args.mmdet_root = args.mmdet_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not (args.mmdet_root / "tools" / "test.py").is_file():
        raise FileNotFoundError(f"MMDetection test entrypoint not found: {args.mmdet_root / 'tools' / 'test.py'}")
    jobs = load_jobs(args.jobs_file.expanduser().resolve())
    print(json.dumps({"jobs": len(jobs), "required_fields": REQUIRED_FIELDS}, sort_keys=True))
    for job in jobs:
        validate_job(job)
        print(json.dumps({"job": job.job_id, "dataset": job.dataset, "model": job.model, "protocol": job.protocol, "seed": job.seed}, sort_keys=True))
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        output = args.output_root / job.job_id / "evaluation_metrics.json"
        if output.is_file() and not args.force:
            print(f"SKIP_EXISTING {job.job_id}")
            continue
        result = evaluate_job(job, args)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"EVALUATION_COMPLETE {job.job_id} -> {output}")


if __name__ == "__main__":
    main()
