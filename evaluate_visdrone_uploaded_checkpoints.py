#!/usr/bin/env python3
"""Re-evaluate uploaded VisDrone MMDetection checkpoints with full COCO area metrics.

This downloads the exact uploaded checkpoints and patched configs, reruns the
MMDetection test loop, then computes AP/AP50/AP75 for all, small, medium, and
large objects with pycocotools. It is intended for the Marimo MMDetection
runtime and must be launched through utils.marimo_ops.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download
from mmengine.fileio import load

REPO = "duyle2408/visdrone2019-mmdet-baselines-1536-seeds"
RUNS = {
    "faster_rcnn": {
        "42": "visdrone2019_mmdet_1536/faster_rcnn/seed42",
        "43": "visdrone2019_mmdet_1536/faster_rcnn/seed43",
    },
    "cascade_rcnn": {
        "42": "visdrone2019_mmdet_1536_fp32_resume/cascade_rcnn/seed42",
        "43": "visdrone2019_mmdet_1536_fp32_resume2/cascade_rcnn/seed43",
    },
    "retinanet": {
        "42": "visdrone2019_mmdet_final/retinanet/seed42",
        "43": "visdrone2019_mmdet_final/retinanet/seed43",
    },
    "rtmdet": {
        "42": "visdrone2019_mmdet_final/rtmdet/seed42",
        "43": "visdrone2019_mmdet_final/rtmdet/seed43",
    },
    "fcos": {
        "42": "visdrone2019_mmdet_1536/fcos/seed42",
        "43": "visdrone2019_mmdet_1536_fp32_resume2/fcos/seed43",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/marimo/visdrone_full_eval_1536"))
    parser.add_argument("--python", default="/marimo/mmdet-venv/bin/python")
    parser.add_argument("--repo-id", default=REPO)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=Path("/marimo/mmdet_code/mmdetection/data/visdrone2019_coco"))
    parser.add_argument("--model-yaml", default="matrix")
    parser.add_argument("--hf-repo-id", default=REPO)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--models", default=",".join(RUNS))
    return parser.parse_args()


def coco_metrics(annotation: Path, predictions: list[dict[str, Any]]) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO(str(annotation))
    result = coco.loadRes(predictions)
    evaluator = COCOeval(coco, result, "bbox")
    evaluator.params.iouThrs = np.arange(0.50, 0.96, 0.05)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    area_names = {"all": "all", "small": "small", "medium": "medium", "large": "large"}
    area_index = {name: evaluator.params.areaRngLbl.index(label) for name, label in area_names.items()}
    output: dict[str, float] = {}
    for name, index in area_index.items():
        for iou_name, iou_index in (("AP50", 0), ("AP75", 5), ("AP", None)):
            values = precision[:, :, :, index, 2] if iou_index is None else precision[iou_index, :, :, index, 2]
            values = values[values > -1]
            output[f"test/{iou_name}_{name}"] = float(values.mean()) if values.size else -1.0
    output["test/mAP50-95"] = output["test/AP_all"]
    output["test/AP50"] = output["test/AP50_all"]
    output["test/AP75"] = output["test/AP75_all"]
    return output


def normalize_predictions(raw: Any, annotation: Path) -> list[dict[str, Any]]:
    from pycocotools.coco import COCO

    image_ids = COCO(str(annotation)).getImgIds()
    output: list[dict[str, Any]] = []
    if isinstance(raw, dict) and "predictions" in raw:
        raw = raw["predictions"]
    for image_index, item in enumerate(raw):
        image_id = image_ids[image_index]
        if isinstance(item, dict) and "pred_instances" in item:
            instances = item["pred_instances"]
            boxes = instances["bboxes"].detach().cpu().numpy()
            scores = instances["scores"].detach().cpu().numpy()
            labels = instances["labels"].detach().cpu().numpy()
            image_id = int(item.get("img_id", image_id))
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(float, box)
                output.append({"image_id": image_id, "category_id": int(label) + 1, "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": float(score)})
            continue
        if hasattr(item, "pred_instances"):
            instances = item.pred_instances
            boxes = instances.bboxes.detach().cpu().numpy()
            scores = instances.scores.detach().cpu().numpy()
            labels = instances.labels.detach().cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(float, box)
                output.append({"image_id": image_id, "category_id": int(label) + 1, "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": float(score)})
            continue
        if isinstance(item, tuple):
            item = item[0]
        if isinstance(item, list):
            for label, detections in enumerate(item):
                array = np.asarray(detections)
                if array.ndim != 2 or array.shape[1] < 5:
                    continue
                for x1, y1, x2, y2, score in array[:, :5]:
                    output.append({"image_id": image_id, "category_id": label + 1, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(score)})
    return output


def prepare_eval_config(model: str, source: Path, run_dir: Path) -> Path:
    config = source / "patched_config.py"
    if model != "rtmdet":
        return config
    from mmengine.config import Config

    cfg = Config.fromfile(str(config))

    def patch(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") in {"Resize", "RandomResize"}:
                value["keep_ratio"] = False
            for child in value.values():
                patch(child)
        elif isinstance(value, list):
            for child in value:
                patch(child)

    patch(cfg.test_dataloader.dataset.pipeline)
    preprocessor = cfg.model.get("data_preprocessor")
    if isinstance(preprocessor, dict):
        preprocessor["pad_size_divisor"] = 32
        preprocessor["batch_augments"] = []
    target = run_dir / "patched_config_eval.py"
    cfg.dump(str(target))
    return target


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    args.project.mkdir(parents=True, exist_ok=True)
    patterns = [f"{prefix}/patched_config.py" for model in RUNS.values() for prefix in model.values()]
    patterns += [f"{prefix}/*.pth" for model in RUNS.values() for prefix in model.values()]
    snapshot = Path(snapshot_download(repo_id=args.repo_id, repo_type="dataset", allow_patterns=patterns, local_dir=str(args.project / "hf_snapshot"), token=token))
    annotation = args.data_root / "annotations" / "test.json"
    if not annotation.is_file():
        raise FileNotFoundError(f"Missing COCO test annotations: {annotation}")
    selected_models = [model for model in args.models.split(",") if model]
    unknown = sorted(set(selected_models) - set(RUNS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    all_results: dict[str, Any] = {}
    for model in selected_models:
        for seed, prefix in RUNS[model].items():
            source = snapshot / prefix
            checkpoints = sorted(source.glob("best_*.pth"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint under {source}")
            run_dir = args.project / model / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config = prepare_eval_config(model, source, run_dir)
            result_pkl = run_dir / "predictions.pkl"
            env = os.environ.copy()
            env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            env["PYTHONPATH"] = "/marimo/mmdet_code:/marimo/mmdet_code/mmdetection:" + env.get("PYTHONPATH", "")
            command = [args.python, "/marimo/mmdet_code/mmdetection/tools/test.py", str(config), str(checkpoints[0]), "--work-dir", str(run_dir), "--out", str(result_pkl)]
            subprocess.run(command, check=True, cwd="/marimo/mmdet_code/mmdetection", env=env)
            raw = load(str(result_pkl))
            predictions = normalize_predictions(raw, annotation)
            metrics = coco_metrics(annotation, predictions)
            (run_dir / "predictions_coco.json").write_text(json.dumps(predictions) + "\n")
            (run_dir / "evaluation_metrics_full.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            all_results[f"{model}/seed{seed}"] = metrics
            print(json.dumps({"run": f"{model}/seed{seed}", **metrics}, sort_keys=True), flush=True)
    (args.project / "summary.json").write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
