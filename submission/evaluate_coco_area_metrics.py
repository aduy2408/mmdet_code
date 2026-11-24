#!/usr/bin/env python3
"""Compute COCO AP by IoU and object-size area from MMDetection predictions."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


AREA_RANGES = {
    "all": [0**2, 1e10**2],
    "small": [0**2, 32**2],
    "medium": [32**2, 96**2],
    "large": [96**2, 1e10**2],
}


def tensor_to_list(value):
    return value.detach().cpu().tolist() if hasattr(value, "detach") else value


def predictions_to_coco(predictions):
    results = []
    for item in predictions:
        instances = item["pred_instances"]
        boxes = tensor_to_list(instances["bboxes"])
        scores = tensor_to_list(instances["scores"])
        labels = tensor_to_list(instances["labels"])
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            results.append(
                {
                    "image_id": int(item["img_id"]),
                    "category_id": int(label) + 1,
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )
    return results


def evaluate(coco, results, iou_thresholds, max_dets):
    prediction = coco.loadRes(results)
    evaluator = COCOeval(coco, prediction, "bbox")
    evaluator.params.iouThrs = np.array(iou_thresholds)
    evaluator.params.areaRng = list(AREA_RANGES.values())
    evaluator.params.areaRngLbl = list(AREA_RANGES)
    evaluator.params.maxDets = [1, 10, max_dets]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    max_det_index = len(evaluator.params.maxDets) - 1
    output = {}
    for area_index, area_name in enumerate(AREA_RANGES):
        values = precision[:, :, :, area_index, max_det_index]
        values = values[values > -1]
        output[area_name] = float(np.mean(values)) if values.size else None
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-dets", type=int, default=1000)
    args = parser.parse_args()

    coco = COCO(args.annotation)
    with Path(args.predictions).open("rb") as handle:
        predictions = pickle.load(handle)
    results = predictions_to_coco(predictions)
    metrics = {
        "AP50": evaluate(coco, results, [0.5], args.max_dets),
        "mAP50-95": evaluate(coco, results, np.arange(0.5, 0.96, 0.05), args.max_dets),
        "prediction_count": len(results),
        "annotation": str(Path(args.annotation).resolve()),
        "predictions": str(Path(args.predictions).resolve()),
        "max_dets": args.max_dets,
    }
    Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
