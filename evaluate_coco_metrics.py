#!/usr/bin/env python3
"""Write a compact AP metric JSON from COCO ground truth and detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def precision(evaluator: COCOeval, iou: float | None, area: str) -> float:
    values = evaluator.eval["precision"]
    if iou is not None:
        values = values[np.where(np.isclose(evaluator.params.iouThrs, iou))[0]]
    area_index = evaluator.params.areaRngLbl.index(area)
    values = values[:, :, :, area_index, -1]
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else -1.0


def load_results(ground_truth: COCO, result_file: Path) -> COCO:
    results = json.loads(result_file.read_text(encoding="utf-8"))
    if results:
        return ground_truth.loadRes(results)
    detections = COCO()
    detections.dataset = {
        "images": ground_truth.dataset["images"],
        "categories": ground_truth.dataset["categories"],
        "annotations": [],
    }
    detections.createIndex()
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--res", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ground_truth = COCO(str(args.gt))
    evaluator = COCOeval(ground_truth, load_results(ground_truth, args.res), "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = {
        "map_50_95": precision(evaluator, None, "all"),
        "ap50": precision(evaluator, 0.50, "all"),
        "ap75": precision(evaluator, 0.75, "all"),
        "ap50_tiny1": None,
        "ap50_tiny2": None,
        "ap50_tiny3": None,
        "ap50_small": precision(evaluator, 0.50, "small"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
