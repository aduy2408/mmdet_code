#!/usr/bin/env python3
"""Merge TinyPerson tile detections and write requested AP metrics as JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, width, height = box
    return np.array([x, y, x + width, y + height], dtype=float)


def nms(detections: list[dict], threshold: float) -> list[dict]:
    """Class-aware NumPy NMS matching TinyBenchmark's merge step."""
    kept: list[dict] = []
    by_category: dict[int, list[dict]] = defaultdict(list)
    for detection in detections:
        by_category[detection["category_id"]].append(detection)
    for category_detections in by_category.values():
        ordered = sorted(
            category_detections, key=lambda item: item["score"], reverse=True
        )
        while ordered:
            best = ordered.pop(0)
            kept.append(best)
            best_box = xywh_to_xyxy(best["bbox"])
            best_area = max(0, best_box[2] - best_box[0]) * max(
                0, best_box[3] - best_box[1]
            )
            remaining = []
            for candidate in ordered:
                box = xywh_to_xyxy(candidate["bbox"])
                intersection = max(
                    0, min(best_box[2], box[2]) - max(best_box[0], box[0])
                ) * max(0, min(best_box[3], box[3]) - max(best_box[1], box[1]))
                area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
                union = best_area + area - intersection
                if union <= 0 or intersection / union <= threshold:
                    remaining.append(candidate)
            ordered = remaining
    return kept


def merge_detections(
    result_file: Path, corner_gt_file: Path, output_file: Path
) -> Path:
    detections = json.loads(result_file.read_text(encoding="utf-8"))
    corner_gt = json.loads(corner_gt_file.read_text(encoding="utf-8"))
    filename_to_source = {
        image["file_name"]: image["id"] for image in corner_gt["old_images"]
    }
    crop_info = {image["id"]: image for image in corner_gt["images"]}
    merged: dict[int, list[dict]] = defaultdict(list)
    for detection in detections:
        image = crop_info[detection["image_id"]]
        x1, y1, _, _ = image["corner"]
        translated = dict(detection)
        translated["image_id"] = filename_to_source[image["file_name"]]
        translated["bbox"] = [
            detection["bbox"][0] + x1,
            detection["bbox"][1] + y1,
            detection["bbox"][2],
            detection["bbox"][3],
        ]
        merged[translated["image_id"]].append(translated)
    output = []
    for source_detections in merged.values():
        output.extend(nms(source_detections, 0.5))
    output_file.write_text(json.dumps(output), encoding="utf-8")
    return output_file


def prepare_ground_truth(source: Path, output: Path) -> Path:
    """Map TinyPerson ignore regions to COCO crowd regions.

    COCO crowd matching uses intersection-over-detection for ignored regions,
    matching TinyBenchmark's ``USE_IOD_FOR_IGNORE`` behavior.
    """
    data = json.loads(source.read_text(encoding="utf-8"))
    for annotation in data["annotations"]:
        ignored = any(
            annotation.get(flag, False) for flag in ("ignore", "uncertain", "logo")
        )
        annotation["iscrowd"] = int(ignored)
    output.write_text(json.dumps(data), encoding="utf-8")
    return output


def evaluate(
    gt_file: Path,
    result_file: Path,
    iou_thresholds: np.ndarray,
    area_ranges: list[list[float]],
    area_labels: list[str],
) -> COCOeval:
    ground_truth = COCO(str(gt_file))
    results = json.loads(result_file.read_text(encoding="utf-8"))
    if results:
        detections = ground_truth.loadRes(results)
    else:
        detections = COCO()
        detections.dataset = {
            "images": ground_truth.dataset["images"],
            "categories": ground_truth.dataset["categories"],
            "annotations": [],
        }
        detections.createIndex()
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.iouThrs = iou_thresholds
    evaluator.params.maxDets = [1, 10, 200]
    evaluator.params.areaRng = area_ranges
    evaluator.params.areaRngLbl = area_labels
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator


def mean_precision(evaluator: COCOeval, iou: float | None, area: str) -> float:
    precision = evaluator.eval["precision"]
    if iou is not None:
        indices = np.where(np.isclose(evaluator.params.iouThrs, iou))[0]
        if not len(indices):
            raise ValueError(f"IoU {iou} is absent from evaluation thresholds")
        precision = precision[indices]
    area_index = evaluator.params.areaRngLbl.index(area)
    precision = precision[:, :, :, area_index, -1]
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res", type=Path, required=True)
    parser.add_argument("--corner-gt", type=Path, required=True)
    parser.add_argument("--merged-gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged_result = args.out.with_name(f"{args.out.stem}_merged_detections.json")
    prepared_gt = args.out.with_name(f"{args.out.stem}_prepared_ground_truth.json")
    merge_detections(args.res, args.corner_gt, merged_result)
    prepare_ground_truth(args.merged_gt, prepared_gt)

    tiny = evaluate(
        prepared_gt,
        merged_result,
        np.array([0.25, 0.50, 0.75]),
        [
            [1**2, 1e5**2],
            [1**2, 20**2],
            [1**2, 8**2],
            [8**2, 12**2],
            [12**2, 20**2],
            [20**2, 32**2],
            [32**2, 1e5**2],
        ],
        ["all", "tiny", "tiny1", "tiny2", "tiny3", "small", "reasonable"],
    )
    coco = evaluate(
        prepared_gt,
        merged_result,
        np.linspace(0.50, 0.95, 10),
        [[0, 1e5**2], [0, 32**2], [32**2, 96**2], [96**2, 1e5**2]],
        ["all", "small", "medium", "large"],
    )
    metrics = {
        "map_50_95": mean_precision(coco, None, "all"),
        "ap50": mean_precision(tiny, 0.50, "all"),
        "ap75": mean_precision(tiny, 0.75, "all"),
        "ap50_tiny1": mean_precision(tiny, 0.50, "tiny1"),
        "ap50_tiny2": mean_precision(tiny, 0.50, "tiny2"),
        "ap50_tiny3": mean_precision(tiny, 0.50, "tiny3"),
        "ap50_small": mean_precision(tiny, 0.50, "small"),
    }
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
