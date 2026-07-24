#!/usr/bin/env python3
"""Screen an FCOS checkpoint for translation-phase sensitivity on LEVIR-Ship."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import mmcv
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OFFSETS = ((0, 0), (1, 0), (2, 0), (4, 0), (0, 1), (0, 2), (0, 4), (4, 4))
OBSERVATION_FIELDS = (
    "image_id",
    "file_name",
    "gt_id",
    "gt_width",
    "gt_height",
    "gt_area",
    "size_bin",
    "dx",
    "dy",
    "phase_x",
    "phase_y",
    "detected",
    "confidence",
    "iou",
    "center_error",
    "normalized_center_error",
)
SUMMARY_FIELDS = (
    "image_id",
    "file_name",
    "gt_id",
    "gt_width",
    "gt_height",
    "gt_area",
    "size_bin",
    "num_offsets",
    "detections",
    "score_mean",
    "score_std",
    "score_range",
    "miss_rate",
    "iou_mean",
    "iou_std",
    "worst_iou",
    "center_error_mean",
    "normalized_center_error_mean",
    "normalized_center_error_std",
)


def translate_image(
    image: np.ndarray, dx: int, dy: int, fill_value: Iterable[float] | float
) -> np.ndarray:
    """Translate an HWC image without wrapping pixels."""
    if image.ndim != 3:
        raise ValueError("image must have shape [H, W, C]")
    height, width = image.shape[:2]
    fill = np.asarray(fill_value, dtype=image.dtype)
    shifted = np.empty_like(image)
    shifted[...] = fill

    src_x1, src_x2 = max(-dx, 0), min(width - dx, width)
    src_y1, src_y2 = max(-dy, 0), min(height - dy, height)
    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1, dst_y1 = src_x1 + dx, src_y1 + dy
        shifted[
            dst_y1 : dst_y1 + (src_y2 - src_y1),
            dst_x1 : dst_x1 + (src_x2 - src_x1),
        ] = image[src_y1:src_y2, src_x1:src_x2]
    return shifted


def translate_boxes(boxes: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.asarray(boxes, dtype=np.float32).copy()
    shifted[:, [0, 2]] += dx
    shifted[:, [1, 3]] += dy
    return shifted


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=1)
    box_area = np.prod(np.maximum(box[2:] - box[:2], 0))
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return intersection / np.maximum(box_area + areas - intersection, 1e-12)


def match_prediction(
    gt_box: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    expansion: float = 1.5,
) -> dict[str, float | bool | None]:
    """Match the best prediction using IoU or the expanded-GT center rule."""
    if len(pred_boxes) == 0:
        return _miss()
    ious = box_iou(gt_box, pred_boxes)
    center = (gt_box[:2] + gt_box[2:]) / 2
    half_size = (gt_box[2:] - gt_box[:2]) * expansion / 2
    pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
    center_inside = np.all(
        (pred_centers >= center - half_size) & (pred_centers <= center + half_size),
        axis=1,
    )
    candidates = np.flatnonzero((ious > 0.1) | center_inside)
    if len(candidates) == 0:
        return _miss()
    index = int(candidates[np.argmax(ious[candidates])])
    center_error = float(np.linalg.norm(pred_centers[index] - center))
    scale = math.sqrt(
        max(float((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])), 0.0)
    )
    return {
        "detected": True,
        "confidence": float(pred_scores[index]),
        "iou": float(ious[index]),
        "center_error": center_error,
        "normalized_center_error": center_error / (scale + 1e-6),
    }


def _miss() -> dict[str, float | bool | None]:
    return {
        "detected": False,
        "confidence": 0.0,
        "iou": 0.0,
        "center_error": None,
        "normalized_center_error": None,
    }


def size_bin(width: float, height: float) -> str:
    radius = math.sqrt(width * height)
    if radius <= 8:
        return "tiny-1"
    if radius <= 16:
        return "tiny-2"
    if radius <= 32:
        return "small"
    return "medium-large"


def load_selection(annotation_file: Path, max_images: int, border: int) -> list[dict]:
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    annotations = defaultdict(list)
    for ann in payload["annotations"]:
        annotations[ann["image_id"]].append(ann)
    images = sorted(
        (image for image in payload["images"] if annotations[image["id"]]),
        key=lambda image: (-len(annotations[image["id"]]), image["id"]),
    )[:max_images]
    selected = []
    for image in images:
        valid = []
        for ann in annotations[image["id"]]:
            x, y, width, height = map(float, ann["bbox"])
            box = np.array([x, y, x + width, y + height], dtype=np.float32)
            if (
                box[0] >= border
                and box[1] >= border
                and box[2] <= image["width"] - border
                and box[3] <= image["height"] - border
            ):
                valid.append((ann, box))
        if valid:
            selected.append({**image, "ground_truth": valid})
    return selected


def summarize_objects(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[(row["image_id"], row["gt_id"])].append(row)
    summaries = []
    for rows in grouped.values():
        first = rows[0]
        scores = np.array([row["confidence"] for row in rows], dtype=float)
        ious = np.array([row["iou"] for row in rows], dtype=float)
        detected = np.array([row["detected"] for row in rows], dtype=bool)
        centers = np.array(
            [row["center_error"] for row in rows if row["center_error"] is not None],
            dtype=float,
        )
        norm_centers = np.array(
            [
                row["normalized_center_error"]
                for row in rows
                if row["normalized_center_error"] is not None
            ],
            dtype=float,
        )
        summaries.append(
            {
                **{key: first[key] for key in SUMMARY_FIELDS[:7]},
                "num_offsets": len(rows),
                "detections": int(detected.sum()),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "score_range": float(scores.max() - scores.min()),
                "miss_rate": float(1 - detected.mean()),
                "iou_mean": float(ious.mean()),
                "iou_std": float(ious.std()),
                "worst_iou": float(ious.min()),
                "center_error_mean": _mean_or_none(centers),
                "normalized_center_error_mean": _mean_or_none(norm_centers),
                "normalized_center_error_std": _std_or_none(norm_centers),
            }
        )
    return summaries


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def _std_or_none(values: np.ndarray) -> float | None:
    return float(values.std()) if len(values) else None


def aggregate_by_size(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for name in ("tiny-1", "tiny-2", "small", "medium-large"):
        rows = [row for row in summaries if row["size_bin"] == name]
        output[name] = {"objects": len(rows)}
        for metric in ("score_std", "score_range", "miss_rate", "iou_mean"):
            values = [row[metric] for row in rows if row[metric] is not None]
            output[name][f"{metric}_mean"] = (
                float(np.mean(values)) if values else None
            )
            output[name][f"{metric}_median"] = (
                float(np.median(values)) if values else None
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(
    output_dir: Path,
    observations: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    strongest = max(summaries, key=lambda row: row["score_range"])
    rows = [
        row
        for row in observations
        if row["image_id"] == strongest["image_id"] and row["gt_id"] == strongest["gt_id"]
    ]
    heatmap = np.full((5, 5), np.nan)
    for row in rows:
        heatmap[row["dy"], row["dx"]] = row["confidence"]
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(heatmap, origin="lower", vmin=0, vmax=1, cmap="viridis")
    for row in rows:
        axis.text(
            row["dx"],
            row["dy"],
            f'{row["confidence"]:.2f}',
            ha="center",
            va="center",
            color="white",
            fontsize=8,
        )
    axis.set(title=f'Confidence: image {strongest["image_id"]}, GT {strongest["gt_id"]}', xlabel="dx", ylabel="dy")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_heatmap.png", dpi=180)
    plt.close(fig)

    bins = ("tiny-1", "tiny-2", "small", "medium-large")
    for metric, filename, ylabel in (
        ("score_range", "score_range_by_size.png", "Confidence range"),
        ("miss_rate", "miss_rate_by_size.png", "Miss rate"),
    ):
        data = [
            [row[metric] for row in summaries if row["size_bin"] == name]
            for name in bins
        ]
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.boxplot(data, tick_labels=bins, showfliers=False)
        axis.set(xlabel="Object size bin", ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "mmdetection/work_dirs/levir_baseline/fcos/patched_config.py",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "best_coco_bbox_mAP_epoch_12.pth"
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=root / "mmdetection/data/levir_ship_coco/annotations/test.json",
    )
    parser.add_argument(
        "--image-root", type=Path, default=root / "LevirShipData/All Images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "mmdetection/outputs/levir_phase_screening",
    )
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--border", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--offsets",
        nargs="+",
        default=[f"{dx},{dy}" for dx, dy in OFFSETS],
        help="Offsets as space-separated dx,dy pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_images < 1:
        raise ValueError("--max-images must be >= 1")
    offsets = [tuple(map(int, value.split(","))) for value in args.offsets]
    for path in (args.config, args.checkpoint, args.ann_file, args.image_root):
        if not path.exists():
            raise FileNotFoundError(path)

    mmdet_root = Path(__file__).resolve().parent / "mmdetection"
    sys.path.insert(0, str(mmdet_root))
    from mmdet.apis import inference_detector, init_detector

    model = init_detector(str(args.config), str(args.checkpoint), device=args.device)
    mean = np.rint(np.asarray(model.cfg.model.data_preprocessor.mean)).astype(np.uint8)
    selection = load_selection(args.ann_file, args.max_images, args.border)
    if not selection:
        raise ValueError("No images contain GT boxes that pass the border filter")

    observations = []
    for image_index, image_info in enumerate(selection, 1):
        image = mmcv.imread(str(args.image_root / image_info["file_name"]))
        if image is None:
            raise FileNotFoundError(args.image_root / image_info["file_name"])
        for dx, dy in offsets:
            shifted = translate_image(image, dx, dy, mean)
            result = inference_detector(model, shifted).pred_instances.cpu()
            pred_boxes = result.bboxes.numpy()
            pred_boxes[:, [0, 2]] -= dx
            pred_boxes[:, [1, 3]] -= dy
            pred_scores = result.scores.numpy()
            pred_labels = result.labels.numpy()
            ship = pred_labels == 0
            for annotation, gt_box in image_info["ground_truth"]:
                match = match_prediction(gt_box, pred_boxes[ship], pred_scores[ship])
                width, height = gt_box[2:] - gt_box[:2]
                observations.append(
                    {
                        "image_id": image_info["id"],
                        "file_name": image_info["file_name"],
                        "gt_id": annotation["id"],
                        "gt_width": float(width),
                        "gt_height": float(height),
                        "gt_area": float(width * height),
                        "size_bin": size_bin(float(width), float(height)),
                        "dx": dx,
                        "dy": dy,
                        "phase_x": float(
                            ((gt_box[0] + gt_box[2]) / 2 + dx) % 8
                        ),
                        "phase_y": float(
                            ((gt_box[1] + gt_box[3]) / 2 + dy) % 8
                        ),
                        **match,
                    }
                )
        print(f"[{image_index}/{len(selection)}] {image_info['file_name']}", flush=True)

    summaries = summarize_objects(observations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "observations.csv", observations, OBSERVATION_FIELDS)
    write_csv(args.output_dir / "object_summary.csv", summaries, SUMMARY_FIELDS)
    report = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "annotation_file": str(args.ann_file.resolve()),
        "selected_images": len(selection),
        "eligible_objects": len(summaries),
        "offsets": offsets,
        "border": args.border,
        "size_bins": aggregate_by_size(summaries),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    plot_outputs(args.output_dir, observations, summaries)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
