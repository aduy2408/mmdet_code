#!/usr/bin/env python3
"""Count YOLO Task-Aligned Assigner positives by detection level on LEVIR-Ship."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml

import train_all_levir_baseline as levir


CHECKPOINTS = {
    "yolov8n_p3_p5": "yolov8n_seed42_best.pt",
    "yolov8n_p2_baseline": "yolov8n_p2_baseline_seed42_best.pt",
    "yolov8n_p2_offset": "yolov8n_p2_offset_seed42_best.pt",
}


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "LevirShipData")
    parser.add_argument("--artifacts", type=Path, default=root / "artifacts/yolo_assignment_levels")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Validation images; 0 means all.")
    return parser.parse_args()


def write_split(data_root: Path, artifacts: Path) -> list[tuple[Path, Path, str]]:
    samples = levir.discover_samples(data_root)
    split = levir.split_by_scene(samples, seed=42)
    names = list(split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert not {sample[2] for sample in split[left]} & {sample[2] for sample in split[right]}
    split_dir = artifacts / "split_seed42"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in split.items():
        (split_dir / f"{name}.txt").write_text("\n".join(str(row[0]) for row in rows) + "\n")
    (split_dir / "levir_ship.yaml").write_text(yaml.safe_dump({
        "path": str(data_root), "train": str(split_dir / "train.txt"),
        "val": str(split_dir / "val.txt"), "test": str(split_dir / "test.txt"),
        "names": {0: "ship"},
    }, sort_keys=False))
    return split["val"]


def letterbox(image: np.ndarray, labels: list[list[float]], size: int):
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized = cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_LINEAR)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    boxes = []
    for _, cx, cy, box_w, box_h in labels:
        boxes.append(((cx * width * ratio + left) / size, (cy * height * ratio + top) / size,
                      box_w * width * ratio / size, box_h * height * ratio / size))
    return canvas, boxes


def batches(rows, batch_size, size):
    for start in range(0, len(rows), batch_size):
        images, indices, classes, boxes = [], [], [], []
        for batch_index, (image_path, label_path, _) in enumerate(rows[start : start + batch_size]):
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            height, width = image.shape[:2]
            labels = [list(map(float, line.split())) for line in label_path.read_text().splitlines() if line.strip()]
            image, mapped_boxes = letterbox(image, labels, size)
            images.append(image[:, :, ::-1].copy())
            indices.extend([batch_index] * len(mapped_boxes))
            classes.extend([0] * len(mapped_boxes))
            boxes.extend(mapped_boxes)
        yield {
            "img": torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255),
            "batch_idx": torch.tensor(indices, dtype=torch.long),
            "cls": torch.tensor(classes, dtype=torch.float32).view(-1, 1),
            "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }


def load_model(path: Path, source: Path, device: torch.device):
    sys.path.insert(0, str(source / "models_related/ultralytics"))
    import ultralytics  # Register checkpoint classes, including P2OffsetRegression.
    del ultralytics
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint["model"].to(device).float()
    model.args = SimpleNamespace(**checkpoint["train_args"])
    model.train()
    return model, model.init_criterion()


def measure(name, path, rows, args):
    source = args.artifacts / "yolo_code"
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    model, criterion = load_model(path, source, device)
    strides = [int(value) for value in model.model[-1].stride.tolist()]
    totals, gt_count, image_count, positive_count = Counter(), 0, 0, 0
    with torch.no_grad():
        for number, batch in enumerate(batches(rows, args.batch_size, args.imgsz), 1):
            gt_count += len(batch["bboxes"])
            image_count += len(batch["img"])
            batch = {key: value.to(device) for key, value in batch.items()}
            preds = criterion.parse_output(model(batch["img"]))
            (fg_mask, *_), _, _ = criterion.get_assigned_targets_and_loss(preds, batch)
            positive_count += int(fg_mask.sum().item())
            level_sizes = [feature.shape[2] * feature.shape[3] for feature in preds["feats"]]
            assert sum(level_sizes) == fg_mask.shape[1]
            offset = 0
            for stride, count in zip(strides, level_sizes):
                totals[f"P{int(np.log2(stride))}"] += int(fg_mask[:, offset : offset + count].sum().item())
                offset += count
            if number % 25 == 0:
                print(f"{name}: {image_count}/{len(rows)} validation images")
    assert sum(totals.values()) == positive_count
    total = sum(totals.values())
    levels = [f"P{int(np.log2(stride))}" for stride in strides]
    return {"model": name, "images": image_count, "gt_boxes": gt_count, "total_positives": total,
            "dominant_level": max(levels, key=lambda level: totals[level]),
            **{level: totals[level] for level in levels},
            **{f"{level}_pct": round(100 * totals[level] / total, 3) if total else 0.0 for level in levels}}


def main():
    args = parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    rows = write_split(args.data_root, args.artifacts)
    rows = rows[: args.limit] if args.limit else rows
    checkpoints = args.artifacts / "checkpoints"
    results = [measure(name, checkpoints / filename, rows, args) for name, filename in CHECKPOINTS.items()]
    fields = sorted({key for result in results for key in result})
    with (args.artifacts / "assignment_levels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)
    (args.artifacts / "assignment_levels.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
