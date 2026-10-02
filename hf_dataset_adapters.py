"""Dataset adapters for the Hugging Face DETR/RT-DETR matrix.

The adapters intentionally return one common record shape so the model runner
never depends on MMDetection configs. Split policy follows the existing
baseline launchers: LEVIR by scene, TinyPerson by source image, Varroa by its
canonical train/val/test folders, and VisDrone by official split.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    boxes: tuple[tuple[float, float, float, float], ...]
    labels: tuple[int, ...]
    image_id: int
    source_group: str


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    num_labels: int
    label_names: tuple[str, ...]
    splits: dict[str, tuple[DetectionSample, ...]]


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _safe_box(x: float, y: float, w: float, h: float, width: int, height: int):
    x = max(0.0, min(float(x), width))
    y = max(0.0, min(float(y), height))
    w = max(0.0, min(float(w), width - x))
    h = max(0.0, min(float(h), height - y))
    return (x, y, w, h) if w > 0 and h > 0 else None


def _split_levir(root: Path, split_seed: int, limit: int) -> dict[str, tuple[DetectionSample, ...]]:
    from train_all_levir_baseline import discover_samples, split_by_scene, yolo_boxes

    grouped = split_by_scene(discover_samples(root), split_seed)
    output = {}
    image_id = 1
    for split, values in grouped.items():
        selected = values[:limit] if limit > 0 else values
        records = []
        for image_path, annotation_path, scene in selected:
            width, height = _image_size(image_path)
            boxes = tuple(tuple(box) for box in yolo_boxes(annotation_path, width, height))
            records.append(DetectionSample(image_path, boxes, (0,) * len(boxes), image_id, scene))
            image_id += 1
        output[split] = tuple(records)
    return output


def _split_varroa(root: Path, split_seed: int, limit: int) -> dict[str, tuple[DetectionSample, ...]]:
    output = {}
    image_id = 1
    for split in ("train", "val", "test"):
        csv_path = root / split / "gt_one.csv"
        records = []
        for line in csv_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            image_path = root / split / parts[0]
            if not image_path.is_file():
                continue
            width, height = _image_size(image_path)
            values = [float(value) for value in parts[2:]]
            boxes = []
            for index in range(0, len(values) - 3, 4):
                x1, y1, x2, y2 = values[index:index + 4]
                box = _safe_box(x1, y1, x2 - x1, y2 - y1, width, height)
                if box is not None:
                    boxes.append(box)
            records.append(DetectionSample(image_path, tuple(boxes), (0,) * len(boxes), image_id, split))
            image_id += 1
        output[split] = tuple(records[:limit] if limit > 0 else records)
    return output


def _load_coco_records(annotation_path: Path, image_root: Path, limit: int = 0):
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations = {}
    for ann in data.get("annotations", []):
        annotations.setdefault(ann["image_id"], []).append(ann)
    records = []
    for image in data.get("images", []):
        image_path = image_root / image["file_name"]
        if not image_path.is_file():
            continue
        boxes, labels = [], []
        for ann in annotations.get(image["id"], []):
            x, y, w, h = ann["bbox"]
            box = _safe_box(x, y, w, h, image["width"], image["height"])
            if box is not None:
                boxes.append(box)
                labels.append(int(ann.get("category_id", 1)) - 1)
        records.append(DetectionSample(image_path, tuple(boxes), tuple(labels), int(image["id"]), image["file_name"]))
        if limit > 0 and len(records) >= limit:
            break
    return records


def _split_tinyperson(root: Path, split_seed: int, limit: int) -> dict[str, tuple[DetectionSample, ...]]:
    train_ann = root / "annotations/corner/task/tiny_set_train_sw640_sh512_all.json"
    test_ann = root / "annotations/corner/task/tiny_set_test_sw640_sh512_all.json"
    train_images = root / "erase_with_uncertain_dataset/train"
    test_images = root / "test"
    data = json.loads(train_ann.read_text(encoding="utf-8"))
    sources = sorted(data.get("old_images", []), key=lambda item: item["file_name"])
    random.Random(split_seed).shuffle(sources)
    val_count = max(1, round(len(sources) * 0.15))
    val_names = {item["file_name"] for item in sources[:val_count]}
    image_names = {item["file_name"] for item in data.get("images", [])}
    train_data = dict(data)
    train_data["images"] = [item for item in data.get("images", []) if item["file_name"] not in val_names]
    train_ids = {item["id"] for item in train_data["images"]}
    train_data["annotations"] = [item for item in data.get("annotations", []) if item["image_id"] in train_ids]
    val_data = dict(data)
    val_data["images"] = [item for item in data.get("images", []) if item["file_name"] in val_names]
    val_ids = {item["id"] for item in val_data["images"]}
    val_data["annotations"] = [item for item in data.get("annotations", []) if item["image_id"] in val_ids]

    def records(coco, image_root, cap):
        temp = root / "hf_matrix_tinyperson_tmp.json"
        temp.write_text(json.dumps(coco), encoding="utf-8")
        try:
            return _load_coco_records(temp, image_root, cap)
        finally:
            temp.unlink(missing_ok=True)

    test_records = _load_coco_records(test_ann, test_images, limit)
    return {
        "train": tuple(records(train_data, train_images, limit)),
        "val": tuple(records(val_data, train_images, limit)),
        "test": tuple(test_records),
    }


def _split_visdrone(root: Path, split_seed: int, limit: int) -> dict[str, tuple[DetectionSample, ...]]:
    names = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
    output = {}
    image_id = 1
    for split, directory in names.items():
        source = root / directory
        records = []
        for ann_path in sorted((source / "annotations").glob("*.txt")):
            image_path = source / "images" / f"{ann_path.stem}.jpg"
            if not image_path.is_file():
                image_path = source / "images" / f"{ann_path.stem}.png"
            if not image_path.is_file():
                continue
            width, height = _image_size(image_path)
            boxes, labels = [], []
            for line in ann_path.read_text(encoding="utf-8").splitlines():
                values = [int(value.strip()) for value in line.split(",")[:6]]
                if len(values) < 6:
                    continue
                x, y, w, h, score, category = values
                if score == 0 or category not in range(1, 11):
                    continue
                box = _safe_box(x, y, w, h, width, height)
                if box is not None:
                    boxes.append(box)
                    labels.append(category - 1)
            records.append(DetectionSample(image_path, tuple(boxes), tuple(labels), image_id, split))
            image_id += 1
        output[split] = tuple(records[:limit] if limit > 0 else records)
    return output


def load_dataset(name: str, root: str | Path, split_seed: int = 42, limit: int = 0) -> DatasetBundle:
    root = Path(root).expanduser().resolve()
    if name == "levirship":
        splits = _split_levir(root, split_seed, limit)
        return DatasetBundle(name, 1, ("ship",), splits)
    if name == "varroa":
        splits = _split_varroa(root, split_seed, limit)
        return DatasetBundle(name, 1, ("varroa",), splits)
    if name == "tinyperson":
        splits = _split_tinyperson(root, split_seed, limit)
        return DatasetBundle(name, 1, ("person",), splits)
    if name == "visdrone":
        splits = _split_visdrone(root, split_seed, limit)
        return DatasetBundle(name, 10, ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"), splits)
    raise ValueError(f"Unknown dataset: {name}")
