#!/usr/bin/env python3
"""Convert the original Varroa bbox dataset to Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
GT_SOURCES = ("gt_one", "gt_filtered")
CLASS_POLICIES = ("all", "only-class-1", "drop-class-3", "map-3-to-1")


@dataclass(frozen=True)
class SampleRecord:
    """Single source image plus its annotation metadata."""

    source_split: str
    rel_image_path: str
    class_token: str
    boxes: list[list[float]]
    is_positive: bool


def resolve_repo_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    missing = [split for split in SPLITS if not (root / split / "videos").is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing split video folders under {root}: {', '.join(missing)}")
    return root


def source_image_path(root_path: Path, record: SampleRecord) -> Path:
    return root_path / record.source_split / record.rel_image_path


def flattened_name(image_path: Path, image_dir: Path) -> str:
    return "__".join(image_path.relative_to(image_dir).parts)


def output_name_for_record(root_path: Path, record: SampleRecord) -> str:
    image_path = source_image_path(root_path, record)
    image_dir = root_path / record.source_split / "videos"
    return f"{record.source_split}__{flattened_name(image_path, image_dir)}"


def iter_images(image_dir: Path) -> list[Path]:
    images: list[Path] = []
    for suffix in IMAGE_EXTENSIONS:
        images.extend(image_dir.rglob(f"*{suffix}"))
    return sorted(images)


def clamp_xyxy_boxes(boxes: list[list[float]], width: int, height: int) -> list[list[float]]:
    clamped: list[list[float]] = []
    for x1, y1, x2, y2 in boxes:
        left, right = sorted((max(0.0, min(float(x1), width)), max(0.0, min(float(x2), width))))
        top, bottom = sorted((max(0.0, min(float(y1), height)), max(0.0, min(float(y2), height))))
        if right > left and bottom > top:
            clamped.append([left, top, right, bottom])
    return clamped


def yolo_rows_from_boxes(boxes: list[list[float]], width: int, height: int) -> list[str]:
    rows: list[str] = []
    for left, top, right, bottom in boxes:
        box_w = right - left
        box_h = bottom - top
        cx = (left + right) * 0.5 / width
        cy = (top + bottom) * 0.5 / height
        rows.append(f"0 {cx:.8f} {cy:.8f} {box_w / width:.8f} {box_h / height:.8f}")
    return rows


def read_original_boxes(label_path: Path) -> list[list[float]]:
    """Read native label txt files where the first row is a count and later rows are xyxy."""
    if not label_path.exists():
        return []

    boxes: list[list[float]] = []
    for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        if idx == 0:
            continue
        parts = line.split()
        if len(parts) >= 4:
            boxes.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
    return boxes


def label_path_for(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    return label_dir / image_path.relative_to(image_dir).with_suffix(".txt")


def parse_gt_csv(csv_path: Path) -> dict[str, list[list[float]]]:
    """Backward-compatible parser for gt_filtered.csv-style files."""
    return {record.rel_image_path: record.boxes for record in parse_gt_records(csv_path, infer_filtered_positive=True)}


def parse_gt_records(csv_path: Path, *, infer_filtered_positive: bool = False) -> list[SampleRecord]:
    """Parse GT CSV rows into a uniform record structure."""
    if not csv_path.exists():
        return []

    source_split = csv_path.parent.name
    records: list[SampleRecord] = []
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        rel_img_path = parts[0]
        class_token = parts[1]
        values = [float(value) for value in parts[2:]]
        boxes = [values[idx : idx + 4] for idx in range(0, len(values) - 3, 4)]
        if infer_filtered_positive and boxes:
            class_token = "1"
        is_positive = bool(boxes) and class_token != "0"
        records.append(
            SampleRecord(
                source_split=source_split,
                rel_image_path=rel_img_path,
                class_token=class_token,
                boxes=boxes,
                is_positive=is_positive,
            )
        )
    return records


def load_records(root_path: Path, gt_source: str) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for split in SPLITS:
        csv_path = root_path / split / f"{gt_source}.csv"
        records.extend(parse_gt_records(csv_path, infer_filtered_positive=(gt_source == "gt_filtered")))
    return records


def filter_records(
    records: list[SampleRecord],
    *,
    only_positives: bool,
    class_policy: str,
) -> list[SampleRecord]:
    filtered: list[SampleRecord] = []
    for record in records:
        if only_positives and not record.is_positive:
            continue
        if class_policy == "only-class-1" and record.class_token != "1":
            continue
        if class_policy == "drop-class-3" and record.class_token == "3":
            continue
        if class_policy == "map-3-to-1" and record.class_token == "3":
            record = SampleRecord(
                source_split=record.source_split,
                rel_image_path=record.rel_image_path,
                class_token="1",
                boxes=record.boxes,
                is_positive=record.is_positive,
            )
        filtered.append(record)
    return filtered


def dedupe_records(records: list[SampleRecord]) -> list[SampleRecord]:
    deduped: list[SampleRecord] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record.source_split, record.rel_image_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def assign_output_splits(
    records: list[SampleRecord],
    *,
    seed: int,
) -> dict[str, list[SampleRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    val_end = round(total * 0.15)
    test_end = val_end + round(total * 0.15)

    return {
        "train": shuffled[test_end:],
        "val": shuffled[:val_end],
        "test": shuffled[val_end:test_end],
    }


def write_data_yaml(
    out_dir: Path,
    *,
    split_counts: dict[str, int],
    gt_source: str,
    only_positives: bool,
    class_policy: str,
    seed: int,
) -> Path:
    yaml_path = out_dir / "varroa.yaml"
    splits_line = (
        f"split_counts: {{train: {split_counts['train']}, "
        f"val: {split_counts['val']}, test: {split_counts['test']}}}"
    )
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names: [varroa]",
                splits_line,
                f"gt_source: {gt_source}",
                f"only_positives: {str(only_positives).lower()}",
                f"class_policy: {class_policy}",
                f"seed: {seed}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def prepare_dataset(
    root: str | Path = ".",
    out_dir: str | Path = "yolo_related/datasets/varroa_yolo",
    *,
    overwrite_labels: bool = True,
    copy_images: bool = True,
    limit: int = 0,
    gt_source: str = "gt_one",
    only_positives: bool = True,
    class_policy: str = "map-3-to-1",
    seed: int = 42,
) -> Path:
    if gt_source not in GT_SOURCES:
        raise ValueError(f"Unsupported gt_source: {gt_source}")
    if class_policy not in CLASS_POLICIES:
        raise ValueError(f"Unsupported class_policy: {class_policy}")

    root_path = resolve_repo_root(root)
    out_path = Path(out_dir).expanduser()
    if not out_path.is_absolute():
        out_path = (root_path / out_path).resolve()

    records = load_records(root_path, gt_source)
    records = filter_records(records, only_positives=only_positives, class_policy=class_policy)
    records = dedupe_records(records)
    split_records = assign_output_splits(records, seed=seed)

    split_counts: dict[str, int] = {}
    for split in SPLITS:
        records_for_split = split_records[split]
        if limit > 0:
            records_for_split = records_for_split[:limit]
            split_records[split] = records_for_split
        split_counts[split] = len(records_for_split)

        yolo_image_dir = out_path / "images" / split
        yolo_label_dir = out_path / "labels" / split
        if copy_images and yolo_image_dir.exists():
            shutil.rmtree(yolo_image_dir)
        if overwrite_labels and yolo_label_dir.exists():
            shutil.rmtree(yolo_label_dir)
        yolo_image_dir.mkdir(parents=True, exist_ok=True)
        yolo_label_dir.mkdir(parents=True, exist_ok=True)

        for record in records_for_split:
            image_path = source_image_path(root_path, record)
            dst_name = output_name_for_record(root_path, record)
            dst_image = yolo_image_dir / dst_name
            dst_label = yolo_label_dir / Path(dst_name).with_suffix(".txt").name

            if copy_images:
                shutil.copy2(image_path, dst_image)

            if overwrite_labels or not dst_label.exists():
                with Image.open(image_path) as image:
                    width, height = image.size
                boxes = clamp_xyxy_boxes(record.boxes, width, height)
                dst_label.write_text("\n".join(yolo_rows_from_boxes(boxes, width, height)), encoding="utf-8")

    return write_data_yaml(
        out_path,
        split_counts=split_counts,
        gt_source=gt_source,
        only_positives=only_positives,
        class_policy=class_policy,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Varroa labels to Ultralytics YOLO format.")
    parser.add_argument("--root", default=".", help="Repo root containing train/val/test folders.")
    parser.add_argument("--out-dir", default="yolo_related/datasets/varroa_yolo")
    parser.add_argument("--limit", type=int, default=0, help="Optional max images per split for smoke tests.")
    parser.add_argument("--no-copy-images", action="store_true", help="Only write labels/YAML; do not copy images.")
    parser.add_argument("--keep-labels", action="store_true", help="Do not overwrite existing YOLO label files.")
    parser.add_argument("--gt-source", default="gt_one", choices=GT_SOURCES)
    parser.add_argument(
        "--only-positives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only positive rows. Use --no-only-positives to keep token 0 rows.",
    )
    parser.add_argument("--class-policy", default="map-3-to-1", choices=CLASS_POLICIES)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml_path = prepare_dataset(
        args.root,
        args.out_dir,
        overwrite_labels=not args.keep_labels,
        copy_images=not args.no_copy_images,
        limit=args.limit,
        gt_source=args.gt_source,
        only_positives=args.only_positives,
        class_policy=args.class_policy,
        seed=args.seed,
    )
    print(f"YOLO dataset ready: {yaml_path}")
    print(yaml_path.read_text(encoding='utf-8'))


if __name__ == "__main__":
    main()
