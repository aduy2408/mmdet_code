#!/usr/bin/env python3
"""Run the YOLO DGFE control on the fixed MMDetection positive split."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def export_yolo(coco_root: Path, output: Path) -> Path:
    for split in ('train', 'val'):
        annotation = json.loads(
            (coco_root / 'annotations' / f'{split}.json').read_text())
        annotations = {}
        for item in annotation['annotations']:
            annotations.setdefault(item['image_id'], []).append(item['bbox'])
        image_dir = output / 'images' / split
        label_dir = output / 'labels' / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image in annotation['images']:
            src = coco_root / 'images' / split / image['file_name']
            dst = image_dir / image['file_name']
            if not dst.exists():
                shutil.copy2(src, dst)
            rows = []
            for x, y, w, h in annotations.get(image['id'], []):
                rows.append(
                    f"0 {(x + w / 2) / image['width']:.8f} "
                    f"{(y + h / 2) / image['height']:.8f} "
                    f"{w / image['width']:.8f} {h / image['height']:.8f}")
            (label_dir / f"{Path(image['file_name']).stem}.txt").write_text(
                '\n'.join(rows) + '\n', encoding='utf-8')
    yaml_path = output / 'varroa_positive30.yaml'
    yaml_path.write_text(
        f"path: {output.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: varroa\n",
        encoding='utf-8')
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-root', required=True)
    parser.add_argument('--dataset-out', required=True)
    parser.add_argument('--yolo-root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--project', required=True)
    args = parser.parse_args()

    yolo_root = Path(args.yolo_root).resolve()
    sys.path.insert(0, str(yolo_root / 'models_related' / 'ultralytics'))
    from ultralytics import YOLO

    data = export_yolo(Path(args.coco_root), Path(args.dataset_out))
    model = YOLO(args.model)
    model.load('yolov8n.pt', smart_transfer=True)
    model.train(
        data=str(data), epochs=10, imgsz=640, batch=4, workers=2,
        device=0, seed=42, optimizer='SGD', lr0=0.001, lrf=1.0,
        momentum=0.9, weight_decay=0.0001, warmup_epochs=0.0,
        mosaic=0.0, close_mosaic=0, patience=0,
        project=str(Path(args.project).resolve()), name='yolo_dgfe_iou',
        exist_ok=True, plots=False)


if __name__ == '__main__':
    main()
