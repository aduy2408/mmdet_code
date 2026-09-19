#!/usr/bin/env python3
"""Prepare LEVIR-Ship for MMDetection v2's flat COCO image loader.

The project annotations store only image basenames while the source archive
stores tiles in nested folders. This creates a reproducible symlink farm and
copies the selected COCO annotations without changing their contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('../mmdetection/LevirShipData'))
    parser.add_argument('--annotations', type=Path, default=Path('../mmdetection/mmdetection/data/phdetr_levir_ship/annotations'))
    parser.add_argument('--output', type=Path, default=Path('data/levir_ship'))
    args = parser.parse_args()

    source = args.source.resolve()
    annotations = args.annotations.resolve()
    output = args.output.resolve()
    image_index = {}
    for path in source.rglob('*.png'):
        if path.name in image_index:
            other = image_index[path.name]
            if (path.stat().st_size != other.stat().st_size or
                    hashlib.sha256(path.read_bytes()).digest() !=
                    hashlib.sha256(other.read_bytes()).digest()):
                raise SystemExit(f'conflicting LEVIR image basename: {path.name}')
            continue
        image_index[path.name] = path

    (output / 'images').mkdir(parents=True, exist_ok=True)
    (output / 'annotations').mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        annotation_path = annotations / f'{split}.json'
        data = json.loads(annotation_path.read_text())
        for image in data['images']:
            name = image['file_name']
            if name not in image_index:
                raise SystemExit(f'missing LEVIR image: {name}')
            link = output / 'images' / name
            if not link.exists():
                link.symlink_to(image_index[name])
        shutil.copy2(annotation_path, output / 'annotations' / annotation_path.name)
    print(f'Prepared {len(image_index)} LEVIR image symlinks at {output}')


if __name__ == '__main__':
    main()
