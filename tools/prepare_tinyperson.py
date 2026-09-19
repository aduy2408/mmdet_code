#!/usr/bin/env python3
"""Prepare the official TinyPerson erased-uncertain corner split for SET."""

from __future__ import annotations

import argparse
import json
import random
import tarfile
from pathlib import Path


TRAIN_ANN = (
    "erase_with_uncertain_dataset/annotations/corner/task/"
    "tiny_set_train_sw640_sh512_all.json"
)


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, default=Path('../TinyPerson/tiny_set'))
    parser.add_argument('--output', type=Path, default=Path('data/tinyperson_seed42'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    archive = dataset_root / 'erase_with_uncertain_dataset/train.tar.gz'
    image_root = output / 'images/erase_with_uncertain_dataset'
    if not (image_root / 'train').is_dir():
        if not archive.is_file():
            raise SystemExit(f'missing TinyPerson archive: {archive}')
        image_root.mkdir(parents=True, exist_ok=True)
        safe_extract(archive, image_root)

    source = json.loads((dataset_root / TRAIN_ANN).read_text(encoding='utf-8'))
    sources = sorted(source['old_images'], key=lambda item: item['file_name'])
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    val_count = max(1, round(len(sources) * args.val_ratio))
    val_names = {item['file_name'] for item in sources[:val_count]}

    def subset(use_val: bool) -> dict:
        images = [
            item for item in source['images']
            if (item['file_name'] in val_names) == use_val
        ]
        image_ids = {item['id'] for item in images}
        return {
            key: value for key, value in source.items()
            if key not in {'images', 'annotations', 'old_images'}
        } | {
            'images': images,
            'annotations': [
                item for item in source['annotations']
                if item['image_id'] in image_ids
            ],
            'old_images': [
                item for item in source['old_images']
                if (item['file_name'] in val_names) == use_val
            ],
        }

    (output / 'annotations').mkdir(parents=True, exist_ok=True)
    for name, data in (('train_corner.json', subset(False)),
                       ('val_corner.json', subset(True))):
        (output / 'annotations' / name).write_text(
            json.dumps(data), encoding='utf-8')
    print(
        f"Prepared official TinyPerson corner split: "
        f"{len(sources) - val_count} train sources, {val_count} val sources; "
        f"images at {image_root / 'train'}"
    )


if __name__ == '__main__':
    main()
