#!/usr/bin/env python3
"""Finish the repaired sweep, then retrain its test winner on exact counts."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

import train_all_highres_repair as repair


SOURCE_SHA = 'c31a1dc39c3f2472347a51ce92243da8635f4649'
HF_REPO_ID = 'duyle2408/levir-highres-sweep-repair'
SPLIT_TARGETS = {
    'train': (2320, 2002),
    'val': (788, 665),
    'test': (788, 552),
}


def write_state(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True), encoding='utf-8')


def restore_1024(work_root: Path, token: str) -> Path:
    snapshot = Path(snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type='dataset',
        token=token,
        allow_patterns=f'{SOURCE_SHA}/1024/*/training/**',
    ))
    destination = work_root / SOURCE_SHA / '1024'
    for variant in repair.DEFAULT_VARIANTS:
        source = snapshot / SOURCE_SHA / '1024' / variant / 'training'
        target = destination / variant
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, target, dirs_exist_ok=True)
    return destination


def run_1376(args: argparse.Namespace) -> None:
    subprocess.run([
        sys.executable,
        str(args.repo_root / 'train_all_highres_repair.py'),
        '--resolutions', '1376',
        '--skip-data-prepare',
        '--data-root', str(args.data_root),
        '--dataset-out', str(args.dataset_out),
        '--work-dir', str(args.work_root),
        '--num-workers', str(args.num_workers),
        '--hf-repo-id', args.hf_repo_id,
    ], cwd=args.repo_root, check=True)


def upload_folder(
        api: HfApi, folder: Path, path_in_repo: str,
        repo_id: str) -> None:
    api.upload_folder(
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type='dataset',
    )


def test_1024(
        root: Path, api: HfApi, repo_id: str) -> list[dict[str, Any]]:
    rows = []
    for variant in repair.DEFAULT_VARIANTS:
        run_dir = root / variant
        config = run_dir / 'patched_config.py'
        for checkpoint in repair.checkpoints(config):
            test_dir, duration = repair.test_config(config, checkpoint)
            metrics = repair.latest_validation_metrics(test_dir)
            upload_folder(
                api,
                test_dir,
                f'{SOURCE_SHA}/1024/{variant}/{checkpoint.stem}',
                repo_id,
            )
            rows.append({
                'resolution': 1024,
                'variant': variant,
                'checkpoint': checkpoint.name,
                'duration_seconds': duration,
                **metrics,
            })
    return rows


def test_rows(root: Path, sha: str, resolution: int) -> list[dict[str, Any]]:
    rows = []
    for variant in repair.DEFAULT_VARIANTS:
        variant_root = root / sha / str(resolution) / variant
        for summary in sorted(
                variant_root.glob('checkpoint_tests/*/run_summary.json')):
            payload = json.loads(summary.read_text())
            rows.append({
                'resolution': resolution,
                'variant': variant,
                'checkpoint': payload['checkpoint'],
                'duration_seconds': payload['duration_seconds'],
                **payload['metrics'],
            })
    return rows


def combine_coco(annotation_dir: Path) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for split in ('train', 'val', 'test'):
        payload = json.loads((annotation_dir / f'{split}.json').read_text())
        annotations: dict[int, list[dict[str, Any]]] = {}
        for annotation in payload['annotations']:
            annotations.setdefault(annotation['image_id'], []).append(annotation)
        for image in payload['images']:
            key = image['file_name']
            if key in records:
                raise ValueError(f'Duplicate image file: {key}')
            records[key] = {
                'image': image,
                'annotations': annotations.get(image['id'], []),
            }
    return list(records.values())


def select_exact(
        records: list[dict[str, Any]],
        image_count: int,
        target_count: int,
        rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives = [
        index for index, record in enumerate(records)
        if record['annotations']
    ]
    zeros = [
        index for index, record in enumerate(records)
        if not record['annotations']
    ]
    rng.shuffle(positives)
    rng.shuffle(zeros)
    paths: dict[int, tuple[int, ...]] = {0: ()}
    for index in positives:
        weight = len(records[index]['annotations'])
        for total in sorted(tuple(paths), reverse=True):
            new_total = total + weight
            if new_total > target_count:
                continue
            candidate = paths[total] + (index,)
            existing = paths.get(new_total)
            if (len(candidate) <= image_count
                    and (existing is None or len(candidate) < len(existing))):
                paths[new_total] = candidate
    selected_positive = paths.get(target_count)
    if selected_positive is None:
        raise RuntimeError(f'Cannot make exact target count {target_count}')
    zero_count = image_count - len(selected_positive)
    if zero_count < 0 or zero_count > len(zeros):
        raise RuntimeError(
            f'Cannot fill {image_count} images with {target_count} targets')
    selected_indices = set(selected_positive) | set(zeros[:zero_count])
    selected = [
        record for index, record in enumerate(records)
        if index in selected_indices
    ]
    remaining = [
        record for index, record in enumerate(records)
        if index not in selected_indices
    ]
    return selected, remaining


def write_coco_split(
        records: list[dict[str, Any]], output: Path) -> None:
    images = []
    annotations = []
    annotation_id = 1
    for image_id, record in enumerate(records, 1):
        image = dict(record['image'])
        image['id'] = image_id
        images.append(image)
        for source in record['annotations']:
            annotation = dict(source)
            annotation['id'] = annotation_id
            annotation['image_id'] = image_id
            annotations.append(annotation)
            annotation_id += 1
    output.write_text(json.dumps({
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 1, 'name': 'ship', 'supercategory': 'ship'}],
    }), encoding='utf-8')


def make_exact_split(
        source: Path, destination: Path, seed: int) -> dict[str, Any]:
    records = combine_coco(source / 'annotations')
    rng = random.Random(seed)
    test, remaining = select_exact(
        records, *SPLIT_TARGETS['test'], rng)
    val, train = select_exact(
        remaining, *SPLIT_TARGETS['val'], rng)
    splits = {'train': train, 'val': val, 'test': test}
    annotation_dir = destination / 'annotations'
    annotation_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': seed, 'scene_leakage_allowed': True, 'splits': {}}
    for split, selected in splits.items():
        output = annotation_dir / f'{split}.json'
        write_coco_split(selected, output)
        image_count = len(selected)
        target_count = sum(len(item['annotations']) for item in selected)
        expected = SPLIT_TARGETS[split]
        if (image_count, target_count) != expected:
            raise AssertionError(
                f'{split}: {(image_count, target_count)} != {expected}')
        manifest['splits'][split] = {
            'images': image_count,
            'targets': target_count,
        }
    (destination / 'split_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    return manifest


def retrain_winner(
        args: argparse.Namespace,
        winner: dict[str, Any],
        sha: str,
        api: HfApi,
) -> None:
    variant = winner['variant']
    resolution = winner['resolution']
    leaky_root = args.work_root / sha / 'leaky_exact_counts'
    config_args = argparse.Namespace(
        batch_size=8,
        num_workers=args.num_workers,
        seed=args.seed,
        work_dir=str(leaky_root),
    )
    config = repair.write_config(
        variant,
        resolution,
        config_args,
        args.leaky_dataset_out,
        args.data_root / 'All Images',
        sha,
    )
    duration = repair.train_config(config, amp=False)
    repair.write_summary(
        config.parent,
        phase='leaky_exact_counts_training',
        source_test_winner=winner,
        duration_seconds=duration,
        peak_vram_mb=repair.peak_vram_mb(config.parent),
        return_code=0,
    )
    prefix = f'{sha}/leaky_exact_counts/{resolution}/{variant}'
    upload_folder(api, config.parent, f'{prefix}/training', args.hf_repo_id)
    for checkpoint in repair.checkpoints(config):
        test_dir, _ = repair.test_config(config, checkpoint)
        upload_folder(
            api, test_dir, f'{prefix}/{checkpoint.stem}', args.hf_repo_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument(
        '--data-root', type=Path, default=Path('/marimo/LevirShipData'))
    parser.add_argument(
        '--dataset-out',
        type=Path,
        default=Path('mmdetection/data/levir_ship_coco'))
    parser.add_argument(
        '--leaky-dataset-out',
        type=Path,
        default=Path('mmdetection/data/levir_ship_coco_exact_counts'))
    parser.add_argument(
        '--work-root',
        type=Path,
        default=Path('mmdetection/work_dirs/highres_sweep_repair'))
    parser.add_argument('--hf-repo-id', default=HF_REPO_ID)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.dataset_out = args.dataset_out.resolve()
    args.leaky_dataset_out = args.leaky_dataset_out.resolve()
    args.work_root = args.work_root.resolve()
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN is required')
    api = HfApi(token=token)
    sha = repair.git_sha()
    state = args.work_root / sha / 'continuation_state.json'

    write_state(state, status='restoring_1024', git_sha=sha)
    restored = restore_1024(args.work_root, token)
    write_state(state, status='running_1376', git_sha=sha)
    run_1376(args)

    write_state(state, status='testing_1024', git_sha=sha)
    rows = test_1024(restored, api, args.hf_repo_id)
    rows.extend(test_rows(args.work_root, sha, 1376))
    if len(rows) != 16:
        raise RuntimeError(f'Expected 16 test rows, got {len(rows)}')
    rows.sort(
        key=lambda row: (
            row['coco/bbox_mAP'],
            row.get('coco/bbox_mAP_75', -1),
        ),
        reverse=True,
    )
    summary = args.work_root / sha / 'all_checkpoint_test_summary.json'
    summary.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    api.upload_file(
        path_or_fileobj=str(summary),
        path_in_repo=f'{sha}/all_checkpoint_test_summary.json',
        repo_id=args.hf_repo_id,
        repo_type='dataset',
    )

    winner = rows[0]
    write_state(
        state, status='creating_exact_count_split',
        git_sha=sha, winner=winner)
    split_manifest = make_exact_split(
        args.dataset_out, args.leaky_dataset_out, args.seed)
    api.upload_file(
        path_or_fileobj=str(
            args.leaky_dataset_out / 'split_manifest.json'),
        path_in_repo=f'{sha}/leaky_exact_counts/split_manifest.json',
        repo_id=args.hf_repo_id,
        repo_type='dataset',
    )

    write_state(
        state, status='retraining_test_winner',
        git_sha=sha, winner=winner, split=split_manifest)
    retrain_winner(args, winner, sha, api)
    write_state(
        state, status='completed',
        git_sha=sha, winner=winner, split=split_manifest)


if __name__ == '__main__':
    main()
