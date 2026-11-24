#!/usr/bin/env python3
"""Create a deterministic, split-agnostic investigation of LEVIR-Ship."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy.optimize import linear_sum_assignment

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


FILENAME_RE = re.compile(
    r'^(?P<satellite>GF\d+)_(?P<sensor>[^_]+)_'
    r'E(?P<longitude>-?\d+(?:\.\d+)?)_'
    r'N(?P<latitude>-?\d+(?:\.\d+)?)_(?P<date>\d{8})_'
    r'(?P<product>[^_]+)_(?P<tile_x>\d+)_(?P<tile_y>\d+)\.png$')
QUANTILES = (0, .01, .05, .25, .5, .75, .95, .99, 1)
BLACK_THRESHOLDS = (0, 5, 10, 20, 40)


def parse_filename(filename: str) -> dict:
    """Parse satellite and tile metadata encoded in a LEVIR filename."""
    match = FILENAME_RE.match(Path(filename).name)
    if not match:
        return {'filename_parse_error': True}
    values = match.groupdict()
    scene_id = Path(filename).stem.rsplit('_', 2)[0]
    return {
        'filename_parse_error': False,
        'satellite': values['satellite'],
        'sensor': values['sensor'],
        'longitude': float(values['longitude']),
        'latitude': float(values['latitude']),
        'capture_date': datetime.strptime(values['date'], '%Y%m%d').date().isoformat(),
        'product_id': values['product'],
        'scene_id': scene_id,
        'tile_x': int(values['tile_x']),
        'tile_y': int(values['tile_y']),
    }


def read_yolo(path: Path, width: int = 512, height: int = 512) -> list[dict]:
    """Read a YOLO file and return both normalized and pixel xywh boxes."""
    boxes = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            boxes.append({
                'line_number': line_number,
                'parse_error': f'expected 5 fields, got {len(fields)}',
                'raw': line,
            })
            continue
        try:
            class_id = int(fields[0])
            cx, cy, bw, bh = map(float, fields[1:])
        except ValueError as error:
            boxes.append({
                'line_number': line_number,
                'parse_error': str(error),
                'raw': line,
            })
            continue
        boxes.append({
            'line_number': line_number,
            'parse_error': '',
            'class_id': class_id,
            'cx_norm': cx,
            'cy_norm': cy,
            'width_norm': bw,
            'height_norm': bh,
            'x': (cx - bw / 2) * width,
            'y': (cy - bh / 2) * height,
            'width': bw * width,
            'height': bh * height,
        })
    return boxes


def xywh_iou(first: Iterable[float], second: Iterable[float]) -> float:
    """Return IoU for two xywh boxes."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.


def perceptual_hash(rgb: np.ndarray) -> int:
    """Compute a deterministic 64-bit DCT perceptual hash."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    low = cv2.dct(np.float32(small))[:8, :8]
    threshold = float(np.median(low.ravel()[1:]))
    bits = (low >= threshold).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def image_metrics(rgb: np.ndarray) -> dict:
    """Calculate deterministic pixel-level quality and colour statistics."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    probability = hist[hist > 0] / gray.size
    entropy = float(-(probability * np.log2(probability)).sum())
    result = {
        'brightness_mean': float(gray.mean()),
        'brightness_std': float(gray.std()),
        'brightness_p01': float(np.quantile(gray, .01)),
        'brightness_p50': float(np.quantile(gray, .5)),
        'brightness_p99': float(np.quantile(gray, .99)),
        'saturation_mean': float(hsv[..., 1].mean()),
        'entropy': entropy,
        'sharpness_laplacian_var': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'white_ratio': float(np.all(rgb >= 250, axis=2).mean()),
    }
    for index, channel in enumerate('rgb'):
        pixels = rgb[..., index]
        result[f'{channel}_mean'] = float(pixels.mean())
        result[f'{channel}_std'] = float(pixels.std())
        result[f'{channel}_p01'] = float(np.quantile(pixels, .01))
        result[f'{channel}_p50'] = float(np.quantile(pixels, .5))
        result[f'{channel}_p99'] = float(np.quantile(pixels, .99))
    maximum = rgb.max(axis=2)
    for threshold in BLACK_THRESHOLDS:
        result[f'black_ratio_le_{threshold}'] = float((maximum <= threshold).mean())
    return result


def robust_outlier_flags(values: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Flag values using a median absolute-deviation robust z-score."""
    numeric = pd.to_numeric(values, errors='coerce')
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=values.index)
    score = .6745 * (numeric - median).abs() / mad
    return score > threshold


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_coco(coco_dir: Path) -> tuple[dict[str, dict], dict[str, list[dict]], list[dict]]:
    images: dict[str, dict] = {}
    annotations: dict[str, list[dict]] = defaultdict(list)
    issues = []
    for path in sorted(coco_dir.glob('*.json')):
        document = json.loads(path.read_text())
        id_to_name = {}
        for image in document.get('images', []):
            filename = Path(image['file_name']).name
            if filename in images:
                issues.append({
                    'filename': filename,
                    'issue_type': 'duplicate_coco_image',
                    'severity': 'data_error',
                    'details': path.name,
                })
            else:
                images[filename] = image
            id_to_name[image['id']] = filename
        for annotation in document.get('annotations', []):
            filename = id_to_name.get(annotation['image_id'])
            if filename is None:
                issues.append({
                    'filename': '',
                    'issue_type': 'orphan_coco_annotation',
                    'severity': 'data_error',
                    'details': f'{path.name}:{annotation.get("id")}',
                })
            else:
                annotations[filename].append(annotation)
    return images, annotations, issues


def _match_annotations(filename: str, yolo: list[dict], coco: list[dict],
                       tolerance: float) -> list[dict]:
    valid_yolo = [box for box in yolo if not box.get('parse_error')]
    rows = []
    for box in yolo:
        if box.get('parse_error'):
            rows.append({
                'filename': filename,
                'issue_type': 'invalid_yolo_line',
                'severity': 'data_error',
                'details': f'line {box["line_number"]}: {box["parse_error"]}',
            })
    if not valid_yolo and not coco:
        return rows
    if not valid_yolo:
        return rows + [{
            'filename': filename,
            'issue_type': 'coco_only_bbox',
            'severity': 'data_error',
            'details': json.dumps(annotation['bbox']),
        } for annotation in coco]
    if not coco:
        return rows + [{
            'filename': filename,
            'issue_type': 'yolo_only_bbox',
            'severity': 'data_error',
            'details': json.dumps([box[key] for key in ('x', 'y', 'width', 'height')]),
        } for box in valid_yolo]
    costs = np.empty((len(valid_yolo), len(coco)))
    for row, box in enumerate(valid_yolo):
        first = np.array([box[key] for key in ('x', 'y', 'width', 'height')])
        for column, annotation in enumerate(coco):
            costs[row, column] = np.max(np.abs(first - np.asarray(annotation['bbox'])))
    yolo_indices, coco_indices = linear_sum_assignment(costs)
    matched_yolo, matched_coco = set(yolo_indices), set(coco_indices)
    for yolo_index, coco_index in zip(yolo_indices, coco_indices):
        box, annotation = valid_yolo[yolo_index], coco[coco_index]
        differences = [
            box[key] - value for key, value in
            zip(('x', 'y', 'width', 'height'), annotation['bbox'])
        ]
        if box['class_id'] + 1 != annotation.get('category_id'):
            rows.append({
                'filename': filename,
                'issue_type': 'class_mismatch',
                'severity': 'data_error',
                'details': f'YOLO={box["class_id"]}, COCO={annotation.get("category_id")}',
            })
        if max(map(abs, differences)) > tolerance:
            rows.append({
                'filename': filename,
                'issue_type': 'bbox_coordinate_mismatch',
                'severity': 'data_error',
                'details': json.dumps({'delta_xywh': differences}),
            })
    for index, box in enumerate(valid_yolo):
        if index not in matched_yolo:
            rows.append({
                'filename': filename,
                'issue_type': 'yolo_only_bbox',
                'severity': 'data_error',
                'details': f'line {box["line_number"]}',
            })
    for index, annotation in enumerate(coco):
        if index not in matched_coco:
            rows.append({
                'filename': filename,
                'issue_type': 'coco_only_bbox',
                'severity': 'data_error',
                'details': str(annotation.get('id')),
            })
    return rows


def _pairwise_box_metrics(boxes: list[dict]) -> tuple[float, float]:
    max_iou, min_distance = 0., math.nan
    for first_index, first in enumerate(boxes):
        if first.get('parse_error'):
            continue
        first_xywh = [first[key] for key in ('x', 'y', 'width', 'height')]
        for second in boxes[first_index + 1:]:
            if second.get('parse_error'):
                continue
            second_xywh = [second[key] for key in ('x', 'y', 'width', 'height')]
            max_iou = max(max_iou, xywh_iou(first_xywh, second_xywh))
            distance = math.hypot(
                first['cx_norm'] - second['cx_norm'],
                first['cy_norm'] - second['cy_norm'])
            min_distance = distance if math.isnan(min_distance) else min(min_distance, distance)
    return max_iou, min_distance


def _foreground_background(gray: np.ndarray, boxes: list[dict]) -> tuple[float, float]:
    mask = np.zeros(gray.shape, dtype=bool)
    for box in boxes:
        if box.get('parse_error'):
            continue
        x1 = max(0, int(math.floor(box['x'])))
        y1 = max(0, int(math.floor(box['y'])))
        x2 = min(gray.shape[1], int(math.ceil(box['x'] + box['width'])))
        y2 = min(gray.shape[0], int(math.ceil(box['y'] + box['height'])))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    foreground = float(gray[mask].mean()) if mask.any() else math.nan
    background = float(gray[~mask].mean()) if (~mask).any() else math.nan
    return foreground, background


def _make_plots(images: pd.DataFrame, objects: pd.DataFrame, scenes: pd.DataFrame,
                output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output / name, dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    bins = np.arange(0, min(20, int(images['object_count'].max())) + 2) - .5
    plt.hist(images['object_count'], bins=bins)
    plt.yscale('log')
    plt.xlabel('Objects per image')
    plt.ylabel('Images (log scale)')
    save('objects_per_image.png')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(objects['area'], bins=50)
    axes[0].set_xscale('log')
    axes[0].set_title('BBox area')
    axes[1].hist(objects['aspect_ratio'], bins=50)
    axes[1].set_title('BBox aspect ratio')
    axes[2].scatter(objects['width'], objects['height'], s=4, alpha=.25)
    axes[2].set_xlabel('Width')
    axes[2].set_ylabel('Height')
    axes[2].set_title('BBox dimensions')
    save('bbox_geometry.png')

    plt.figure(figsize=(6, 5))
    heat, _, _ = np.histogram2d(
        objects['cy_norm'], objects['cx_norm'], bins=20, range=((0, 1), (0, 1)))
    plt.imshow(heat, cmap='magma', origin='upper', extent=(0, 1, 1, 0))
    plt.colorbar(label='BBox centres')
    plt.xlabel('Normalized x')
    plt.ylabel('Normalized y')
    save('bbox_center_heatmap.png')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, column, title in zip(
            axes.flat,
            ('brightness_mean', 'brightness_std', 'entropy',
             'sharpness_laplacian_var'),
            ('Brightness', 'Contrast', 'Entropy', 'Sharpness')):
        axis.hist(images[column].dropna(), bins=50)
        axis.set_title(title)
    save('image_quality.png')

    plt.figure(figsize=(8, 5))
    for threshold in BLACK_THRESHOLDS:
        column = f'black_ratio_le_{threshold}'
        plt.hist(images[column], bins=50, histtype='step', label=f'≤{threshold}')
    plt.xlabel('Black-pixel ratio')
    plt.ylabel('Images')
    plt.legend()
    save('black_pixel_ratios.png')

    sensor = images.groupby('sensor', dropna=False).agg(
        images=('filename', 'size'), objects=('object_count', 'sum'))
    sensor.plot.bar(figsize=(8, 5), secondary_y='objects')
    save('sensor_coverage.png')

    plt.figure(figsize=(9, 5))
    top = scenes.dropna(subset=['scene_id']).nlargest(
        30, 'image_count').sort_values('image_count')
    plt.barh(top['scene_id'].str[-22:], top['image_count'])
    plt.xlabel('Images')
    plt.ylabel('Scene suffix')
    save('largest_scenes.png')

    numeric = images[[
        'object_count', 'brightness_mean', 'brightness_std', 'entropy',
        'sharpness_laplacian_var', 'black_ratio_le_0', 'saturation_mean'
    ]].corr(method='spearman')
    plt.figure(figsize=(8, 7))
    plt.imshow(numeric, vmin=-1, vmax=1, cmap='coolwarm')
    plt.xticks(range(len(numeric)), numeric.columns, rotation=60, ha='right')
    plt.yticks(range(len(numeric)), numeric.columns)
    plt.colorbar(label='Spearman correlation')
    save('feature_correlations.png')


def _annotated_thumbnail(row: pd.Series, boxes: list[dict], size: int = 240) -> Image.Image:
    image = Image.open(row['image_path']).convert('RGB')
    draw = ImageDraw.Draw(image)
    for box in boxes:
        if box.get('parse_error'):
            continue
        draw.rectangle(
            (box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']),
            outline=(255, 40, 40), width=2)
    image.thumbnail((size, size))
    canvas = Image.new('RGB', (size, size + 34), 'white')
    canvas.paste(image, ((size - image.width) // 2, 0))
    ImageDraw.Draw(canvas).text(
        (4, size + 3), f'{row["filename"][:31]}\nobjects={row["object_count"]}',
        fill='black')
    return canvas


def _make_gallery(name: str, frame: pd.DataFrame, box_map: dict[str, list[dict]],
                  output: Path, limit: int) -> None:
    if frame.empty:
        return
    selected = frame.head(limit)
    columns = min(4, len(selected))
    rows = math.ceil(len(selected) / columns)
    tiles = [
        _annotated_thumbnail(row, box_map.get(row['filename'], []))
        for _, row in selected.iterrows()
    ]
    gallery = Image.new('RGB', (columns * 240, rows * 274), (235, 235, 235))
    for index, tile in enumerate(tiles):
        gallery.paste(tile, ((index % columns) * 240, (index // columns) * 274))
    gallery.save(output / f'{name}.jpg', quality=90)


def _make_galleries(images: pd.DataFrame, objects: pd.DataFrame,
                    box_map: dict[str, list[dict]], duplicate_names: set[str],
                    output: Path, limit: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    selections = {
        'blackest': images.sort_values('black_ratio_le_10', ascending=False),
        'brightest': images.sort_values('brightness_mean', ascending=False),
        'darkest': images.sort_values('brightness_mean'),
        'blurriest': images.sort_values('sharpness_laplacian_var'),
        'lowest_entropy': images.sort_values('entropy'),
        'most_objects': images.sort_values('object_count', ascending=False),
        'highest_bbox_overlap': images.sort_values('max_pairwise_iou', ascending=False),
        'duplicates': images[images['filename'].isin(duplicate_names)],
    }
    object_orders = {
        'smallest_bbox': objects.sort_values('area'),
        'largest_bbox': objects.sort_values('area', ascending=False),
        'longest_bbox': objects.sort_values('aspect_ratio', ascending=False),
        'closest_to_edge': objects.sort_values('edge_distance'),
    }
    for name, object_frame in object_orders.items():
        selections[name] = images.set_index('filename').loc[
            object_frame['filename'].drop_duplicates()].reset_index()
    for name, frame in selections.items():
        _make_gallery(name, frame, box_map, output, limit)


def _describe(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors='coerce').dropna()
    return {
        f'p{int(quantile * 100):02d}': float(clean.quantile(quantile))
        for quantile in QUANTILES
    } if len(clean) else {}


def _markdown_table(rows: list[tuple[str, object]]) -> str:
    body = ['| Metric | Value |', '|---|---:|']
    body.extend(f'| {key} | {value} |' for key, value in rows)
    return '\n'.join(body)


def _write_report(output: Path, images: pd.DataFrame, objects: pd.DataFrame,
                  scenes: pd.DataFrame, mismatches: pd.DataFrame,
                  anomalies: pd.DataFrame, duplicates: pd.DataFrame,
                  black_inventory: dict[str, float]) -> None:
    empty = int((images['object_count'] == 0).sum())
    positive = len(images) - empty
    invalid = int((objects['valid_geometry'] == False).sum())  # noqa: E712
    sensors = images.groupby(['satellite', 'sensor'], dropna=False).agg(
        images=('filename', 'size'), objects=('object_count', 'sum'),
        empty_images=('object_count', lambda values: int((values == 0).sum())))
    black_flagged = int((images['black_ratio_le_0'] > .5).sum())
    inventory_names = set(black_inventory)
    measured_names = set(images.loc[images['black_ratio_le_0'] > .5, 'filename'])
    report = f"""# LEVIR-Ship Dataset Investigation

Generated deterministically from the full dataset union; split membership is
intentionally excluded from every analysis.

## Executive summary

{_markdown_table([
    ('Images', f'{len(images):,}'),
    ('YOLO objects', f'{len(objects):,}'),
    ('Scenes', f'{images["scene_id"].nunique():,}'),
    ('Positive images', f'{positive:,} ({positive / len(images):.2%})'),
    ('Empty images', f'{empty:,} ({empty / len(images):.2%})'),
    ('Objects per image', f'{len(objects) / len(images):.4f}'),
    ('Objects per positive image', f'{len(objects) / positive:.4f}'),
    ('Annotation mismatches', f'{len(mismatches):,}'),
    ('Invalid bbox geometries', f'{invalid:,}'),
    ('Exact/near duplicate rows', f'{len(duplicates):,}'),
    ('Images with >50% exact-black pixels', f'{black_flagged:,}'),
])}

## Dataset integrity

- PNG without TXT: **{int(images["missing_txt"].sum()):,}**
- TXT without PNG: **{int(images["missing_image"].sum()):,}**
- PNG absent from COCO: **{int(images["missing_coco"].sum()):,}**
- Corrupt/unreadable PNG: **{int(images["image_error"].notna().sum()):,}**
- Filename parse failures: **{int(images["filename_parse_error"].sum()):,}**
- TXT↔COCO mismatch records: **{len(mismatches):,}**
- Black inventory entries: **{len(inventory_names):,}**; measured >50% exact
  black: **{len(measured_names):,}**; symmetric difference:
  **{len(inventory_names ^ measured_names):,}**

## Annotation geometry

{_markdown_table([
    ('BBox width p01 / p50 / p99',
     f'{objects["width"].quantile(.01):.2f} / {objects["width"].median():.2f} / {objects["width"].quantile(.99):.2f} px'),
    ('BBox height p01 / p50 / p99',
     f'{objects["height"].quantile(.01):.2f} / {objects["height"].median():.2f} / {objects["height"].quantile(.99):.2f} px'),
    ('BBox area p01 / p50 / p99',
     f'{objects["area"].quantile(.01):.2f} / {objects["area"].median():.2f} / {objects["area"].quantile(.99):.2f} px²'),
    ('Aspect ratio p50 / p95 / max',
     f'{objects["aspect_ratio"].median():.3f} / {objects["aspect_ratio"].quantile(.95):.3f} / {objects["aspect_ratio"].max():.3f}'),
    ('COCO small / medium / large',
     ' / '.join(f'{name}: {count:,}' for name, count in objects["coco_size"].value_counts().items())),
    ('Boxes touching image edge', f'{int(objects["touches_edge"].sum()):,}'),
    ('Images with pairwise IoU ≥0.5', f'{int((images["max_pairwise_iou"] >= .5).sum()):,}'),
])}

## Pixel and visual quality

{_markdown_table([
    ('Brightness mean p01 / p50 / p99',
     f'{images["brightness_mean"].quantile(.01):.2f} / {images["brightness_mean"].median():.2f} / {images["brightness_mean"].quantile(.99):.2f}'),
    ('Contrast p01 / p50 / p99',
     f'{images["brightness_std"].quantile(.01):.2f} / {images["brightness_std"].median():.2f} / {images["brightness_std"].quantile(.99):.2f}'),
    ('Entropy p01 / p50 / p99',
     f'{images["entropy"].quantile(.01):.3f} / {images["entropy"].median():.3f} / {images["entropy"].quantile(.99):.3f} bits'),
    ('Sharpness p01 / p50 / p99',
     f'{images["sharpness_laplacian_var"].quantile(.01):.2f} / {images["sharpness_laplacian_var"].median():.2f} / {images["sharpness_laplacian_var"].quantile(.99):.2f}'),
    ('Mean foreground/background brightness',
     f'{images["foreground_brightness"].mean():.2f} / {images["background_brightness"].mean():.2f}'),
])}

## Sensor and scene coverage

{sensors.to_markdown()}

The filename longitude/latitude values describe the nominal scene and are not
treated as exact tile geolocation. See `scene_statistics.csv` for every scene,
sensor, acquisition date, geographic label, tile range and object density.

## Bias, correlations, and anomalies

- Spearman correlation, object count vs exact-black ratio:
  **{images["object_count"].corr(images["black_ratio_le_0"], method="spearman"):.4f}**
- Spearman correlation, object count vs brightness:
  **{images["object_count"].corr(images["brightness_mean"], method="spearman"):.4f}**
- Spearman correlation, object count vs sharpness:
  **{images["object_count"].corr(images["sharpness_laplacian_var"], method="spearman"):.4f}**
- Robust image-feature anomaly records: **{len(anomalies):,}**
- Scene image-count p50/p95/max:
  **{scenes["image_count"].median():.0f} /
  {scenes["image_count"].quantile(.95):.0f} /
  {scenes["image_count"].max():.0f}**
- Scene object-density p05/p50/p95:
  **{scenes["objects_per_image"].quantile(.05):.3f} /
  {scenes["objects_per_image"].median():.3f} /
  {scenes["objects_per_image"].quantile(.95):.3f}**

`anomalies.csv` distinguishes data errors, suspicious-but-valid observations,
and distribution characteristics. No image or annotation was modified.

## Reproducibility and detailed outputs

- `image_statistics.csv`: one row per filename with pixels, metadata and label density.
- `object_statistics.csv`: one row per YOLO object with geometry and edge flags.
- `scene_statistics.csv`: one row per nominal source scene.
- `annotation_mismatches.csv`: all inventory and TXT↔COCO discrepancies.
- `anomalies.csv`: robust outliers and quality flags.
- `duplicate_groups.csv`: exact hashes and perceptual near-duplicate pairs.
- `plots/` and `galleries/`: deterministic aggregate figures and visual audit sheets.
- `summary.json` and `manifest.json`: machine-readable findings and provenance.

Expected missing values are limited to object-dependent image metrics:
`foreground_brightness` is empty for images without boxes, and
`min_pairwise_center_distance_norm` is empty for images with fewer than two
boxes. These values are undefined by construction, not failed measurements.
"""
    (output / 'report.md').write_text(report)


def investigate(data_root: Path, coco_dir: Path, output: Path,
                mismatch_tolerance: float = .51, near_hash_distance: int = 4,
                gallery_size: int = 12) -> dict:
    """Run the complete investigation and write all artifacts."""
    image_dir = data_root / 'All Images'
    annotation_dir = data_root / 'All Annotations'
    black_csv = data_root / 'Black Images Over 50 Percent' / 'black_images_inventory.csv'
    image_paths = {path.name: path for path in sorted(image_dir.glob('*.png'))}
    txt_paths = {path.stem + '.png': path for path in sorted(annotation_dir.glob('*.txt'))}
    coco_images, coco_annotations, mismatches = _load_coco(coco_dir)
    filenames = sorted(set(image_paths) | set(txt_paths) | set(coco_images))
    box_map: dict[str, list[dict]] = {}
    image_rows, object_rows = [], []

    for filename in filenames:
        path = image_paths.get(filename)
        txt_path = txt_paths.get(filename)
        coco_image = coco_images.get(filename)
        expected_width = int(coco_image.get('width', 512)) if coco_image else 512
        expected_height = int(coco_image.get('height', 512)) if coco_image else 512
        boxes = read_yolo(txt_path, expected_width, expected_height) if txt_path else []
        box_map[filename] = boxes
        valid_boxes = [box for box in boxes if not box.get('parse_error')]
        max_iou, min_distance = _pairwise_box_metrics(valid_boxes)
        row = {
            'filename': filename,
            'image_path': str(path.resolve()) if path else '',
            'annotation_path': str(txt_path.resolve()) if txt_path else '',
            'missing_image': path is None,
            'missing_txt': txt_path is None,
            'missing_coco': coco_image is None,
            'object_count': len(valid_boxes),
            'max_pairwise_iou': max_iou,
            'min_pairwise_center_distance_norm': min_distance,
            **parse_filename(filename),
        }
        if path is None:
            row['image_error'] = 'missing'
        else:
            try:
                with Image.open(path) as pil:
                    pil.load()
                    rgb = np.asarray(pil.convert('RGB'))
                    row.update({
                        'width_px': pil.width,
                        'height_px': pil.height,
                        'image_mode': pil.mode,
                        'bit_depth': np.asarray(pil).dtype.itemsize * 8,
                        'file_size_bytes': path.stat().st_size,
                        'sha256': _sha256(path),
                        'perceptual_hash': f'{perceptual_hash(rgb):016x}',
                        **image_metrics(rgb),
                    })
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    foreground, background = _foreground_background(gray, valid_boxes)
                    row['foreground_brightness'] = foreground
                    row['background_brightness'] = background
                    row['image_error'] = None
                    if (coco_image and
                            (pil.width != expected_width or pil.height != expected_height)):
                        mismatches.append({
                            'filename': filename,
                            'issue_type': 'image_dimension_mismatch',
                            'severity': 'data_error',
                            'details': f'PNG={pil.size}, COCO={(expected_width, expected_height)}',
                        })
            except Exception as error:  # audit must record and continue
                row['image_error'] = repr(error)
                mismatches.append({
                    'filename': filename,
                    'issue_type': 'unreadable_image',
                    'severity': 'data_error',
                    'details': repr(error),
                })
        if path is None or txt_path is None or coco_image is None:
            for missing, issue in (
                (path is None, 'missing_png'),
                (txt_path is None, 'missing_txt'),
                (coco_image is None, 'missing_coco_image')):
                if missing:
                    mismatches.append({
                        'filename': filename,
                        'issue_type': issue,
                        'severity': 'data_error',
                        'details': '',
                    })
        mismatches.extend(_match_annotations(
            filename, boxes, coco_annotations.get(filename, []), mismatch_tolerance))
        image_rows.append(row)

        for object_index, box in enumerate(valid_boxes):
            x, y, width, height = (box[key] for key in ('x', 'y', 'width', 'height'))
            x2, y2 = x + width, y + height
            valid_geometry = (
                box['class_id'] == 0 and width > 0 and height > 0 and
                x >= 0 and y >= 0 and x2 <= expected_width and y2 <= expected_height)
            area = width * height
            edge_distance = min(x, y, expected_width - x2, expected_height - y2)
            object_rows.append({
                'filename': filename,
                'object_index': object_index,
                'line_number': box['line_number'],
                'class_id': box['class_id'],
                **{key: box[key] for key in (
                    'cx_norm', 'cy_norm', 'width_norm', 'height_norm',
                    'x', 'y', 'width', 'height')},
                'x2': x2,
                'y2': y2,
                'area': area,
                'diagonal': math.hypot(width, height),
                'aspect_ratio': max(width / height, height / width)
                if width > 0 and height > 0 else math.inf,
                'edge_distance': edge_distance,
                'touches_edge': edge_distance <= .51,
                'valid_geometry': valid_geometry,
                'coco_size': 'small' if area < 32**2 else
                ('medium' if area < 96**2 else 'large'),
                'scene_id': row.get('scene_id'),
                'sensor': row.get('sensor'),
                'capture_date': row.get('capture_date'),
            })
            if not valid_geometry:
                mismatches.append({
                    'filename': filename,
                    'issue_type': 'invalid_yolo_geometry',
                    'severity': 'data_error',
                    'details': f'line {box["line_number"]}',
                })

    images = pd.DataFrame(image_rows).sort_values('filename').reset_index(drop=True)
    objects = pd.DataFrame(object_rows).sort_values(
        ['filename', 'object_index']).reset_index(drop=True)
    mismatch_frame = pd.DataFrame(
        mismatches, columns=('filename', 'issue_type', 'severity', 'details'))
    mismatch_frame = mismatch_frame.sort_values(
        ['filename', 'issue_type']).reset_index(drop=True)

    scenes = images.groupby('scene_id', dropna=False).agg(
        satellite=('satellite', 'first'),
        sensor=('sensor', 'first'),
        longitude=('longitude', 'first'),
        latitude=('latitude', 'first'),
        capture_date=('capture_date', 'first'),
        product_id=('product_id', 'first'),
        image_count=('filename', 'size'),
        object_count=('object_count', 'sum'),
        positive_images=('object_count', lambda values: int((values > 0).sum())),
        tile_x_min=('tile_x', 'min'),
        tile_x_max=('tile_x', 'max'),
        tile_y_min=('tile_y', 'min'),
        tile_y_max=('tile_y', 'max'),
        black_ratio_mean=('black_ratio_le_0', 'mean'),
        brightness_mean=('brightness_mean', 'mean'),
        sharpness_mean=('sharpness_laplacian_var', 'mean'),
    ).reset_index()
    scenes['objects_per_image'] = scenes['object_count'] / scenes['image_count']
    scenes['positive_image_ratio'] = scenes['positive_images'] / scenes['image_count']

    duplicate_rows = []
    exact_groups = images.groupby('sha256', dropna=True)['filename'].apply(list)
    duplicate_names = set()
    group_id = 0
    for digest, names in exact_groups.items():
        if len(names) < 2:
            continue
        group_id += 1
        duplicate_names.update(names)
        for name in names:
            duplicate_rows.append({
                'group_id': f'exact-{group_id}',
                'duplicate_type': 'exact',
                'filename': name,
                'other_filename': '',
                'distance': 0,
                'hash': digest,
            })
    hashes = [
        (row.filename, int(row.perceptual_hash, 16))
        for row in images[['filename', 'perceptual_hash']].dropna().itertuples()
    ]
    near_group = 0
    exact_name_pairs = {
        frozenset(names) for names in exact_groups if len(names) == 2
    }
    for index, (first_name, first_hash) in enumerate(hashes):
        for second_name, second_hash in hashes[index + 1:]:
            distance = (first_hash ^ second_hash).bit_count()
            if distance <= near_hash_distance:
                if frozenset((first_name, second_name)) in exact_name_pairs:
                    continue
                near_group += 1
                duplicate_names.update((first_name, second_name))
                duplicate_rows.append({
                    'group_id': f'near-{near_group}',
                    'duplicate_type': 'perceptual_near',
                    'filename': first_name,
                    'other_filename': second_name,
                    'distance': distance,
                    'hash': '',
                })
    duplicates = pd.DataFrame(duplicate_rows, columns=(
        'group_id', 'duplicate_type', 'filename', 'other_filename', 'distance', 'hash'))

    anomaly_rows = list(mismatches)
    features = {
        'very_dark': ('brightness_mean', 'low'),
        'very_bright': ('brightness_mean', 'high'),
        'unusually_low_contrast': ('brightness_std', 'low'),
        'unusually_high_contrast': ('brightness_std', 'high'),
        'unusually_low_entropy': ('entropy', 'low'),
        'unusually_blurry': ('sharpness_laplacian_var', 'low'),
        'unusually_many_objects': ('object_count', 'high'),
        'unusually_black': ('black_ratio_le_0', 'high'),
    }
    for issue, (column, direction) in features.items():
        flags = robust_outlier_flags(images[column])
        median = images[column].median()
        if direction == 'low':
            flags &= images[column] < median
        else:
            flags &= images[column] > median
        for row in images.loc[flags, ['filename', column]].itertuples(index=False):
            anomaly_rows.append({
                'filename': row[0],
                'issue_type': issue,
                'severity': 'suspicious_but_valid',
                'details': f'{column}={row[1]:.6g}',
            })
    for row in scenes.itertuples():
        if row.image_count >= scenes['image_count'].quantile(.95):
            anomaly_rows.append({
                'filename': '',
                'issue_type': 'large_scene',
                'severity': 'distribution_characteristic',
                'details': f'{row.scene_id}: images={row.image_count}',
            })
        if row.objects_per_image >= scenes['objects_per_image'].quantile(.95):
            anomaly_rows.append({
                'filename': '',
                'issue_type': 'object_dense_scene',
                'severity': 'distribution_characteristic',
                'details': f'{row.scene_id}: objects/image={row.objects_per_image:.4f}',
            })
    for row in images.loc[
            images['black_ratio_le_0'] > .5,
            ['filename', 'black_ratio_le_0']].itertuples(index=False):
        anomaly_rows.append({
            'filename': row.filename,
            'issue_type': 'black_pixels_over_50_percent',
            'severity': 'suspicious_but_valid',
            'details': f'black_ratio={row.black_ratio_le_0:.6g}',
        })
    edge_counts = objects.loc[objects['touches_edge']].groupby('filename').size()
    for filename, count in edge_counts.items():
        anomaly_rows.append({
            'filename': filename,
            'issue_type': 'bbox_touches_image_edge',
            'severity': 'suspicious_but_valid',
            'details': f'boxes={count}',
        })
    for row in duplicates.itertuples(index=False):
        anomaly_rows.append({
            'filename': row.filename,
            'issue_type': row.duplicate_type,
            'severity': 'suspicious_but_valid',
            'details': (
                f'other={row.other_filename}; perceptual_distance={row.distance}'
                if row.other_filename else f'group={row.group_id}'),
        })
    anomalies = pd.DataFrame(anomaly_rows, columns=(
        'filename', 'issue_type', 'severity', 'details')).sort_values(
            ['severity', 'issue_type', 'filename']).reset_index(drop=True)

    output.mkdir(parents=True, exist_ok=True)
    images.to_csv(output / 'image_statistics.csv', index=False)
    objects.to_csv(output / 'object_statistics.csv', index=False)
    scenes.to_csv(output / 'scene_statistics.csv', index=False)
    mismatch_frame.to_csv(output / 'annotation_mismatches.csv', index=False)
    anomalies.to_csv(output / 'anomalies.csv', index=False)
    duplicates.to_csv(output / 'duplicate_groups.csv', index=False)
    _make_plots(images, objects, scenes, output / 'plots')
    _make_galleries(
        images, objects, box_map, duplicate_names, output / 'galleries', gallery_size)

    black_inventory = {}
    if black_csv.exists():
        with black_csv.open(newline='') as stream:
            for row in csv.DictReader(stream):
                black_inventory[row['image']] = float(row['black_percent']) / 100
    _write_report(
        output, images, objects, scenes, mismatch_frame, anomalies, duplicates,
        black_inventory)

    summary = {
        'images': len(images),
        'yolo_objects': len(objects),
        'coco_objects': sum(map(len, coco_annotations.values())),
        'scenes': int(images['scene_id'].nunique()),
        'positive_images': int((images['object_count'] > 0).sum()),
        'empty_images': int((images['object_count'] == 0).sum()),
        'annotation_mismatches': len(mismatch_frame),
        'anomalies': len(anomalies),
        'exact_duplicate_images': int((duplicates['duplicate_type'] == 'exact').sum()),
        'near_duplicate_pairs': int(
            (duplicates['duplicate_type'] == 'perceptual_near').sum()),
        'image_quantiles': {
            column: _describe(images[column])
            for column in ('object_count', 'brightness_mean', 'brightness_std',
                           'entropy', 'sharpness_laplacian_var', 'black_ratio_le_0')
        },
        'object_quantiles': {
            column: _describe(objects[column])
            for column in ('width', 'height', 'area', 'aspect_ratio', 'edge_distance')
        },
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    manifest = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'data_root': str(data_root.resolve()),
        'coco_dir': str(coco_dir.resolve()),
        'parameters': {
            'mismatch_tolerance': mismatch_tolerance,
            'near_hash_distance': near_hash_distance,
            'gallery_size': gallery_size,
        },
        'input_counts': {
            'png': len(image_paths),
            'txt': len(txt_paths),
            'coco_json': len(list(coco_dir.glob('*.json'))),
        },
        'input_digest': hashlib.sha256(
            ''.join(
                f'{row.filename}:{row.sha256}'
                for row in images[['filename', 'sha256']].fillna('').itertuples()
            ).encode()).hexdigest(),
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--coco-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mismatch-tolerance', type=float, default=.51)
    parser.add_argument('--near-hash-distance', type=int, default=4)
    parser.add_argument('--gallery-size', type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = investigate(
        args.data_root, args.coco_dir, args.output_dir,
        mismatch_tolerance=args.mismatch_tolerance,
        near_hash_distance=args.near_hash_distance,
        gallery_size=args.gallery_size)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
