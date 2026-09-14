# TinyPerson dataset protocol

## Dataset identity

TinyPerson is **not** a normal full-image COCO detection run. The baseline
protocol trains and evaluates on native-scale sliding-window tiles, then merges
tile detections back into source-image coordinates.

- Task: one-class bounding-box detection.
- Class: `person`.
- Format: TinyPerson corner-window COCO annotations.
- Tile size: `640 x 512` (`sw640`, `sh512`).
- Split seed: `42`.
- Split rule: deterministic 85/15 split by source image, so overlapping tiles
  from one source image cannot leak between train and validation.

## Required files

The exact archive and annotation layout may differ between mounts, but the
canonical prepared files are:

```text
mmdetection/data/tinyperson_baseline_seed42/
├── train_corner.json
└── val_corner.json
```

The corner JSON records must contain, for each tile image, a `corner` field such
as `[x1, y1, x2, y2]`, plus tile-relative bounding boxes. The official test
uses the corner-window annotation from the TinyPerson distribution, not the
full-image annotation alone.

## Mandatory loader and preprocessing

Use the custom project components:

- dataset type: `TinyPersonDataset`
- image transform: `LoadTinyPersonImageFromFile`
- load the source image named by `file_name`
- crop the `[x1, y1, x2, y2]` corner window
- pad the cropped tile to a stride-compatible shape when necessary
- apply the detector's tile pipeline after cropping

Do **not** use plain `CocoDataset` with the full source image. Do **not** resize
the original 3840x2160 source image directly to 640x640. That destroys the
small-object scale and produces an invalid TinyPerson benchmark.

Training and validation annotations are tile-level. The baseline training
pipeline may use its configured tile augmentations, but the source-image crop
must happen first through `LoadTinyPersonImageFromFile`.

## Evaluation

Evaluation has two stages:

1. Run MMDetection on corner tiles.
2. Translate each tile detection by its corner offset, merge detections for the
   source image, and apply class-aware NMS at IoU `0.5`.

Then evaluate against the merged full-image ground truth. Ignore, uncertain,
and logo regions are represented using COCO crowd matching. Report the
TinyPerson-specific metrics from `evaluate_tinyperson_metrics.py`, including
`map_50_95`, `ap50`, `ap75`, and `ap50_tiny1/tiny2/tiny3`.

## Required config checks

Before launch, verify all of the following in the resolved config:

```text
dataset_type = TinyPersonDataset
custom_imports includes projects.tinyperson_baselines
train/val annotations are *_corner.json
loader is LoadTinyPersonImageFromFile
val/test evaluation includes tile-to-source coordinate merging
split_seed = 42
```

A config containing `CocoDataset` plus a direct full-image `Resize` is a
configuration error for TinyPerson and must be rejected before training.

## Common mistakes

- Using the official merged full-image JSON directly with plain COCO loading.
- Resizing the entire source image before cropping tiles.
- Evaluating tile coordinates against full-image ground truth without offset
  translation and NMS merging.
- Splitting individual tiles instead of splitting by source image.
- Treating a zero AP from the wrong full-image pipeline as a model result.
