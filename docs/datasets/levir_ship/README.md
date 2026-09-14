# LEVIR-Ship dataset protocol

## Dataset identity

- Task: one-class bounding-box detection.
- Class: `ship`.
- Format: COCO JSON generated from the scene-safe split.
- Canonical image root: `LevirShipData/All Images/`.
- Canonical annotation root: `mmdetection/data/levir_ship_coco_recovered/annotations/`.
- Split seed: `42`.

## Required files

```text
mmdetection/data/levir_ship_coco_recovered/annotations/
├── train.json
├── val.json
└── test.json

LevirShipData/All Images/
└── <all source images referenced by the three JSON files>
```

The split is scene-safe. Keep split seed `42` fixed when comparing training
seeds or model variants. Every image referenced in the JSON files must resolve
under `LevirShipData/All Images/`.

## Preprocessing

The current SR-TOD Faster R-CNN setup uses the normal COCO path:

- `LoadImageFromFile`
- `LoadAnnotations(with_bbox=True)`
- `Resize(scale=(1333, 800), keep_ratio=True)`
- random horizontal flip with probability `0.5` for training
- `PackDetInputs`

LEVIR-Ship does **not** use TinyPerson corner-window cropping or the
`TinyPersonDataset` custom loader.

## Evaluation

Use `CocoMetric(metric='bbox')` with the matching split annotation file. Report
COCO bbox `mAP`, `AP50`, and `AP75`. The TinyPerson-specific `tiny1`, `tiny2`,
`tiny3`, and tile-merge metrics do not apply.

## Common mistakes

- Do not regenerate a random image-level split. Use the scene-safe split and
  keep split seed `42`.
- Do not use `TinyPersonDataset` or translate detections from tile coordinates.
- Do not infer the image root from a checkpoint name.
