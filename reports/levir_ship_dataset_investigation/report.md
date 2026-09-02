# LEVIR-Ship Dataset Investigation

Generated deterministically from the full dataset union; split membership is
intentionally excluded from every analysis.

## Executive summary

| Metric | Value |
|---|---:|
| Images | 3,896 |
| YOLO objects | 3,219 |
| Scenes | 114 |
| Positive images | 1,973 (50.64%) |
| Empty images | 1,923 (49.36%) |
| Objects per image | 0.8262 |
| Objects per positive image | 1.6315 |
| Annotation mismatches | 0 |
| Invalid bbox geometries | 0 |
| Exact/near duplicate rows | 175 |
| Images with >50% exact-black pixels | 88 |

## Dataset integrity

- PNG without TXT: **0**
- TXT without PNG: **0**
- PNG absent from COCO: **0**
- Corrupt/unreadable PNG: **0**
- Filename parse failures: **0**
- TXT↔COCO mismatch records: **0**
- Black inventory entries: **88**; measured >50% exact
  black: **88**; symmetric difference:
  **0**

## Annotation geometry

| Metric | Value |
|---|---:|
| BBox width p01 / p50 / p99 | 8.00 / 20.00 / 41.00 px |
| BBox height p01 / p50 / p99 | 9.00 / 19.00 / 36.06 px |
| BBox area p01 / p50 / p99 | 84.19 / 378.00 / 1178.00 px² |
| Aspect ratio p50 / p95 / max | 1.248 / 1.863 / 3.571 |
| COCO small / medium / large | small: 3,145 / medium: 74 |
| Boxes touching image edge | 165 |
| Images with pairwise IoU ≥0.5 | 0 |

## Pixel and visual quality

| Metric | Value |
|---|---:|
| Brightness mean p01 / p50 / p99 | 16.31 / 74.11 / 193.39 |
| Contrast p01 / p50 / p99 | 0.40 / 5.38 / 76.42 |
| Entropy p01 / p50 / p99 | 0.569 / 3.504 / 7.087 bits |
| Sharpness p01 / p50 / p99 | 1.29 / 9.02 / 197.40 |
| Mean foreground/background brightness | 75.22 / 75.40 |

## Sensor and scene coverage

|                 |   images |   objects |   empty_images |
|:----------------|---------:|----------:|---------------:|
| ('GF1', 'WFV1') |      235 |       306 |             97 |
| ('GF1', 'WFV2') |      366 |       387 |            137 |
| ('GF1', 'WFV3') |      799 |       784 |            320 |
| ('GF1', 'WFV4') |      871 |       524 |            466 |
| ('GF6', 'WFV')  |     1625 |      1218 |            903 |

The filename longitude/latitude values describe the nominal scene and are not
treated as exact tile geolocation. See `scene_statistics.csv` for every scene,
sensor, acquisition date, geographic label, tile range and object density.

## Bias, correlations, and anomalies

- Spearman correlation, object count vs exact-black ratio:
  **0.0207**
- Spearman correlation, object count vs brightness:
  **-0.0114**
- Spearman correlation, object count vs sharpness:
  **-0.0649**
- Robust image-feature anomaly records: **1,166**
- Scene image-count p50/p95/max:
  **14 /
  133 /
  204**
- Scene object-density p05/p50/p95:
  **0.250 /
  0.391 /
  1.310**

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
