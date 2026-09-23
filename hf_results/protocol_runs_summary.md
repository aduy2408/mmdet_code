# Verified protocol runs summary

This report separates MMDetection runs from the historical STW-YOLO runs and does not infer missing metrics.

## Verified metrics

| family | dataset | protocol | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 | source |
|---|---|---|---:|---:|---:|---:|---|
| FCOS + SR-TOD | TinyPerson | corner-window, Mosaic | 0.247 | 0.079 | 0.216 | 0.072 | `duyle2408/srtod-tinyperson-protocol-runs/fcos/` |
| SR-TOD Faster R-CNN | Varroa | standard COCO | 0.930 | 0.334 | 0.908 | 0.334 | `duyle2408/srtod-varroa-protocol-runs/srtod_faster/` |
| STW-YOLO | LEVIR-Ship | independent test evaluation | unknown | unknown | 0.7981 | 0.2949 | `duyle2408/stw-yolo-runs/runs/levirship/seed_42/evaluation_metrics.json` |
| STW-YOLO | Varroa | independent test evaluation | unknown | unknown | 0.9166 | 0.3534 | `duyle2408/stw-yolo-runs/runs/varroa/seed_42/evaluation_metrics.json` |
| STW-YOLO | TinyPerson | independent corner-tile test evaluation | unknown | unknown | 0.5685 | 0.2149 | `duyle2408/stw-yolo-runs/runs/tinyperson/seed_42/evaluation_metrics.json` |

## Verified run status without copied metrics

| family | dataset | evidence | status |
|---|---|---|---|
| FCOS-SET | LEVIR-Ship | corrected checkpoint, final results, and HF prefix under `duyle2408/set-fcos-stw-protocol-runs/levirship/no_mosaic/patience15/seed42/fcos_set` | completed artifacts verified, split metrics not yet transcribed here |
| FCOS-SET | TinyPerson | corrected checkpoint, final results, and HF prefix under `duyle2408/set-fcos-stw-protocol-runs/tinyperson/mosaic/patience15/seed42/fcos_set` | completed artifacts verified, split metrics not yet transcribed here |
| FCOS + SR-TOD | LEVIR-Ship | test JSON recorded at `test/AP50=0.611`, `test/mAP50-95=0.188` and HF upload was reported verified | test metrics available, validation metrics not yet transcribed here |

## Protocol notes

- STW-YOLO uses `duyle2408/stw-yolo-runs`, model `p2_rp5_yolo12s.yaml`, pretrained `yolo12s.pt`, 100 epochs, batch 16, workers 8, seed/split seed 42, and `optimizer=auto`.
- TinyPerson STW-YOLO's independent evaluation is corner-tile based. Its merged source-image artifact reports `test_merged/AP50=0.5891` and `test_merged/mAP50-75=0.3603`, which is not interchangeable with mAP50-95 and is therefore kept separate from the table above.
- Unknown values are intentionally left blank until a split-specific artifact is directly checked.
