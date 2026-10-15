# DETR main evaluation results

Verified evaluation report for the pinned checkpoints evaluated on the live Marimo server on **October 6, 2026**.

## Completion status

- Evaluation-only workflow: **24/24 validation/test runs passed**.
- Models: Deformable DETR R50 and RT-DETR R18.
- Datasets: Varroa, TinyPerson, and LEVIR-Ship.
- Training seeds represented by the checkpoints: `42` and `43`.
- Each model/dataset/seed combination has both validation and test results.
- Augmentation variant from the evaluated configs: Varroa uses `no_mosaic`; TinyPerson and LEVIR-Ship use `mosaic` for both model families and both seeds.
- No evaluation process remained active after completion.

## Metric definition

Metrics use the MMDetection COCO bbox evaluator and the `bbox_mAP_copypaste` order:

1. `mAP50-95`
2. `AP50`
3. `AP75`
4. `AP50-Small`
5. `AP-small`
6. `AP-medium`
7. `AP-large`

Metrics use the MMDetection COCO bbox evaluator. `AP50-Small` is computed separately from the existing prediction artifacts with COCOeval at IoU 0.50 over the small-object area bucket. `AP-small` remains the area AP averaged over IoU 0.50:0.95. The values below are decimals, not percentages.

## Per-run validation and test results

| Model | Dataset | Variant | Seed | Split | mAP50-95 | AP50 | AP75 | AP50-Small | AP-small | AP-medium | AP-large |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Deformable DETR R50 | Varroa | no_mosaic | 42 | Val | 0.2220 | 0.7210 | 0.0500 | 0.7057 | 0.2280 | 0.2310 | -1.0000 |
| Deformable DETR R50 | Varroa | no_mosaic | 42 | Test | 0.2210 | 0.7270 | 0.0500 | 0.6750 | 0.2080 | 0.2430 | -1.0000 |
| Deformable DETR R50 | Varroa | no_mosaic | 43 | Val | 0.2410 | 0.7930 | 0.0440 | 0.7593 | 0.2280 | 0.2680 | -1.0000 |
| Deformable DETR R50 | Varroa | no_mosaic | 43 | Test | 0.2490 | 0.7500 | 0.0670 | 0.7003 | 0.2250 | 0.2840 | -1.0000 |
| Deformable DETR R50 | TinyPerson | mosaic | 42 | Val | 0.0690 | 0.2400 | 0.0170 | 0.1966 | 0.0520 | 0.2260 | 0.3010 |
| Deformable DETR R50 | TinyPerson | mosaic | 42 | Test | 0.0540 | 0.1970 | 0.0120 | 0.1677 | 0.0430 | 0.1530 | 0.0620 |
| Deformable DETR R50 | TinyPerson | mosaic | 43 | Val | 0.0720 | 0.2550 | 0.0200 | 0.2117 | 0.0550 | 0.2120 | 0.3880 |
| Deformable DETR R50 | TinyPerson | mosaic | 43 | Test | 0.0600 | 0.2180 | 0.0130 | 0.1908 | 0.0500 | 0.1540 | 0.0660 |
| Deformable DETR R50 | LEVIR-Ship | mosaic | 42 | Val | 0.1800 | 0.6020 | 0.0270 | 0.6047 | 0.1820 | 0.1160 | -1.0000 |
| Deformable DETR R50 | LEVIR-Ship | mosaic | 42 | Test | 0.1780 | 0.5850 | 0.0330 | 0.5824 | 0.1770 | 0.2610 | -1.0000 |
| Deformable DETR R50 | LEVIR-Ship | mosaic | 43 | Val | 0.1690 | 0.6190 | 0.0260 | 0.6184 | 0.1690 | 0.1870 | -1.0000 |
| Deformable DETR R50 | LEVIR-Ship | mosaic | 43 | Test | 0.1570 | 0.5580 | 0.0170 | 0.5538 | 0.1550 | 0.2630 | -1.0000 |
| RT-DETR R18 | Varroa | no_mosaic | 42 | Val | 0.2372 | 0.8313 | 0.0393 | 0.8278 | 0.2476 | 0.2362 | -1.0000 |
| RT-DETR R18 | Varroa | no_mosaic | 42 | Test | 0.2245 | 0.7616 | 0.0501 | 0.7456 | 0.2174 | 0.2377 | -1.0000 |
| RT-DETR R18 | Varroa | no_mosaic | 43 | Val | 0.2296 | 0.8350 | 0.0243 | 0.8241 | 0.2326 | 0.2312 | -1.0000 |
| RT-DETR R18 | Varroa | no_mosaic | 43 | Test | 0.2177 | 0.7793 | 0.0368 | 0.7330 | 0.2002 | 0.2450 | -1.0000 |
| RT-DETR R18 | TinyPerson | mosaic | 42 | Val | 0.2003 | 0.5409 | 0.1070 | 0.5085 | 0.1751 | 0.3477 | 0.5665 |
| RT-DETR R18 | TinyPerson | mosaic | 42 | Test | 0.1658 | 0.4641 | 0.0822 | 0.4529 | 0.1562 | 0.2397 | 0.1279 |
| RT-DETR R18 | TinyPerson | mosaic | 43 | Val | 0.2027 | 0.5512 | 0.1069 | 0.5201 | 0.1814 | 0.3481 | 0.5605 |
| RT-DETR R18 | TinyPerson | mosaic | 43 | Test | 0.1643 | 0.4628 | 0.0824 | 0.4517 | 0.1553 | 0.2391 | 0.1196 |
| RT-DETR R18 | LEVIR-Ship | mosaic | 42 | Val | 0.2815 | 0.7599 | 0.1031 | 0.7624 | 0.2848 | 0.2006 | -1.0000 |
| RT-DETR R18 | LEVIR-Ship | mosaic | 42 | Test | 0.2421 | 0.6939 | 0.0917 | 0.6931 | 0.2431 | 0.2850 | -1.0000 |
| RT-DETR R18 | LEVIR-Ship | mosaic | 43 | Val | 0.2802 | 0.7734 | 0.1083 | 0.7752 | 0.2818 | 0.2297 | -1.0000 |
| RT-DETR R18 | LEVIR-Ship | mosaic | 43 | Test | 0.2459 | 0.6773 | 0.0813 | 0.6803 | 0.2450 | 0.3350 | -1.0000 |

## Test-set comparison by seed

| Dataset | Model | Seed 42 mAP50-95 / AP50 | Seed 43 mAP50-95 / AP50 |
|---|---|---:|---:|
| Varroa | Deformable DETR R50 | 0.2210 / 0.7270 | 0.2490 / 0.7500 |
| Varroa | RT-DETR R18 | 0.2245 / 0.7616 | 0.2177 / 0.7793 |
| TinyPerson | Deformable DETR R50 | 0.0540 / 0.1970 | 0.0600 / 0.2180 |
| TinyPerson | RT-DETR R18 | 0.1658 / 0.4641 | 0.1643 / 0.4628 |
| LEVIR-Ship | Deformable DETR R50 | 0.1780 / 0.5850 | 0.1570 / 0.5580 |
| LEVIR-Ship | RT-DETR R18 | 0.2421 / 0.6939 | 0.2459 / 0.6773 |

## Protocol and provenance

- Experiment ID: `detr_eval_20261006`
- Runner: MMDetection `tools/test.py`
- Python: `/marimo/mmdet-venv/bin/python`
- Deformable DETR source checkout: `/marimo/mmdet_code`, commit `17449f3b0b471510071ff683f2f59868cb25e432`
- RT-DETR source checkout: `/marimo/rtdetr-mmdet`, commit `66365c1553ffd121ca4ee2be9d091735faf3c182`
- Split seed: `42`
- Evaluation checkpoints: two seeds per model/dataset, selected from the recorded checkpoint manifest on the server
- Dataset roots:
  - Varroa: `/marimo/Varroa`
  - TinyPerson: `/marimo/TinyPerson`
  - LEVIR-Ship: `/marimo/LevirShip/LevirShipData`
- Local server area-metric artifact: `/marimo/detr_eval_20261006/final_metrics/area_metrics_v2.json`
- AP50-Small evaluation: COCOeval recomputation from the existing `predictions.pkl` artifacts; no retraining was performed
- Hugging Face artifact repository: `duyle2408/detr_eval_20261006`
- Hugging Face remote prefix: `evaluation/`
- Upload verification: **32 expected files present, no missing files**

## Interpretation and limitations

- RT-DETR R18 is stronger than Deformable DETR R50 on TinyPerson and LEVIR-Ship in the verified test results.
- On Varroa, both models are similar in mAP50-95, while RT-DETR has higher AP50 for both seeds.
- `AP-large = -1.0000` for Varroa and LEVIR-Ship follows the COCO evaluator convention when that area bucket is unavailable. It is not a negative measured AP.
- These are COCO bbox metrics from the evaluation checkpoint artifacts. They should not be pooled across datasets or treated as a cross-dataset leaderboard because dataset protocols and object-size distributions differ.
- The report records the six standard COCO metrics available from the evaluator. Dataset-specific extra metrics are not inferred when they were not emitted by the evaluation path.

## TinyPerson merged-corner re-evaluation on October 15, 2026

This section is separate from the older native MMDetection evaluation above. It evaluates the archived TinyPerson checkpoints on both validation and test corner windows, then merges corner-window detections back to source-image coordinates before scoring. No retraining was performed.

- Evaluator commit: `59fad320cdfd8294cc27036d0c2b2f111ca2e96e`
- RT-DETR runtime commit: `66365c1553ffd121ca4ee2be9d091735faf3c182`
- Split seed: `42`; checkpoint seeds: `42`, `43`
- NMS IoU: `0.5`
- HF dataset repository: `duyle2408/tinyperson-detr-merged-eval-20261015-runs`
- Remote prefix: `tinyperson_detr_merged_eval/rtdetr_r18/seed{seed}`
- HF verification: all six RT-DETR per-run files plus `summary.json` present, no missing paths.

### RT-DETR R18 per-seed merged metrics

| Seed | Split | mAP50-95 | AP50 | AP50-Small | AP75 | AP-small | AP-medium | AP50-Tiny1 | AP50-Tiny2 | AP50-Tiny3 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | Val merged | 0.241436 | 0.610152 | 0.682263 | 0.147049 | 0.212474 | 0.395240 | 0.377536 | 0.611200 | 0.636824 |
| 43 | Val merged | 0.240350 | 0.608427 | 0.678653 | 0.135165 | 0.212187 | 0.395658 | 0.362757 | 0.622015 | 0.633923 |
| 42 | Test merged | 0.199871 | 0.545406 | 0.623657 | 0.100836 | 0.183680 | 0.275467 | 0.315801 | 0.523867 | 0.591419 |
| 43 | Test merged | 0.197591 | 0.541340 | 0.611770 | 0.099708 | 0.183907 | 0.270444 | 0.318234 | 0.522625 | 0.595387 |

### RT-DETR R18 merged mean ± standard deviation across seeds

| Split | mAP50-95 | AP50 | AP50-Small | AP75 | AP-small | AP-medium | AP50-Tiny1 | AP50-Tiny2 | AP50-Tiny3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Val merged | 0.240893 ± 0.000768 | 0.609289 ± 0.001220 | 0.680458 ± 0.002553 | 0.141107 ± 0.008403 | 0.212331 ± 0.000204 | 0.395449 ± 0.000296 | 0.370147 ± 0.010450 | 0.616608 ± 0.007648 | 0.635374 ± 0.002051 |
| Test merged | 0.198731 ± 0.001612 | 0.543373 ± 0.002875 | 0.617713 ± 0.008406 | 0.100272 ± 0.000798 | 0.183794 ± 0.000161 | 0.272956 ± 0.003552 | 0.317017 ± 0.001721 | 0.523246 ± 0.000879 | 0.593403 ± 0.002806 |

The merged artifacts are independently stored as `evaluation_metrics.json`, `evaluation_manifest.json`, and `evaluation_records.json` for each seed. These merged results must not be mixed with the older native-corner rows above.
