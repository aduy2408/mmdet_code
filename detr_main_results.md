# DETR main evaluation results

Verified evaluation report for the pinned checkpoints evaluated on the live Marimo server on **October 6, 2026**.

## Completion status

- Evaluation-only workflow: **24/24 validation/test runs passed**.
- Models: Deformable DETR R50 and RT-DETR R18.
- Datasets: Varroa, TinyPerson, and LEVIR-Ship.
- Training seeds represented by the checkpoints: `42` and `43`.
- Each model/dataset/seed combination has both validation and test results.
- No evaluation process remained active after completion.

## Metric definition

Metrics use the MMDetection COCO bbox evaluator and the `bbox_mAP_copypaste` order:

1. `mAP50-95`
2. `AP50`
3. `AP75`
4. `AP-small`
5. `AP-medium`
6. `AP-large`

Values are decimals, not percentages. For example, `0.2245` means `22.45%`.

## Per-run validation and test results

| Model | Dataset | Seed | Split | mAP50-95 | AP50 | AP75 | AP-small | AP-medium | AP-large |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| Deformable DETR R50 | Varroa | 42 | Val | 0.2220 | 0.7210 | 0.0500 | 0.2280 | 0.2310 | -1.0000 |
| Deformable DETR R50 | Varroa | 42 | Test | 0.2210 | 0.7270 | 0.0500 | 0.2080 | 0.2430 | -1.0000 |
| Deformable DETR R50 | Varroa | 43 | Val | 0.2410 | 0.7930 | 0.0440 | 0.2280 | 0.2680 | -1.0000 |
| Deformable DETR R50 | Varroa | 43 | Test | 0.2490 | 0.7500 | 0.0670 | 0.2250 | 0.2840 | -1.0000 |
| Deformable DETR R50 | TinyPerson | 42 | Val | 0.0690 | 0.2400 | 0.0170 | 0.0520 | 0.2260 | 0.3010 |
| Deformable DETR R50 | TinyPerson | 42 | Test | 0.0540 | 0.1970 | 0.0120 | 0.0430 | 0.1530 | 0.0620 |
| Deformable DETR R50 | TinyPerson | 43 | Val | 0.0720 | 0.2550 | 0.0200 | 0.0550 | 0.2120 | 0.3880 |
| Deformable DETR R50 | TinyPerson | 43 | Test | 0.0600 | 0.2180 | 0.0130 | 0.0500 | 0.1540 | 0.0660 |
| Deformable DETR R50 | LEVIR-Ship | 42 | Val | 0.1800 | 0.6020 | 0.0270 | 0.1820 | 0.1160 | -1.0000 |
| Deformable DETR R50 | LEVIR-Ship | 42 | Test | 0.1780 | 0.5850 | 0.0330 | 0.1770 | 0.2610 | -1.0000 |
| Deformable DETR R50 | LEVIR-Ship | 43 | Val | 0.1690 | 0.6190 | 0.0260 | 0.1690 | 0.1870 | -1.0000 |
| Deformable DETR R50 | LEVIR-Ship | 43 | Test | 0.1570 | 0.5580 | 0.0170 | 0.1550 | 0.2630 | -1.0000 |
| RT-DETR R18 | Varroa | 42 | Val | 0.2372 | 0.8313 | 0.0393 | 0.2476 | 0.2362 | -1.0000 |
| RT-DETR R18 | Varroa | 42 | Test | 0.2245 | 0.7616 | 0.0501 | 0.2174 | 0.2377 | -1.0000 |
| RT-DETR R18 | Varroa | 43 | Val | 0.2296 | 0.8350 | 0.0243 | 0.2326 | 0.2312 | -1.0000 |
| RT-DETR R18 | Varroa | 43 | Test | 0.2177 | 0.7793 | 0.0368 | 0.2002 | 0.2450 | -1.0000 |
| RT-DETR R18 | TinyPerson | 42 | Val | 0.2003 | 0.5409 | 0.1070 | 0.1751 | 0.3477 | 0.5665 |
| RT-DETR R18 | TinyPerson | 42 | Test | 0.1658 | 0.4641 | 0.0822 | 0.1562 | 0.2397 | 0.1279 |
| RT-DETR R18 | TinyPerson | 43 | Val | 0.2027 | 0.5512 | 0.1069 | 0.1814 | 0.3481 | 0.5605 |
| RT-DETR R18 | TinyPerson | 43 | Test | 0.1643 | 0.4628 | 0.0824 | 0.1553 | 0.2391 | 0.1196 |
| RT-DETR R18 | LEVIR-Ship | 42 | Val | 0.2815 | 0.7599 | 0.1031 | 0.2848 | 0.2006 | -1.0000 |
| RT-DETR R18 | LEVIR-Ship | 42 | Test | 0.2421 | 0.6939 | 0.0917 | 0.2431 | 0.2850 | -1.0000 |
| RT-DETR R18 | LEVIR-Ship | 43 | Val | 0.2802 | 0.7734 | 0.1083 | 0.2818 | 0.2297 | -1.0000 |
| RT-DETR R18 | LEVIR-Ship | 43 | Test | 0.2459 | 0.6773 | 0.0813 | 0.2450 | 0.3350 | -1.0000 |

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
- Local server artifact directory: `/marimo/detr_eval_20261006/final_metrics/`
- Hugging Face artifact repository: `duyle2408/detr_eval_20261006`
- Hugging Face remote prefix: `evaluation/`
- Upload verification: **32 expected files present, no missing files**

## Interpretation and limitations

- RT-DETR R18 is stronger than Deformable DETR R50 on TinyPerson and LEVIR-Ship in the verified test results.
- On Varroa, both models are similar in mAP50-95, while RT-DETR has higher AP50 for both seeds.
- `AP-large = -1.0000` for Varroa and LEVIR-Ship follows the COCO evaluator convention when that area bucket is unavailable. It is not a negative measured AP.
- These are COCO bbox metrics from the evaluation checkpoint artifacts. They should not be pooled across datasets or treated as a cross-dataset leaderboard because dataset protocols and object-size distributions differ.
- The report records the six standard COCO metrics available from the evaluator. Dataset-specific extra metrics are not inferred when they were not emitted by the evaluation path.
