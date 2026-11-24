# Submission code map

Branch: `submission`. This repository remains the MMDetection base. The curated entrypoints below reference the existing source files; the full MMDetection base and required project directories remain available.

Datasets: Varroa, LEVIR-Ship, TinyPerson. Split seed: `42`. Dataset roots, checkpoints, `work_dirs/`, caches, and uploaded artifacts are runtime inputs and are not submission source.

| Area | Canonical source |
|---|---|
| MMDetection baselines | `train_all_mmdet.py`, `train_all_levir_baseline.py`, `train_all_tinyperson_baseline.py`, `run_two_server_baselines.py` |
| Baseline evaluation | `evaluate_coco_metrics.py`, `evaluate_coco_area_metrics.py`, `evaluate_missing_baseline_metrics.py` |
| FCOS-SET (maintained MMDetection 3.x port) | `run_set_fcos_mmdet3_queue.py`, `projects/set/`, `mmdetection/configs/set/` |
| STW-matched protocol | `run_set_stw_protocol_queue.py` (FCOS-SET protocol, not an STW architecture) |
| TPH-YOLO / STW-YOLO | External YOLO backends; see `yolo_related/submission/` and `other_repo/` |
| SR-TOD | `train_srtod_mmdet.py`, `setup_srtod_datasets.py`, `srtod_setup/`, `SR-TOD/` |
| Marimo connection | `utils/marimo_ops.py`, `docs/local_github_marimo_workflow.md` |

Use the launchers under `train/` and `evaluate/`. Run MMDetection jobs with `/marimo/mmdet-venv/bin/python` through the repository Marimo workflow; do not launch long-running training locally. The exact model/config, split, seed, artifact, and upload target must be recorded in each run manifest.

SR-TOD exposes `srtod_faster` and `srtod_cascade`; it is not an FCOS implementation. FCOS-SET is the separate comparison model.
