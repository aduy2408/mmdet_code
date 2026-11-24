# Submission code map

Branch: `submission`. This folder is the self-contained MMDetection submission payload; the base framework and required runners are physically copied here.

## Included payload

| Area | Submission source |
|---|---|
| MMDetection base | `submission/mmdetection/` |
| Baseline training across Varroa/LEVIR-Ship/TinyPerson | `submission/train_all_mmdet.py`, `submission/train_all_levir_baseline.py`, `submission/train_all_tinyperson_baseline.py`, `submission/run_two_server_baselines.py` |
| FCOS-SET and STW protocol | `submission/run_set_fcos_mmdet3_queue.py`, `submission/run_set_stw_protocol_queue.py`, `submission/projects/` |
| SR-TOD | `submission/train_srtod_mmdet.py`, `submission/setup_srtod_datasets.py`, `submission/srtod_setup/`, `submission/SR-TOD/` |
| DETR/RT-DETR | `submission/run_detr_r18_rtdetr_r18_matrix.py`, `submission/run_detr_r18_rtdetr_r18_two_server.py` |
| Baseline/SOTA evaluation | `submission/evaluate_coco_metrics.py`, `submission/evaluate_coco_area_metrics.py`, `submission/evaluate_missing_baseline_metrics.py`, `submission/evaluate_tinyperson_metrics.py` |
| Python entrypoints | `submission/train/*.py`, `submission/evaluate/*.py` |
| Marimo workflow | `submission/utils/`, `submission/docs/`, `submission/marimo/` |
| CI/CD map | `submission/ci_cd/` |

Use `/marimo/mmdet-venv/bin/python` for long-running jobs. Split seed is `42`; training runners use seeds `42,43,44`, 100 epochs, and patience `0` where the protocol supports those arguments. Runtime datasets, checkpoints, `work_dirs/`, caches, and generated reports are excluded.

Run from the submission root so copied runners resolve the copied `mmdetection/` and project files locally.
