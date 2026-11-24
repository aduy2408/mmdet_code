# Evaluation

- Baseline and SOTA COCO detections: `coco.sh` or `sota_coco.sh`; pass the model's ground-truth and detection JSON explicitly.
- Area and missing-result audits remain `evaluate_coco_area_metrics.py` and `evaluate_missing_baseline_metrics.py`.
- TinyPerson merged/corner-window evaluation remains `evaluate_tinyperson_detr_merged.py` and `evaluate_tinyperson_metrics.py`.

The evaluator writes split-qualified results only when the caller supplies the corresponding split artifact. Do not relabel validation output as test output.
