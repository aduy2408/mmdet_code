# MMDetection HF Results

## mAP / AP50

| model | mAP mean | mAP std | AP50 mean | AP50 std | GFLOPs | Params(M) | FLOPs input | seeds |
|---|---:|---:|---:|---:|---:|---:|---|---|
| cascade_rcnn | 0.3340 | 0.0121 | 0.9000 | 0.0104 | 87.898 | 69.152 | 640x384 | 42 43 44 |
| faster_rcnn | 0.3383 | 0.0099 | 0.8967 | 0.0156 | 60.099 | 41.348 | 640x384 | 42 43 44 |
| fcos | 0.2553 | 0.0741 | 0.7773 | 0.1753 | 47.148 | 32.113 | 640x384 | 42 43 44 |

## Full Metrics

| model | variant | metric | n | mean | std | seeds | status |
|---|---|---|---:|---:|---:|---|---|
| cascade_rcnn | base | coco/bbox_mAP | 3 | 0.3340 | 0.0121 | 42 43 44 | ok |
| cascade_rcnn | base | coco/bbox_mAP_50 | 3 | 0.9000 | 0.0104 | 42 43 44 | ok |
| cascade_rcnn | base | coco/bbox_mAP_75 | 3 | 0.1210 | 0.0156 | 42 43 44 | ok |
| cascade_rcnn | base | coco/bbox_mAP_s | 3 | 0.3340 | 0.0087 | 42 43 44 | ok |
| cascade_rcnn | base | coco/bbox_mAP_m | 3 | 0.3407 | 0.0162 | 42 43 44 | ok |
| cascade_rcnn | base | coco/bbox_mAP_l | 3 | -1.0000 | 0.0000 | 42 43 44 | ok |
| faster_rcnn | base | coco/bbox_mAP | 3 | 0.3383 | 0.0099 | 42 43 44 | ok |
| faster_rcnn | base | coco/bbox_mAP_50 | 3 | 0.8967 | 0.0156 | 42 43 44 | ok |
| faster_rcnn | base | coco/bbox_mAP_75 | 3 | 0.1157 | 0.0124 | 42 43 44 | ok |
| faster_rcnn | base | coco/bbox_mAP_s | 3 | 0.3333 | 0.0199 | 42 43 44 | ok |
| faster_rcnn | base | coco/bbox_mAP_m | 3 | 0.3497 | 0.0095 | 42 43 44 | ok |
| faster_rcnn | base | coco/bbox_mAP_l | 3 | -1.0000 | 0.0000 | 42 43 44 | ok |
| fcos | base | coco/bbox_mAP | 3 | 0.2553 | 0.0741 | 42 43 44 | ok |
| fcos | base | coco/bbox_mAP_50 | 3 | 0.7773 | 0.1753 | 42 43 44 | ok |
| fcos | base | coco/bbox_mAP_75 | 3 | 0.0690 | 0.0340 | 42 43 44 | ok |
| fcos | base | coco/bbox_mAP_s | 3 | 0.2680 | 0.0207 | 42 43 44 | ok |
| fcos | base | coco/bbox_mAP_m | 3 | 0.2777 | 0.0959 | 42 43 44 | ok |
| fcos | base | coco/bbox_mAP_l | 3 | -1.0000 | 0.0000 | 42 43 44 | ok |

## Notes

- GFLOPs/Params were computed with `mmdetection/tools/analysis_tools/get_flops.py`, `--num-images 1`, local `.venv-mmdet`, and `val_dataloader.dataset.pipeline.1.scale=(640,640)` to match the current `train_all_mmdet.py` default; unsupported ops warnings from MMEngine still apply.
- duplicate seed=42 model=cascade_rcnn variant=base chose hf_results/downloads/duyle2408__varroa_mmdet_runs/cascade_rcnn/base skipped hf_results/downloads/duyle2408__varroa_mmdet_runs_seed43/varroa_4models_seed42/cascade_rcnn/base
