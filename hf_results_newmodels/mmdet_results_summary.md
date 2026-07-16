# MMDetection HF Results

## mAP / AP50

| model | variant | n | mAP mean | mAP std | AP50 mean | AP50 std | seeds |
|---|---|---:|---:|---:|---:|---:|---|
| cascade_rcnn | base | 1 | 0.3200 | — | 0.8880 | — | 44 |
| cascade_rcnn | dgfe_api | 3 | 0.3500 | 0.0242 | 0.9073 | 0.0117 | 42 43 44 |
| faster_rcnn | base | 1 | 0.3320 | — | 0.9060 | — | 44 |
| faster_rcnn | dgfe_api | 3 | 0.3530 | 0.0200 | 0.9027 | 0.0095 | 42 43 44 |
| fcos | base | 1 | 0.3000 | — | 0.8830 | — | 44 |
| fcos | dgfe_api | 3 | 0.3160 | 0.0195 | 0.8847 | 0.0180 | 42 43 44 |

## Seed 44: DGFE API vs Base

| model | base mAP | dgfe_api mAP | delta |
|---|---:|---:|---:|
| cascade_rcnn | 0.3200 | 0.3220 | +0.0020 |
| faster_rcnn | 0.3320 | 0.3300 | -0.0020 |
| fcos | 0.3000 | 0.2940 | -0.0060 |

## Full Metrics

| model | variant | metric | n | mean | std | seeds | status |
|---|---|---|---:|---:|---:|---|---|
| cascade_rcnn | base | coco/bbox_mAP | 1 | 0.3200 |  | 44 | ok |
| cascade_rcnn | base | coco/bbox_mAP_50 | 1 | 0.8880 |  | 44 | ok |
| cascade_rcnn | base | coco/bbox_mAP_75 | 1 | 0.1030 |  | 44 | ok |
| cascade_rcnn | base | coco/bbox_mAP_s | 1 | 0.3240 |  | 44 | ok |
| cascade_rcnn | base | coco/bbox_mAP_m | 1 | 0.3220 |  | 44 | ok |
| cascade_rcnn | base | coco/bbox_mAP_l | 1 | -1.0000 |  | 44 | ok |
| cascade_rcnn | dgfe_api | coco/bbox_mAP | 3 | 0.3500 | 0.0242 | 42 43 44 | ok |
| cascade_rcnn | dgfe_api | coco/bbox_mAP_50 | 3 | 0.9073 | 0.0117 | 42 43 44 | ok |
| cascade_rcnn | dgfe_api | coco/bbox_mAP_75 | 3 | 0.1427 | 0.0440 | 42 43 44 | ok |
| cascade_rcnn | dgfe_api | coco/bbox_mAP_s | 3 | 0.3457 | 0.0163 | 42 43 44 | ok |
| cascade_rcnn | dgfe_api | coco/bbox_mAP_m | 3 | 0.3610 | 0.0327 | 42 43 44 | ok |
| cascade_rcnn | dgfe_api | coco/bbox_mAP_l | 3 | -1.0000 | 0.0000 | 42 43 44 | ok |
| faster_rcnn | base | coco/bbox_mAP | 1 | 0.3320 |  | 44 | ok |
| faster_rcnn | base | coco/bbox_mAP_50 | 1 | 0.9060 |  | 44 | ok |
| faster_rcnn | base | coco/bbox_mAP_75 | 1 | 0.1150 |  | 44 | ok |
| faster_rcnn | base | coco/bbox_mAP_s | 1 | 0.3340 |  | 44 | ok |
| faster_rcnn | base | coco/bbox_mAP_m | 1 | 0.3390 |  | 44 | ok |
| faster_rcnn | base | coco/bbox_mAP_l | 1 | -1.0000 |  | 44 | ok |
| faster_rcnn | dgfe_api | coco/bbox_mAP | 3 | 0.3530 | 0.0200 | 42 43 44 | ok |
| faster_rcnn | dgfe_api | coco/bbox_mAP_50 | 3 | 0.9027 | 0.0095 | 42 43 44 | ok |
| faster_rcnn | dgfe_api | coco/bbox_mAP_75 | 3 | 0.1580 | 0.0344 | 42 43 44 | ok |
| faster_rcnn | dgfe_api | coco/bbox_mAP_s | 3 | 0.3500 | 0.0171 | 42 43 44 | ok |
| faster_rcnn | dgfe_api | coco/bbox_mAP_m | 3 | 0.3647 | 0.0253 | 42 43 44 | ok |
| faster_rcnn | dgfe_api | coco/bbox_mAP_l | 3 | -1.0000 | 0.0000 | 42 43 44 | ok |
| fcos | base | coco/bbox_mAP | 1 | 0.3000 |  | 44 | ok |
| fcos | base | coco/bbox_mAP_50 | 1 | 0.8830 |  | 44 | ok |
| fcos | base | coco/bbox_mAP_75 | 1 | 0.0810 |  | 44 | ok |
| fcos | base | coco/bbox_mAP_s | 1 | 0.2830 |  | 44 | ok |
| fcos | base | coco/bbox_mAP_m | 1 | 0.3330 |  | 44 | ok |
| fcos | base | coco/bbox_mAP_l | 1 | -1.0000 |  | 44 | ok |
| fcos | dgfe_api | coco/bbox_mAP | 3 | 0.3160 | 0.0195 | 42 43 44 | ok |
| fcos | dgfe_api | coco/bbox_mAP_50 | 3 | 0.8847 | 0.0180 | 42 43 44 | ok |
| fcos | dgfe_api | coco/bbox_mAP_75 | 3 | 0.1130 | 0.0139 | 42 43 44 | ok |
| fcos | dgfe_api | coco/bbox_mAP_s | 3 | 0.3000 | 0.0156 | 42 43 44 | ok |
| fcos | dgfe_api | coco/bbox_mAP_m | 3 | 0.3493 | 0.0281 | 42 43 44 | ok |
| fcos | dgfe_api | coco/bbox_mAP_l | 3 | -1.0000 | 0.0000 | 42 43 44 | ok |

## Notes

- GFLOPs/Params were computed with `mmdetection/tools/analysis_tools/get_flops.py`, `--num-images 1`, local `.venv-mmdet`, and `val_dataloader.dataset.pipeline.1.scale=(640,640)` to match the current `train_all_mmdet.py` default; unsupported ops warnings from MMEngine still apply.
- no duplicates or missing parsed runs detected
