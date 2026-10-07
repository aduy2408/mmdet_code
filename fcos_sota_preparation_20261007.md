# FCOS SOTA preparation record

Date: 2026-10-07
Status: setup and static preflight only. No training or evaluation was launched.

## FCOS-SET

- experiment_id: `fcos-set-eval-prep-20261007`
- baseline: `mmdetection/mmdetection/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py`
- variant: `mmdetection/mmdetection/configs/set/fcos_r50_set.py`
- source_commit: `19672cf557bea70f7919ce6762d50fe3de084330`
- model: `FCOS_set`, ResNet-50, `torchvision://resnet50`
- runner: `mmdetection/train_all_mmdet.py` plus dataset-specific launchers
- canonical Marimo Python: `/marimo/mmdet3-blackwell-venv/bin/python`
- dataset roots: `/marimo/Varroa`, `/marimo/LevirShip/LevirShipData`, `/marimo/TinyPerson`
- split_seed: `42`
- training_seed: `42`
- protocol: `640x640`, batch `8`, workers `8`, `100` epochs, patience `0`, AMP `false`, MuSGD `lr=0.01`, NMS IoU `0.5`
- HF repo: `unknown until the evaluation run contract is created`
- upload_required: `true` for the eventual evaluation workflow
- sequencing: Varroa -> artifact/HF verification -> LEVIR-Ship -> verification -> TinyPerson
- canonical environment warning: the legacy `/marimo/set-venv` Torch 1.12.1/CUDA 11.3 stack must not be used on Blackwell

## FCOS-SR-TOD

- experiment_id: `fcos-sr-tod-eval-prep-20261007`
- baseline: `unknown: no canonical SR-TOD baseline manifest was recorded`
- variant: `mmdetection/srtod_setup/{varroa,levir_ship,tinyperson}_srtod_faster_r50_fpn.py`
- active source commit: `mmdetection=19672cf557bea70f7919ce6762d50fe3de084330`
- upstream reference commit: `SR-TOD-upstream=1c4d93b104450055ecbfed178e6ffbbf1aedc9f6`
- model: `SRTOD_FasterRCNN`, ResNet-50, `torchvision://resnet50`, reconstruction + DGFE
- runner: SR-TOD/MMDetection `tools/train.py` or an explicit evaluation runner, to be selected in the run contract
- canonical Marimo Python: `/marimo/mmdet-venv/bin/python`
- dataset annotations: generated COCO train/val/test configs for Varroa, LEVIR-Ship, TinyPerson
- split_seed: `42`
- training_seed: `42` for any reproduction run, not used by evaluation-only checkpoint loading
- current generated-config defaults: `1333x800`, batch `2`, workers `2`, `12` epochs, NMS IoU `0.5`
- HF repo: `unknown until the evaluation run contract is created`
- upload_required: `true` for the eventual evaluation workflow
- blocker before launch: verify checkpoint availability and exact test protocol for LEVIR-Ship and TinyPerson, extract TinyPerson archives if the target server is not already prepared, and correct the stale generated manifest/source-description mismatch before creating the final run contract

## Validation performed

- Python compilation passed for the SR-TOD generator and all three generated configs.
- FCOS-SET canonical config exists and was inspected.
- SR-TOD canonical config and custom implementation paths exist.
- No remote process, training job, evaluation job, or Hugging Face upload was started by this preparation step.
