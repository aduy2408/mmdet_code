# SR-TOD custom-model setup

This directory contains generated, non-canonical configs for the SR-TOD
`SRTOD_FasterRCNN` variant. The model uses the paper's reconstruction branch
and difference-guided feature enhancement (DGFE) module. It is the selected
model for the Varroa, TinyPerson, and LEVIR-Ship rerun.

## Selected model

- Model: `SRTOD_FasterRCNN`
- Backbone: ResNet-50
- Pretrained source: `torchvision://resnet50`
- Variant source: `SR-TOD/srtod_project/srtod_cascade_rcnn/config/srtod-cascade-rcnn_r50_fpn_visdrone.py`
- Classes: one class per dataset (`varroa`, `person`, or `ship`)
- NMS IoU: 0.5
- Training: not launched by this setup

## Dataset configs

- `varroa_srtod_faster_r50_fpn.py`
- `tinyperson_srtod_faster_r50_fpn.py`
- `levir_ship_srtod_faster_r50_fpn.py`
- `experiment_manifest.json`

- TinyPerson uses the prepared scene-safe train/validation split and the official
  `TinyPerson/tiny_set/annotations/task/tiny_set_test_all.json` test annotations.
  The generated configs do not mix the training validation annotations with the
  test image archive.

TinyPerson images are still archived in the current checkout. Extract them before
the Marimo smoke test:

```bash
tar -xzf TinyPerson/tiny_set/erase_with_uncertain_dataset/train.tar.gz \
  -C TinyPerson/tiny_set/erase_with_uncertain_dataset

tar -xzf TinyPerson/tiny_set/erase_with_uncertain_dataset/test.tar.gz \
  -C TinyPerson/tiny_set/erase_with_uncertain_dataset
```

The setup generator is rerunnable and only rewrites this generated directory:

```bash
python mmdetection/setup_srtod_datasets.py
```

## Marimo handoff

Use the live Marimo kernel and the project Marimo/MMDetection workflow for the
next state transition. The intended executable is:

```text
/marimo/mmdet-venv/bin/python
```

Before training, run a one-epoch smoke test for exactly one generated config,
then verify the checkpoint, evaluation output, and any required remote artifact
paths. Training seeds and Hugging Face destination are intentionally `unknown`
in this setup manifest because they were not requested yet.

The local `/mnt/data/varroa/mmdetection/.venv-mmdet` was inspected but is not a
usable training environment in this checkout: it contains a torch namespace
without `torch.Tensor`. Do not use it for the Marimo run.
