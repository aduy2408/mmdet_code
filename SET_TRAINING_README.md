# SET training on the Blackwell Marimo server

This repository contains the MMDetection 3.x port of the SET FCOS detector and the launch workflow used for Varroa, LEVIR-Ship, and TinyPerson.

## Canonical Blackwell environment

Use the live Marimo kernel. Do not use the legacy SET environment on the Blackwell GPU.

```text
server:      https://sb-7b6ace927855e6b4.sb.molab.run/
python:      /marimo/mmdet3-blackwell-venv/bin/python
Torch:       2.7.0+cu128
MMDetection: 3.3.0
MMCV:        mmcv-lite 2.1.0 plus blackwell_compat/sitecustomize.py
GPU:         NVIDIA RTX PRO 6000 Blackwell, sm_120
source:      branch set-blackwell-data, commit 7e20ee86 or newer
```

The legacy SET stack uses Torch 1.12.1 and CUDA 11.3. It fails on this GPU with `no kernel image is available for execution on the device`. Do not launch it on Blackwell.

## Required sequential workflow

Run **one full dataset at a time**:

```text
Varroa -> verify local artifacts -> verify HF remote prefix
       -> LEVIR-Ship -> verify local artifacts -> verify HF remote prefix
       -> TinyPerson -> verify local artifacts -> verify HF remote prefix
```

Never launch the three full jobs simultaneously. The GPU is shared, and the upload/evaluation gate for one dataset must complete before the next dataset starts. A smoke run may use a separate work directory, but it must still upload when upload is required.

Before every launch, write a manifest containing:

```text
experiment_id
control_config_or_baseline_config
variant_config_or_explicit_change
source_commit
runner
python_executable
dataset_root
split_seed=42
training_seed
model/backbone/pretrained source
image_size, batch_size, epochs, patience, AMP
NMS IoU
HF repo and remote prefix
required artifacts
upload_required=true
```

Retrieve `HF_TOKEN` from the live Marimo global namespace. Never print it or put it on a command line. Pass it only in the detached process environment.

## Marimo launch pattern

Use `utils.marimo_ops.preflight` and `utils.marimo_ops.launch_detached`. The command must use the exact Python above, must not contain `--no-hf-upload` or `--no-upload`, and must match the manifest. Include the compatibility path in `PYTHONPATH`:

```python
env["HF_TOKEN"] = globals()["HF_TOKEN"]
env["PYTHONPATH"] = os.pathsep.join([
    "/marimo/mmdet_code/blackwell_compat",
    "/marimo/mmdet_code",
    "/marimo/mmdet_code/mmdetection",
])
```

The full Varroa command is:

```text
/marimo/mmdet3-blackwell-venv/bin/python /marimo/mmdet_code/train_all_mmdet.py \
  --data-root /marimo/Varroa \
  --dataset-out /marimo/mmdet_code/work_dirs/set_fcos_blackwell_full/data/varroa_coco \
  --work-dir /marimo/mmdet_code/work_dirs/set_fcos_blackwell_full/varroa_full \
  --models fcos_set --variants base \
  --epochs 12 --batch-size 2 --num-workers 4 \
  --img-scale 640 640 --seed 42 \
  --hf-repo-id duyle2408/set_fcos_runs
```

Use the corresponding `train_all_levir_baseline.py` and `train_all_tinyperson_baseline.py` launchers for the other datasets, passing `--python /marimo/mmdet3-blackwell-venv/bin/python`, `--seed 42`, and `--split-seed 42`.

## Completion gate per dataset

Do not start the next dataset until all checks pass:

1. Training exited successfully.
2. Checkpoint exists, normally `best_*.pth` or `epoch_12.pth`.
3. Evaluation output exists, including predictions and final metrics.
4. The run manifest exists beside the work directory.
5. The launcher uploaded the work directory to Hugging Face.
6. `utils.marimo_ops.verify_hf_remote(repo_id, remote_prefix, token=HF_TOKEN)` returns at least one matching file.

A PID, checkpoint, local marker, or successful upload call alone is not completion evidence.

## Dataset preparation

The complete prepared data is already mirrored on the server. If regeneration is needed, use the repository preparation helpers with split seed 42:

```text
Varroa:     /marimo/Varroa
LEVIR-Ship: /marimo/LevirShip/LevirShipData -> SET/data/levir_ship
TinyPerson: /marimo/TinyPerson -> SET/data/tinyperson_seed42
```

Do not replace the complete Varroa annotations with the original one-image placeholders. The canonical corrected split contains 2,554/451/942 positive images for train/val/test.
