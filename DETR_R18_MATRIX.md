# DETR-R18 / RT-DETR-R18 matrix

`run_detr_r18_rtdetr_r18_matrix.py` is the single tracked launcher to sync to both Marimo checkouts. It defines a variant-aware matrix:

- Models: `detr_r18`, `rtdetr_r18`
- Datasets: Varroa, LEVIR-Ship, TinyPerson, VisDrone
- Training seeds: `42`, `43`
- Split seed: `42`
- Epochs: `100`
- Patience: `15`
- Optimizer: MuSGD, `lr=0.01`, momentum `0.9`, Nesterov, weight decay `0.0005`, `muon=0.2`, `sgd=1.0`
- NMS IoU: `0.5`
- Uploads: required to `duyle2408/detr_r18_rtdetr_r18_matrix_runs`

Varroa, LEVIR-Ship, and TinyPerson each run both `no_mosaic` and `mosaic`.
VisDrone currently has only the official `no_mosaic` path because its launcher
does not yet implement a Mosaic dataset wrapper. The resulting matrix has 28
jobs. Do not duplicate VisDrone's no-Mosaic command under a fake Mosaic label.

## Per-dataset protocol

| Dataset | Runner | Image | Batch | Workers | Variant | Window images |
|---|---|---:|---:|---:|---|---|
| Varroa | `train_all_mmdet.py` | 640x640 | 8 | 8 | `no_mosaic`, `mosaic` | no |
| LEVIR-Ship | `train_all_levir_baseline.py` | 512x512 | 4 | 4 | `no_mosaic`, `mosaic` | no |
| TinyPerson | `train_all_tinyperson_baseline.py` | 640x640 | 8 | 8 | `no_mosaic`, `mosaic` | **yes** |
| VisDrone | `train_visdrone_mmdet_baselines.py` | 1536x1536 | 8 | 8 | `musgd_default` | no |

TinyPerson is not trained on full source images. The launcher passes the canonical TinyPerson root and a seed-specific prepared annotation directory. The existing TinyPerson runner materializes the erased training window images and keeps the source-image split fixed.

## Local dry-run

Run this after syncing the file into the repository. The command writes per-job manifests under the selected manifest root and prints the exact commands without training:

```bash
python mmdetection/run_detr_r18_rtdetr_r18_matrix.py \
  --machine 1 \
  --dry-run \
  --python /marimo/mmdet-venv/bin/python \
  --rtdetr-root /marimo/rtdetr-mmdet \
  --manifest-root /marimo/mmdet_code/work_dirs/detr_r18_matrix/manifests \
  --work-root /marimo/mmdet_code/work_dirs/detr_r18_matrix
```

Use `--machine 2` for the second shard. The assignment is deterministic and printed before any launch. Each manifest and work directory includes the augmentation variant, so Mosaic and non-Mosaic runs cannot overwrite each other.

## Sync and prepare each Marimo server

Sync the same committed checkout to both servers. Then prepare the pinned external RT-DETR checkout outside the main worktree:

```python
import subprocess

subprocess.run([
    "bash", "/marimo/mmdet_code/install.sh",
], check=True)
subprocess.run([
    "/marimo/mmdet-venv/bin/python",
    "/marimo/mmdet_code/mmdetection/setup_rtdetr_mmdet.py",
    "--root", "/marimo/rtdetr-mmdet",
], check=True)
```

The launcher requires a clean, exact checkout for a real run. `HF_TOKEN` must be read from the live Marimo global namespace and passed only into the detached child environment.

## Launch machine 1

```python
import os
import subprocess

env = os.environ.copy()
env["HF_TOKEN"] = globals()["HF_TOKEN"]
env["PYTHONPATH"] = os.pathsep.join([
    "/marimo/mmdet_code",
    "/marimo/mmdet_code/mmdetection",
])

subprocess.run([
    "/marimo/mmdet-venv/bin/python", "-m", "utils.marimo_ops", "launch",
    "--cwd", "/marimo/mmdet_code",
    "--run-dir", "/marimo/mmdet_code/work_dirs/detr_r18_matrix/supervisor1",
    "--artifact-root", "/marimo/mmdet_code/work_dirs/detr_r18_matrix",
    "--",
    "/marimo/mmdet-venv/bin/python",
    "/marimo/mmdet_code/mmdetection/run_detr_r18_rtdetr_r18_matrix.py",
    "--machine", "1",
    "--python", "/marimo/mmdet-venv/bin/python",
    "--rtdetr-root", "/marimo/rtdetr-mmdet",
], cwd="/marimo/mmdet_code", env=env, check=True)
```

Use the identical cell on the second server with `supervisor2` and `--machine 2`. Do not add `--no-hf-upload` or `--no-upload`. Before the real launch, run the same command with `--dry-run` and inspect every generated manifest.

## Completion evidence

For each of the 16 jobs, require:

- generated experiment manifest beside the work directory
- resolved canonical DETR or pinned RT-DETR config
- checkpoint and final test results
- validation and test metrics
- Hugging Face remote prefix verified
- queue state marked `completed`

A running PID, checkpoint, or successful upload call alone is not completion evidence.
