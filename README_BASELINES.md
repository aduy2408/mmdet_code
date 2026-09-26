# MMDetection baselines: LEVIR-Ship and TinyPerson

This setup runs RetinaNet R50-FPN, Cascade R-CNN R50-FPN, and RTMDet-S on
both datasets. The launchers also support MMDetection's DETR and DINO-DETR
implementations through the `detr` and `dino` model names.

## Environment

The notebook Python is not used for MMDetection. Install the environment first:

```python
import subprocess

subprocess.run(
    ["bash", "/marimo/mmdet_code/install.sh"],
    check=True,
)
```

All launchers then use:

```text
/marimo/mmdet-venv/bin/python
```

Override paths with `--python`, `--code-root`, `--levir-root`, or
`--tinyperson-root` when the server mount differs.

## Job assignment

| Server | Jobs |
|---|---|
| Machine 1 | LEVIR RetinaNet, TinyPerson Cascade R-CNN, LEVIR RTMDet |
| Machine 2 | TinyPerson RetinaNet, LEVIR Cascade R-CNN, TinyPerson RTMDet |

This balances one-stage and heavier two-stage work across the servers.

## Smoke test first

Machine 1:

```python
subprocess.run([
    "/marimo/mmdet-venv/bin/python",
    "/marimo/mmdet_code/run_two_server_baselines.py",
    "--machine", "1",
    "--smoke-test",
    "--amp",
], check=True)
```

Machine 2 uses the same command with `--machine 2`.

The smoke test uses one epoch and limited samples. Run it on both servers before
starting the full queue because the local machine does not have the same MMCV
and CUDA runtime as Marimo.

## Full run

Machine 1:

```python
subprocess.run([
    "/marimo/mmdet-venv/bin/python",
    "/marimo/mmdet_code/run_two_server_baselines.py",
    "--machine", "1",
    "--epochs", "12",
    "--amp",
], check=True)
```

Machine 2:

```python
subprocess.run([
    "/marimo/mmdet-venv/bin/python",
    "/marimo/mmdet_code/run_two_server_baselines.py",
    "--machine", "2",
    "--epochs", "12",
    "--amp",
], check=True)
```

Add `--resume` after an interrupted run. Use `--dry-run` to generate and inspect
all assigned configs without training.

To run the transformer baselines directly, use one launcher per dataset:

```bash
/marimo/mmdet-venv/bin/python /marimo/mmdet_code/train_all_tinyperson_baseline.py \
  --models detr,dino --epochs 12 --amp

/marimo/mmdet-venv/bin/python /marimo/mmdet_code/train_all_levir_baseline.py \
  --models detr,dino --epochs 12 --amp --no-hf-upload
```

Run the same commands with `--dry-run` first to prepare and inspect the patched
configs. DINO uses the four-scale R50 config and is substantially more
memory-intensive than DETR, so reduce `--batch-size` if CUDA memory is limited.

## Dataset protocol

- **LEVIR-Ship:** reuses the existing scene-safe 70/15/15 split, seed 42, and
  baseline resize protocol. The default image size remains 512 for compatibility
  with the previous launcher. Pass `--image-size 768` to the LEVIR launcher if
  the earlier completed baselines used 768.
  Keep `--split-seed 42` fixed while varying the training `--seed` for a valid
  multi-seed comparison.
- **TinyPerson:** uses the official erased-uncertain training archive and
  640x512 sliding-window annotations. Tiles remain at native scale and are
  cropped dynamically from source images. A deterministic 85/15 split is made
  by source image, preventing overlapping tiles from leaking into validation.
  Keep `--split-seed 42` fixed while varying the training `--seed` across 42,
  43, and 44. The split annotations are written to the shared prepared
  annotation directory, while each training seed must use its own work
  directory.

## Evaluation and outputs

Each model is evaluated on validation and test. Its `final_results.json` stores:

- `map_50_95`
- `ap50`
- `ap75`
- `ap50_tiny1`
- `ap50_tiny2`
- `ap50_tiny3`
- `ap50_small`

TinyPerson tile detections are translated back to full-image coordinates and
merged with class-aware NMS before evaluation. Ignore, uncertain, and logo
regions use COCO crowd matching, which applies intersection-over-detection for
ignored regions. LEVIR has no TinyPerson Tiny1/2/3 definitions, so those fields
are `null`.

Outputs:

```text
mmdetection/work_dirs/levir_baseline/<model>/final_results.json
mmdetection/work_dirs/tinyperson_baseline/<model>/final_results.json
```

## Varroa YOLO-matched protocol

`train_all_mmdet.py` now supports the Varroa protocol used by
`yolo_related/train_all_yolo_baselines_no_mosaic.py`. The explicit
`no_mosaic` variant uses the YOLO-equivalent affine, HSV, flip, resize, pad,
and filter pipeline, MuSGD, 100 epochs, batch size 8, and 8 workers. The
`mosaic` variant uses MMDetection's native `MultiImageMixDataset` and
`Mosaic`, then switches to the no-mosaic pipeline for the final 10 epochs.

```bash
/marimo/mmdet-venv/bin/python mmdetection/train_all_mmdet.py \
  --data-root /marimo/Varroa \
  --models fcos \
  --variants base \
  --yolo-protocol no_mosaic \
  --epochs 100 --early-stop-patience 15 \
  --batch-size 8 --num-workers 8 --img-scale 640 640 \
  --split-seed 42 --seed 42 \
  --python /marimo/mmdet-venv/bin/python \
  --upload-interval-hours 1
```

Use `--yolo-protocol mosaic` to enable the corresponding Mosaic variant.
Run through `utils.marimo_ops` preflight and launch on Marimo. Do not launch
training directly from the local machine.

## VisDrone2019-DET MMDetection baseline matrix

`train_visdrone_mmdet_baselines.py` provides an explicit registry for six
MMDetection baselines:

| Name | Canonical config |
|---|---|
| `fcos` | `configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py` |
| `faster_rcnn` | `configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py` |
| `atss` | `configs/atss/atss_r50_fpn_1x_coco.py` |
| `cascade_rcnn` | `configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py` |
| `rtmdet` | `configs/rtmdet/rtmdet_s_8xb32-300e_coco.py` |
| `retinanet` | `configs/retinanet/retinanet_r50_fpn_1x_coco.py` |

The launcher retains the official `VisDrone2019-DET-train`, `-val`, and
`-test-dev` split, converts annotations to COCO, and applies the same protocol
to every model: 1536x1536, batch 8, 8 workers, MuSGD with lr 0.01, 100 epochs,
early-stop patience 15, and NMS IoU 0.5. The default AMP setting is off and
must be enabled explicitly with `--amp` for all rows of a comparison.

Run the dry-run first in the Marimo MMDetection environment:

```bash
/marimo/mmdet-venv/bin/python mmdetection/train_visdrone_mmdet_baselines.py \
  --data-root /marimo/VisDrone2019 \
  --models fcos,faster_rcnn,atss,cascade_rcnn,rtmdet,retinanet \
  --epochs 100 --early-stop-patience 15 \
  --lr 0.01 --batch-size 8 --workers 8 --image-size 1536 1536 \
  --hf-repo-id <hf-user>/visdrone-mmdet-baselines \
  --dry-run
```

After preflight, launch the same command through `utils.marimo_ops`. Full runs
require `HF_TOKEN` in the detached Marimo child environment and verify each
model's remote prefix before the next model starts. The generated
`experiment_manifest.json` records the exact canonical config, resolved data
root, source commit, seeds, optimizer, runtime settings, and required
artifacts.

For the requested two-server matrix, use
`mmdetection/run_varroa_yolo_two_server.py`. It assigns the four explicit
baselines (`fcos`, `faster_rcnn`, `cascade_rcnn`, `rtmdet`) across two servers,
runs both `no_mosaic` and `mosaic`, and varies training seeds 42 and 43 while
keeping split seed 42 fixed. Each job uses 640x640, 100 epochs, patience 15,
batch size 8, 8 workers, MuSGD, and hourly in-progress HF snapshots.
