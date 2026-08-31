# MMDetection baselines: LEVIR-Ship and TinyPerson

This setup runs RetinaNet R50-FPN, Cascade R-CNN R50-FPN, and RTMDet-S on
both datasets. The six jobs are split across two independent Marimo servers.

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

## Dataset protocol

- **LEVIR-Ship:** reuses the existing scene-safe 70/15/15 split, seed 42, and
  baseline resize protocol. The default image size remains 512 for compatibility
  with the previous launcher. Pass `--image-size 768` to the LEVIR launcher if
  the earlier completed baselines used 768.
- **TinyPerson:** uses the official erased-uncertain training archive and
  640x512 sliding-window annotations. Tiles remain at native scale and are
  cropped dynamically from source images. A deterministic 85/15 split is made
  by source image, preventing overlapping tiles from leaking into validation.

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
