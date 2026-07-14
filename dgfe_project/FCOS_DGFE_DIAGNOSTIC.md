# FCOS DGFE Diagnostic (Positive 20/10, Seed 42)

## Setup

- 20 positive train images, 10 disjoint positive val images from `data`.
- FCOS ResNet-50 Caffe pretrained backbone, batch 4, CUDA, 10 epochs.
- Fixed SGD LR 0.001, no iteration warmup or LR milestones.
- Metrics are diagnostic only; the validation set has 11 instances.

## Results

| Run | Best epoch | mAP | mAP50 |
|---|---:|---:|---:|
| FCOS base | 8 | 0.180 | 0.656 |
| DGFE module, aux off | 6 | 0.134 | 0.463 |
| DGFE reconstruction only | 10 | 0.119 | 0.473 |
| DGFE + corrected IoU spatial | 10 | **0.227** | 0.536 |
| DGFE IoU + boxgrad API | 10 | 0.149 | 0.542 |
| DGFE IoU, spatial gain 0.01 | 10 | 0.181 | 0.590 |
| YOLO ultra DGFE control | 1-10 | 0.000 | 0.000 |

The same 50-step control does not show that YOLO is intrinsically stable while
FCOS is not. The successful historical YOLO runs use substantially more steps.

## Findings

1. The original FCOS smoke run omitted pretrained initialization. The full run
   then collapsed exactly when LR jumped from 0.00333 to 0.01 at iteration 500;
   batch 8 was using the LR intended for base batch 16.
2. MMDetection reconstructed a per-image min-max stretched Caffe-normalized
   tensor. YOLO reconstructs an image in `[0, 1]`. The port now reverses the
   detector mean/std and uses one shared image target for gating and loss.
3. YOLO gains are not optimizer-invariant. At gain 0.45, the measured FCOS
   spatial gradient norm on DGFE parameters was 0.315 versus 0.000918 from the
   detection losses. Threshold moved from 0.010 to 0.247 in 50 steps. Gain
   0.01 kept threshold at 0.054, but the tiny validation set cannot select the
   production gain reliably.
4. API is the first clear negative component in this diagnostic: adding
   boxgrad to the best IoU setup reduced mAP from 0.227 to 0.149 and mean
   assigned IoU in the probe from 0.466 to 0.425. Perturbation norm itself was
   correct at 0.005.

## Recommended Full Run

- Use pretrained initialization and batch-scaled LR (`0.005` for batch 8), or
  pass an explicit validated LR.
- DGFE spatial supervision always uses assigned localization IoU.
- Enable DGFE/API warmup for the full-data experiment.
- Compare DGFE-only before enabling API; sweep spatial gain on the real val set.
- Do not use a per-image min-max reconstruction target.

Gradient probe JSON files and checkpoints are under the corresponding
`mmdetection/work_dirs/fcos_dgfe_diag_*` directories.
