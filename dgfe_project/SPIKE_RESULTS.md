# Two-stage DGFE architecture spike

## Decision

Keep these two production variants:

1. `dgfe_api` (default): model-specific detector plus model-specific ROI head.
2. `dgfe_api_roi`: stock detector plus model-specific ROI head discovered by
   the generic adapter hook.

Both variants use exact `SamplingResult.pos_assigned_gt_inds`, decoded positive
ROI predictions, and aligned prediction/GT IoU. Faster R-CNN exports its only
ROI stage. Cascade R-CNN stores every stage and exports only the final stage to
the DGFE spatial target. Boxgrad accepts ROI `loss_bbox` keys and excludes RPN
and classification losses.

FCOS uses `FCOSDGFEHead` to decode positive point targets and predictions per
FPN level. Its spatial quality is aligned predicted-box IoU against the GT
assigned by FCOS; centerness is deliberately excluded. A synthetic DGFE/API
run produced 16 assignment records, included
`loss_dgfe_spatial`, and retained one backbone/neck pass with two head passes.

## Candidate outcome

| Candidate | Result | Reason |
| --- | --- | --- |
| Detector-only (SR-TOD style) | Rejected | A stock ROI head does not return sampling metadata. Instrumenting it turns this into the hybrid candidate while retaining duplicated detector flow. |
| ROI-head subclass | Required component | It is the only clean point with exact sampled assignments and decoded bbox predictions. It needs either the generic or hybrid retrieval policy below. |
| Generic adapter | Kept as `dgfe_api_roi` | Correct and smallest diff. It scans for the ROI adapter and has no replay-specific detector API. |
| Hybrid detector + ROI head | Kept as `dgfe_api` | Same metadata and runtime, explicit ROI policy, and exposes `dgfe_replay_losses()` for future partial forward. |

The four proposals are not four independent end-to-end implementations:
candidate B provides metadata, while C and D are the two viable orchestration
choices around B. Candidate A cannot pass the metadata correctness gate without
becoming D.

## Reproducible smoke benchmark

Command (CPU, 64x64 synthetic image, one warmup, three measured iterations):

```bash
.venv-mmdet/bin/python dgfe_project/tools/spike_two_stage.py \
  --size 64 --warmup 1 --iterations 3 <patched-configs...>
```

| Model | Variant | Mean loss+backward | Passes/iteration | Replay seam |
| --- | --- | ---: | ---: | --- |
| Faster R-CNN | `dgfe_api` | 1.2045 s | backbone/neck 1, heads 2 | yes |
| Faster R-CNN | `dgfe_api_roi` | 1.2321 s | all modules 2 | no |
| Cascade R-CNN | `dgfe_api` | 3.1859 s | backbone/neck 1, heads 2 | yes |
| Cascade R-CNN | `dgfe_api_roi` | 4.0950 s | all modules 2 | no |
| ATSS | `dgfe_api` | 0.3671 s | backbone/neck 1, head 2 | yes |

CPU timings are a smoke comparison, not a training throughput claim. CUDA peak
memory was not available in this run; the benchmark reports it automatically
when run on CUDA. Fixed seed produced equal assignment counts between the two
variants for each model.

A deterministic Faster R-CNN parity check copied identical weights into the
replay and full-recompute variants. All loss keys matched with zero absolute
loss difference. After backward, 170 parameter-gradient tensors matched with a
maximum absolute difference of `1.21e-8`.

## Partial-forward finding

API now caches the post-DGFE FPN tuple and perturbs a copied tuple. Single-stage
models replay `bbox_head.loss()`. Hybrid two-stage models replay
`loss_from_features()`, so RPN proposals and ROI losses are recomputed while
backbone, FPN, and DGFE run once. Stock two-stage `dgfe_api_roi` models retain
the full-recompute fallback.
