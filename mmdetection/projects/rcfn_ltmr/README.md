# RCFN-R2 + LTMR-L1

Minimal FCOS experiments for standardized local P3 enhancement and a
training-only local tiny-object logit margin.

```bash
# Baseline, R2-only, or L1-only:
python tools/train.py projects/rcfn_ltmr/configs/fcos_baseline.py
python tools/train.py projects/rcfn_ltmr/configs/fcos_r2.py
python tools/train.py projects/rcfn_ltmr/configs/fcos_l1.py

# Per-tiny-GT P3 margin diagnostics:
python projects/rcfn_ltmr/tools/diagnose_ltmr.py \
  path/to/config.py path/to/checkpoint.pth \
  --output-dir path/to/diagnostics
```

Dataset, schedule, and work directory overrides use normal MMEngine config
options. The diagnostic writes `tiny_gt_margins.csv` and `summary.json`.
