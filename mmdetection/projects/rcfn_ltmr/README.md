# RCFN-R2, PG-RCFN, and LTMR-L1

Minimal FCOS experiments for standardized local P3 enhancement and a
Gaussian-supervised position gate. LTMR remains a separate training-only
local tiny-object logit-margin ablation.

The standalone configs target LEVIR-Ship by default, using annotations under
`data/levir_ship_coco/` and images under `../LevirShipData/All Images/`.
Tiny-object eligibility is measured in original-image coordinates; Gaussian
centers and widths remain aligned to the resized training feature map.

```bash
# Baseline and R2:
python tools/train.py projects/rcfn_ltmr/configs/fcos_baseline.py
python tools/train.py projects/rcfn_ltmr/configs/fcos_r2.py

# Position auxiliary loss, H gate, and C x H gate:
python tools/train.py projects/rcfn_ltmr/configs/fcos_pg_aux.py
python tools/train.py projects/rcfn_ltmr/configs/fcos_pg_h.py
python tools/train.py projects/rcfn_ltmr/configs/fcos_pg_ch.py

# Separate LTMR ablation:
python tools/train.py projects/rcfn_ltmr/configs/fcos_l1.py

# Tiny-GT margin and PG-RCFN position diagnostics:
python projects/rcfn_ltmr/tools/diagnose_ltmr.py \
  path/to/config.py path/to/checkpoint.pth \
  --variant pg_h \
  --output-dir path/to/diagnostics
```

Dataset, schedule, and work directory overrides use normal MMEngine config
options. The diagnostic writes `tiny_gt_margins.csv`, `summary.json`, and up
to 16 red-target/green-prediction overlays under `position_maps/`.
