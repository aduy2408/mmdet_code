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

# Low-interference decision runs:
python tools/train.py projects/rcfn_ltmr/configs/fcos_pg_aux_w01.py
python tools/train.py projects/rcfn_ltmr/configs/fcos_pg_ch_w01_floor.py

# Separate LTMR ablation:
python tools/train.py projects/rcfn_ltmr/configs/fcos_l1.py

# Tiny-GT margin and PG-RCFN position diagnostics:
python projects/rcfn_ltmr/tools/diagnose_ltmr.py \
  path/to/config.py path/to/checkpoint.pth \
  --variant pg_h \
  --output-dir path/to/diagnostics
```

The decision pair uses `loss_pos_weight=0.1`. The PG-CH variant additionally
floors the complete product gate as `0.1 + 0.9 * (H * C)`, so the learned gate
can attenuate but cannot erase the R2 enhancement. Select only these runs with:

```bash
python ../train_rcfn_ltmr_levir.py \
  --models pg_aux_w01,pg_ch_w01_floor --epochs 30
```

Dataset, schedule, and work directory overrides use normal MMEngine config
options. The diagnostic writes `tiny_gt_margins.csv`, `summary.json`, and up
to 16 red-target/green-prediction overlays under `position_maps/`. When given
R2 and candidate prediction files, it also writes `paired_gate_gt.csv` and
`paired_gate_summary.json` with retained/lost/gained/R2-miss gate statistics.
