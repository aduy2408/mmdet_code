# Position-aware Haar Recomposition

PAHR is a project-local FCOS ablation for LEVIR-Ship. It applies a fixed Haar
transform to P3, predicts a supervised tiny-object position/offset map, and
uses that map to gate learned residual corrections to the three detail bands.
P4 through P7 and the FCOS prediction interface are unchanged.

```bash
python ../train_all_levir_baseline.py \
  --models pahr --epochs 12 --no-hf-upload

python projects/pahr/tools/diagnose_pahr.py \
  path/to/patched_config.py path/to/checkpoint.pth \
  --output-dir work_dirs/pahr_diagnostics
```
