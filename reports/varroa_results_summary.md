# Varroa MMDetection Results Summary

Downloaded Hugging Face snapshots into `hf_runs/` with `*.pt` and `*.pth` ignored.

## Mean By Model/Variant

| model | variant | n | seeds | mAP | mAP50 | mAP75 | per-run mAP | notes |
|---|---:|---:|---|---:|---:|---:|---|---|
| atss | base | 3 | 42,43,44 | 0.335 | 0.907 | 0.119 | 0.343,0.324,0.337 |  |
| atss | dgfe_api | 3 | 42,43,44 | 0.336 | 0.910 | 0.117 | 0.340,0.340,0.328 | multiple test jsons: 2; used latest 20260716_002117 | multiple test jsons: 2; used latest 20260716_004551 |
| cascade | srtod | 1 | 44 | 0.315 | 0.877 | 0.101 | 0.315 | multiple test jsons: 3; used latest 20260713_155800 |
| cascade_rcnn | base | 3 | 42,43,44 | 0.334 | 0.900 | 0.121 | mean from `hf_results/mmdet_results_summary.md` |  |
| cascade_rcnn | dgfe_api | 3 | 42,43,44 | 0.350 | 0.907 | 0.143 | 0.364,0.364,0.322 | multiple test jsons: 3; used latest 20260713_114535 | multiple test jsons: 4; used latest 20260713_114351 |
| dyhead | base | 3 | 42,43,44 | 0.331 | 0.905 | 0.112 | 0.334,0.329,0.330 | multiple test jsons: 2; used latest 20260715_135849 |
| dyhead | dgfe_api | 2 | 42,43 | 0.335 | 0.908 | 0.121 | 0.342,0.327 |  |
| faster | srtod | 1 | 44 | 0.316 | 0.885 | 0.106 | 0.316 | multiple test jsons: 3; used latest 20260713_162458 |
| faster_rcnn | base | 3 | 42,43,44 | 0.338 | 0.897 | 0.116 | mean from `hf_results/mmdet_results_summary.md` |  |
| faster_rcnn | dgfe_api | 3 | 42,43,44 | 0.353 | 0.903 | 0.158 | 0.366,0.363,0.330 | multiple test jsons: 3; used latest 20260713_114426 | multiple test jsons: 3; used latest 20260713_114608 |
| fcos | base | 3 | 42,43,44 | 0.255 | 0.777 | 0.069 | mean from `hf_results/mmdet_results_summary.md` |  |
| fcos | dgfe_api | 3 | 42,43,44 | 0.303 | 0.873 | 0.090 | 0.320,0.320,0.268 |  |
| fcos | srtod | 1 | 44 | 0.287 | 0.865 | 0.086 | 0.287 | multiple test jsons: 3; used latest 20260713_152230 |
| tood | base | 3 | 42,43,44 | 0.335 | 0.884 | 0.125 | 0.334,0.330,0.341 |  |
| tood | dgfe_api | 3 | 42,43,44 | 0.334 | 0.884 | 0.114 | 0.335,0.336,0.331 | multiple test jsons: 2; used latest 20260716_012245 | multiple test jsons: 2; used latest 20260716_014150 |

## Notes

- `vfnet` and `reppoints` are not included in the mean table because the user noted they failed.
- The mean table deduplicates by `(model, variant, seed)` and keeps the latest `finished_at`; all raw rows are still in the full CSV.
- If a run had multiple `test_results/*/*.json`, the latest timestamped JSON was used and noted.
- Selected per-run table: `reports/varroa_results_selected_runs.csv`.
- Full raw per-run table: `reports/varroa_results_all_runs.csv`.
