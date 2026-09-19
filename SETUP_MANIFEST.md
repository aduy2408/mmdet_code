# SET setup manifest

## Control and variant

- `experiment_id`: `setup-only-set-three-datasets`
- `control_config_or_baseline_config`: upstream `configs/aitod/fcos_r50_baseline.py`, commit `9208fbc4cfe571be4c15dccad8db1665cfdcb9d6`
- `variant_config_or_explicit_change`: six dataset overlays under `configs/aitod/`; explicit changes are COCO dataset paths/classes and `bbox_head.num_classes=1`
- `source_commit`: `9208fbc4cfe571be4c15dccad8db1665cfdcb9d6`
- `runner`: upstream `scripts/train.sh` and `tools/train.py`; no training launched
- `python_executable`: scratch Python 3.10 runtime used for acceptance; project `.mmdet-venv` is absent
- `dataset_root`: project-local defaults documented in `README_DATASETS.md`; LEVIR requires generated `data/levir_ship`
- `split_seed`: TinyPerson prepared split seed `42`; Varroa and LEVIR inherited from existing project annotations; no new split created
- `training_seed(s)`: upstream config seed `42`; no training run
- `model/backbone/pretrained source`: upstream FCOS + ResNet-50 + `torchvision://resnet50`
- `image_size, batch_size, epochs, patience, AMP`: upstream config defaults; no overrides introduced; exact runtime values are not an experiment result
- `NMS IoU`: upstream config `0.5`
- `HF repo and remote prefix`: not applicable, no training/upload requested
- `required artifacts`: repository setup, six configs, LEVIR/TinyPerson preparation helpers, setup documentation
- `upload_required=false`

## State

`implementation -> local validation -> public-interface acceptance`

Acceptance checks performed after setup:

- `python tools/prepare_levir_ship.py` completed against the real project data and produced `2728` train, `584` validation, and `584` test image symlinks under `SET/data/levir_ship`.
- `python tools/prepare_tinyperson.py` completed against the official erased-uncertain archive and corner annotation, producing a seed-42 source-safe split with `634` train sources and `112` validation sources under `SET/data/tinyperson_seed42`.
- All six public dataset config paths and all default Varroa/TinyPerson annotation paths exist. The configs were loaded through a real OpenMMLab `Config.fromfile`-compatible loader with `SET_ROOT` set, verifying inheritance, one-class heads, dataset paths, and inherited pipelines for all six configs. `SET_*_ROOT` overrides were also verified.
- This boundary check caught and fixed two compatibility issues before completion: config files cannot rely on `__file__` being injected by MMDetection v2, and dataset overlays must not duplicate the upstream AITOD base keys. The final overlays use `SET_ROOT`/cwd defaults and merge only dataset fields.
- Exhaustive COCO/image checks passed for Varroa and LEVIR-Ship. TinyPerson was additionally checked through the real corner loader: every train/val/test config uses `LoadTinyPersonImageFromFile`, official corner metadata, erased-uncertain train images, 640x512 crops, stride-32 padding, and one real sample per split.
- Comparing every dataset overlay with its upstream baseline/SET control confirmed model, optimizer, optimizer config, LR config, seed, workflow, checkpoint config, and inherited pipelines are unchanged; only dataset fields and `bbox_head.num_classes=1` differ.
- In a scratch Python 3.10 runtime with `torch==1.12.1+cu113` and `mmcv-full==1.6.0`, the upstream `tools/train.py --help` entrypoint passed, all three SET variants built through the real `mmdet.datasets` registry for train/val/test, and one real training sample loaded through each train pipeline. FCOS and custom `FCOS_set` model factories also built successfully.
- The exact runtime check found one compatibility gap from the completed v2.28 dataset backport: `mmdet.utils.get_device` was absent from SET's v2.23 utility exports. A small compatibility implementation was added and the registry/sample checks then passed.
- The TinyPerson audit found that standard `LoadImageFromFile` would have been wrong because it reads the full source image while annotations are tile-relative. SET now crops the official `corner` window before resize/normalize, matching the project TinyPerson protocol.
- The host Python environment lacks `torch`/`mmcv` and `.mmdet-venv` is absent, but the exact SET dependency runtime was created in scratch and `python tools/train.py --help` passed there. Full training was intentionally not launched.

No Marimo preflight, smoke run, training, evaluation, or upload was performed. Any future experiment must create a separate manifest and preserve the upstream baseline files.
