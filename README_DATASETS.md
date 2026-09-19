# SET dataset setup for Varroa, TinyPerson, and LEVIR-Ship

This checkout is pinned to the upstream SET commit `9208fbc4cfe571be4c15dccad8db1665cfdcb9d6`.
The upstream model, optimizer, schedule, runtime, image pipeline, and seed are kept unchanged. The additions here only provide the missing MMDetection v2 dataset package, the missing upstream AITOD compatibility base, dataset paths, and one-class model overlays.

## Environment

Use the upstream environment for SET. It requires Python 3.9, PyTorch 1.12.1, torchvision 0.13.1, CUDA 11.3, and `mmcv-full==1.6.0`:

```bash
cd SET
conda create -n set python=3.9 -y
conda activate set
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -U openmim
mim install mmcv-full==1.6.0
pip install -v -e .
pip install -r requirements/runtime.txt
pip install -v -e cocoapi-aitod-master/aitodpycocotools
```

The project currently has no `.mmdet-venv`; no training or dependency installation was run during setup.

## Dataset preparation

Varroa and TinyPerson use the existing project-local data by default. TinyPerson uses the prepared seed-42, source-image-separated corner split:

```text
Varroa:     ../mmdetection/mmdetection/data/varroa_coco
TinyPerson: ../TinyPerson/tiny_set
            official corner annotations: ../TinyPerson/tiny_set/annotations/corner/task
            training images: extracted erased-uncertain archive
```

Prepare the official TinyPerson 640x512 sliding-window corner protocol:

```bash
cd SET
python tools/prepare_tinyperson.py
```

The helper safely extracts `erase_with_uncertain_dataset/train.tar.gz`, creates a
source-image-separated seed-42 split, and writes ignored data under
`SET/data/tinyperson_seed42/`. The configs use `LoadTinyPersonImageFromFile` to
crop each `corner=[x1,y1,x2,y2]` window before applying the unchanged SET
resize/normalize pipeline. Test uses the official single-class corner annotation
`tiny_set_test_sw640_sh512_all.json`.

LEVIR-Ship annotations contain only basenames while the source images are nested. Prepare a flat symlink index inside SET:

```bash
cd SET
python tools/prepare_levir_ship.py
```

The command writes `SET/data/levir_ship/`, which is ignored as data and can be regenerated. It does not alter the MMDetection reference repository.

Override any root without editing configs:

```bash
export SET_ROOT=/path/to/SET
export SET_VARROA_ROOT=/path/to/varroa_coco
export SET_TINYPERSON_ROOT=/path/to/tiny_set
export SET_TINYPERSON_PREPARED_ROOT=/path/to/tinyperson_seed42
export SET_LEVIR_ROOT=/path/to/prepared/levir_ship
```

## Configs

Each dataset has a baseline and a SET variant:

```text
configs/aitod/fcos_r50_varroa_baseline.py
configs/aitod/fcos_r50_varroa_set.py
configs/aitod/fcos_r50_tinyperson_baseline.py
configs/aitod/fcos_r50_tinyperson_set.py
configs/aitod/fcos_r50_levir_ship_baseline.py
configs/aitod/fcos_r50_levir_ship_set.py
```

The Varroa and LEVIR-Ship SET variants inherit the unchanged upstream
`fcos_r50_set.py`. TinyPerson uses the same upstream SET model definition via a
small explicit overlay because its dataset pipeline must replace the first
loader with the corner-aware loader. Apart from dataset fields, the only model
adaptation is `bbox_head.num_classes=1`.

A later training launch should use the upstream script, for example:

```bash
bash scripts/train.sh configs/aitod/fcos_r50_varroa_set.py 1 work_dirs/varroa_set
```

Do not add ad-hoc hyperparameter flags if the goal is to reproduce SET defaults.
