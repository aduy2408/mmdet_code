#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-/tmp/mmdet-venv}"
BUILD_ROOT="${BUILD_ROOT:-/marimo}"
MMCV_REPO="${MMCV_REPO:-$BUILD_ROOT/mmcv}"
MMDET_REPO="${MMDET_REPO:-$BUILD_ROOT/mmdetection}"
WHEELHOUSE="${WHEELHOUSE:-$BUILD_ROOT/wheelhouse}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$BUILD_ROOT"
if [ ! -d "$MMCV_REPO" ]; then
    git clone https://github.com/open-mmlab/mmcv.git "$MMCV_REPO"
fi

cd "$MMCV_REPO"
export MMCV_WITH_OPS=1
rm -rf build dist
python setup.py bdist_wheel -v

mkdir -p "$WHEELHOUSE/mmcv"
cp dist/*.whl "$WHEELHOUSE/mmcv/"
python -m pip install --force-reinstall "$WHEELHOUSE"/mmcv/mmcv-*.whl

python - <<'PY'
import torch, mmcv
print("torch:", torch.__version__, torch.version.cuda)
print("mmcv :", mmcv.__version__)
PY

python -m pip install -U mmengine

cd "$BUILD_ROOT"
if [ ! -d "$MMDET_REPO" ]; then
    git clone https://github.com/open-mmlab/mmdetection.git "$MMDET_REPO"
fi

cd "$MMDET_REPO"
rm -rf build dist
python setup.py bdist_wheel

mkdir -p "$WHEELHOUSE/mmdetection"
cp dist/*.whl "$WHEELHOUSE/mmdetection/"
python -m pip install --force-reinstall "$WHEELHOUSE"/mmdetection/mmdet-*.whl

python - <<'PY'
import torch
import mmcv
import mmengine
import mmdet

print("torch    :", torch.__version__, "CUDA", torch.version.cuda)
print("mmcv     :", mmcv.__version__)
print("mmengine :", mmengine.__version__)
print("mmdet    :", mmdet.__version__)
PY
