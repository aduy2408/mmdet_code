#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-/marimo/mmdet-venv}"
BUILD_ROOT="${BUILD_ROOT:-/marimo}"
MMCV_REPO="${MMCV_REPO:-$BUILD_ROOT/mmcv}"
MMDET_REPO="${MMDET_REPO:-$BUILD_ROOT/mmdetection}"
WHEELHOUSE="${WHEELHOUSE:-$BUILD_ROOT/wheelhouse}"
MMCV_REF="${MMCV_REF:-v2.1.0}"
MMDET_REF="${MMDET_REF:-v3.3.0}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
OPENCV_VERSION="${OPENCV_VERSION:-4.11.0.86}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;12.0}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$BUILD_ROOT"
if [ ! -d "$MMCV_REPO" ]; then
    git clone https://github.com/open-mmlab/mmcv.git "$MMCV_REPO"
fi

cd "$MMCV_REPO"
git fetch --tags
git checkout "$MMCV_REF"
export MMCV_WITH_OPS=1
export MAX_JOBS
export TORCH_CUDA_ARCH_LIST
rm -rf build dist
python setup.py bdist_wheel -v

mkdir -p "$WHEELHOUSE/mmcv"
rm -f "$WHEELHOUSE"/mmcv/mmcv-*.whl
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
git fetch --tags
git checkout "$MMDET_REF"
rm -rf build dist
python setup.py bdist_wheel

mkdir -p "$WHEELHOUSE/mmdetection"
rm -f "$WHEELHOUSE"/mmdetection/mmdet-*.whl
cp dist/*.whl "$WHEELHOUSE/mmdetection/"
python -m pip install --force-reinstall "$WHEELHOUSE"/mmdetection/mmdet-*.whl
python -m pip install --force-reinstall "numpy==${NUMPY_VERSION}"
python -m pip install --force-reinstall --no-deps "opencv-python==${OPENCV_VERSION}"

python - <<'PY'
import torch
import mmcv
import mmengine
import mmdet
import numpy as np
import cv2

print("torch    :", torch.__version__, "CUDA", torch.version.cuda)
print("mmcv     :", mmcv.__version__)
print("mmengine :", mmengine.__version__)
print("mmdet    :", mmdet.__version__)
print("numpy    :", np.__version__)
print("opencv   :", cv2.__version__)
PY
