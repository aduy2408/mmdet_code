#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/marimo/mmdet-venv}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
OPENCV_VERSION="${OPENCV_VERSION:-4.11.0.86}"
CUDA_FLAVOR="${CUDA_FLAVOR:-cu121}"
BUILD_ONLY="${BUILD_ONLY:-0}"
MMCV_WHEEL_GLOB="${MMCV_WHEEL_GLOB:-$SCRIPT_DIR/mmcv-*.whl}"

uv venv "$VENV_DIR" --python 3.11 --seed
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install "setuptools==60.2.0"

# ------------------------------------------------------------------
# Install PyTorch
# ------------------------------------------------------------------
if [ "$CUDA_FLAVOR" = "cu130" ]; then
    python -m pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu130
else
    python -m pip install torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cu121
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

# ------------------------------------------------------------------
# Install CUDA Toolkit (Debian 13)
# ------------------------------------------------------------------
wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get install -y cuda-toolkit-13-1

export CUDA_HOME=/usr/local/cuda-13.1
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  export CUDA_HOME="$(dirname "$(dirname "$(find /usr/local -name nvcc | head -n 1)")")"
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME=$CUDA_HOME"
which nvcc
nvcc --version

if [ "$BUILD_ONLY" = "1" ]; then
    echo "Build-only env ready. Run ./build_mmdet_wheels.sh next."
    exit 0
fi

# ------------------------------------------------------------------
# Install MMCV
# ------------------------------------------------------------------
if ! compgen -G "$MMCV_WHEEL_GLOB" > /dev/null; then
    echo "Missing MMCV wheel: $MMCV_WHEEL_GLOB"
    exit 1
fi
python -m pip install --force-reinstall $MMCV_WHEEL_GLOB

python - <<'PY'
import torch, mmcv
print("torch:", torch.__version__, torch.version.cuda)
print("mmcv :", mmcv.__version__)
PY

# ------------------------------------------------------------------
# Install MMEngine + MMDetection
# ------------------------------------------------------------------
python -m pip install \
    "mmengine>=0.7.1,<1.0.0" \
    "mmdet==3.3.0"

# ------------------------------------------------------------------
# Pin NumPy + OpenCV
# ------------------------------------------------------------------
python -m pip install --force-reinstall "numpy==${NUMPY_VERSION}"
python -m pip install --force-reinstall --no-deps "opencv-python==${OPENCV_VERSION}"

# ------------------------------------------------------------------
# Verify
# ------------------------------------------------------------------
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
