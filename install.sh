#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-/marimo/mmdet-venv}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
OPENCV_VERSION="${OPENCV_VERSION:-4.11.0.86}"

uv venv "$VENV_DIR" --python 3.11 --seed
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install "setuptools==60.2.0"

# ------------------------------------------------------------------
# Install PyTorch (CUDA 12.1)
# ------------------------------------------------------------------
python -m pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

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

# ------------------------------------------------------------------
# Install MMCV
# ------------------------------------------------------------------
python -m pip install \
    mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

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
