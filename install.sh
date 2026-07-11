#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-/tmp/mmdet-venv}"
MMDET_WHEEL_DIR="${MMDET_WHEEL_DIR:-/marimo/mmdet_code}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"

uv venv "$VENV_DIR" --python 3.11 --seed
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ------------------------------------------------------------------
# Install PyTorch (CUDA 13.0)
# ------------------------------------------------------------------
python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu130

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
python -m pip install --force-reinstall "$MMDET_WHEEL_DIR"/mmcv-*.whl

python - <<'PY'
import torch, mmcv
print("torch:", torch.__version__, torch.version.cuda)
print("mmcv :", mmcv.__version__)
PY

# ------------------------------------------------------------------
# Install MMEngine
# ------------------------------------------------------------------
python -m pip install -U mmengine

# ------------------------------------------------------------------
# Install MMDetection
# ------------------------------------------------------------------
python -m pip install --force-reinstall "$MMDET_WHEEL_DIR"/mmdet-*.whl

# ------------------------------------------------------------------
# Pin NumPy
# ------------------------------------------------------------------
python -m pip install --force-reinstall "numpy==${NUMPY_VERSION}"

# ------------------------------------------------------------------
# Verify
# ------------------------------------------------------------------
python - <<'PY'
import torch
import mmcv
import mmengine
import mmdet
import numpy as np

print("torch    :", torch.__version__, "CUDA", torch.version.cuda)
print("mmcv     :", mmcv.__version__)
print("mmengine :", mmengine.__version__)
print("mmdet    :", mmdet.__version__)
print("numpy    :", np.__version__)
PY
