#!/usr/bin/env bash
set -euo pipefail

# Build an isolated MMDetection 2.x runtime for HuixinSun/SET.
# This intentionally does not touch the MMDetection 3.x installer or venv.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/../SET" ]]; then
    DEFAULT_SET_DIR="$SCRIPT_DIR/../SET"
elif [[ -d "$SCRIPT_DIR/../../SET" ]]; then
    DEFAULT_SET_DIR="$SCRIPT_DIR/../../SET"
else
    DEFAULT_SET_DIR="/marimo/SET"
fi
SET_DIR="${SET_DIR:-$DEFAULT_SET_DIR}"
VENV_DIR="${VENV_DIR:-/marimo/mmdet2-venv}"
# Do not honor the generic PYTHON_VERSION from a Marimo kernel. Those sessions
# commonly export 3.13, which has no Torch 1.12 wheel. Override deliberately
# with SET_PYTHON_VERSION when needed.
PYTHON_VERSION="${SET_PYTHON_VERSION:-3.10}"
MAX_JOBS="${MAX_JOBS:-4}"
INSTALL_CUDA_TOOLKIT="${INSTALL_CUDA_TOOLKIT:-0}"
MMCV_VERSION="${MMCV_VERSION:-1.6.0}"
TORCH_VERSION="${TORCH_VERSION:-1.12.1+cu113}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.13.1+cu113}"
NUMPY_VERSION="${NUMPY_VERSION:-1.23.5}"
OPENCV_VERSION="${OPENCV_VERSION:-4.11.0.86}"

if [[ ! -d "$SET_DIR" ]]; then
    echo "Missing SET checkout: $SET_DIR" >&2
    echo "Set SET_DIR=/path/to/SET and rerun." >&2
    exit 1
fi

command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }

export MAX_JOBS
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export MAKEFLAGS="-j$MAX_JOBS"
export PIP_DISABLE_PIP_VERSION_CHECK=1

printf 'SET_DIR=%s\nVENV_DIR=%s\nPYTHON_VERSION=%s\nMAX_JOBS=%s\n' \
    "$SET_DIR" "$VENV_DIR" "$PYTHON_VERSION" "$MAX_JOBS"

# Keep this runtime separate from /marimo/mmdet-venv, which is used by MMDetection 3.x.
uv venv "$VENV_DIR" --python "$PYTHON_VERSION" --seed
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

case "$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" in
    3.9|3.10) ;;
    *) echo "SET requires Python 3.9 or 3.10, got $(python --version)" >&2; exit 1 ;;
esac

python -m pip install --upgrade 'pip<25' 'setuptools==65.5.0' 'wheel<1'

# SET's released code uses the MMDetection 2.x API and the Torch 1.12 ABI.
python -m pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Some Marimo images reject the executable-stack flag shipped in the Torch 1.12
# wheel. Clear it before importing torch. This only changes the isolated venv.
python -m pip install 'patchelf==0.19.1.0'
TORCH_CPU_LIB=("$VENV_DIR"/lib/python*/site-packages/torch/lib/libtorch_cpu.so)
if [[ "${#TORCH_CPU_LIB[@]}" -ne 1 ]]; then
    echo "Could not locate exactly one libtorch_cpu.so in $VENV_DIR" >&2
    exit 1
fi
"$VENV_DIR/bin/patchelf" --clear-execstack "${TORCH_CPU_LIB[0]}"

python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
PY

# The prebuilt MMCV wheel is sufficient for SET. Do not install a CUDA toolkit
# into the notebook process unless a source build is explicitly requested.
if [[ "$INSTALL_CUDA_TOOLKIT" == "1" ]]; then
    : "${CUDA_APT_PACKAGE:=cuda-toolkit-13-1}"
    export DEBIAN_FRONTEND=noninteractive
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get install -y --no-install-recommends "$CUDA_APT_PACKAGE"
fi

python -m pip install \
    "mmcv-full==$MMCV_VERSION" \
    -f "https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html"

# Keep old Torch and MMCV compatible with NumPy 1.x. Install SET's legacy
# packages without dependency resolution so timm/kornia cannot replace Torch
# 1.12 with a current Torch 2.x wheel.
python -m pip install --no-deps \
    "numpy==$NUMPY_VERSION" \
    "opencv-python==$OPENCV_VERSION" \
    'matplotlib==3.7.5' \
    'contourpy==1.1.1' \
    'cycler==0.12.1' \
    'fonttools==4.53.1' \
    'kiwisolver==1.4.5' \
    'packaging==24.1' \
    'pillow==10.4.0' \
    'pyparsing==3.1.2' \
    'python-dateutil==2.9.0.post0' \
    'pycocotools==2.0.6' \
    'yapf==0.40.1' \
    'six' \
    'terminaltables' \
    'timm==0.6.13' \
    'kornia'


# Albumentations is optional in SET's pipeline and is not needed by the
# default FCOS configs. Keep upload support separate from Torch resolution.
python -m pip install 'huggingface_hub'

# SET has no native extension of its own. Install it without dependency
# resolution so pip cannot replace the pinned Torch/MMCV stack.
python -m pip install \
    --no-deps \
    --no-build-isolation \
    --editable "$SET_DIR"

# Verify the actual SET public import path and FCOS_set registration.
SET_DIR="$SET_DIR" PYTHONPATH="$SET_DIR${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import os
from mmcv import Config
import mmcv
import torch
import mmdet
from mmdet.models.builder import DETECTORS

assert mmcv.__version__ == '1.6.0', mmcv.__version__
assert torch.__version__.startswith('1.12.1'), torch.__version__
assert 'FCOS_set' in DETECTORS.module_dict
Config.fromfile(os.path.join(os.environ['SET_DIR'], 'configs/aitod/fcos_r50_set.py'))
print('mmcv:', mmcv.__version__)
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('mmdet:', mmdet.__version__)
print('FCOS_set: registered')
print('SET MMDetection 2.x runtime: ready')
PY
