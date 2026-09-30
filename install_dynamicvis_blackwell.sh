#!/usr/bin/env bash
set -euo pipefail

# Rebuild the DynamicVis environment used on NVIDIA Blackwell Marimo hosts.
# Keep compilation deliberately conservative. This script does not train.

VENV_DIR="${VENV_DIR:-/marimo/dynamicvis-blackwell-venv}"
DYNAMICVIS_DIR="${DYNAMICVIS_DIR:-/marimo/DynamicVis}"
MMDET_DIR="${MMDET_DIR:-/marimo/mmdet_code}"
CUDA_HOME_DEFAULT="${CUDA_HOME:-/usr/local/cuda-13.1}"
DYNAMICVIS_REF="${DYNAMICVIS_REF:-265c414f328f410e8c16f9e31f03ff2a6ecc03b0}"
CAUSAL_REF="${CAUSAL_REF:-82867a9d2e6907cc0f637ac6aff318f696838548}"
MAMBA_REF="${MAMBA_REF:-95d8aba8a8c75aedcaa6143713b11e745e7cd0d9}"
MMCV_WHEEL="${MMCV_WHEEL:-${MMDET_DIR}/mmcv-2.1.0-cp311-cp311-linux_x86_64.whl}"
MAX_JOBS="${MAX_JOBS:-2}"

export MAX_JOBS
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export MAKEFLAGS="${MAKEFLAGS:--j2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CUDA_HOME="${CUDA_HOME_DEFAULT}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required" >&2
  exit 1
fi

if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  apt-get update
  apt-get install -y wget ca-certificates git build-essential
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update
  apt-get install -y cuda-toolkit-13-1
fi

if [ ! -d "${DYNAMICVIS_DIR}/.git" ]; then
  git clone https://github.com/KyanChen/DynamicVis.git "${DYNAMICVIS_DIR}"
fi
git -C "${DYNAMICVIS_DIR}" fetch --depth 1 origin "${DYNAMICVIS_REF}"
git -C "${DYNAMICVIS_DIR}" checkout --detach "${DYNAMICVIS_REF}"

uv venv "${VENV_DIR}" --python 3.11 --seed
PYTHON="${VENV_DIR}/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
"${PYTHON}" -m pip install mmengine==0.10.7 numpy==1.26.4 opencv-python==4.11.0.86
"${PYTHON}" -m pip install transformers==4.50.0 model-index pyparsing==3.2.5 protobuf opentelemetry-api googleapis-common-protos
"${PYTHON}" -m pip install huggingface_hub einops mambapy ipdb braceexpand mat4py pycocotools shapely ftfy scipy terminaltables wandb prettytable torchmetrics importlib_metadata

if [ ! -f "${MMCV_WHEEL}" ]; then
  echo "ERROR: full MMCV wheel not found: ${MMCV_WHEEL}" >&2
  echo "Provide MMCV_WHEEL=/path/to/mmcv-2.1.0-cp311-cp311-linux_x86_64.whl" >&2
  exit 1
fi
"${PYTHON}" -m pip install --force-reinstall --no-deps "${MMCV_WHEEL}"

clone_or_reset() {
  local url="$1"
  local ref="$2"
  local dest="$3"
  if [ ! -d "${dest}/.git" ]; then
    git clone "$url" "${dest}"
  fi
  git -C "${dest}" fetch --depth 1 origin "$ref"
  git -C "${dest}" checkout --detach "$ref"
}

clone_or_reset https://github.com/Dao-AILab/causal-conv1d.git "${CAUSAL_REF}" /marimo/causal-conv1d-src
clone_or_reset https://github.com/state-spaces/mamba.git "${MAMBA_REF}" /marimo/mamba-src

"${PYTHON}" - <<'PY'
from pathlib import Path

causal = Path('/marimo/causal-conv1d-src/setup.py')
text = causal.read_text()
old = '''        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_53,code=sm_53")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_62,code=sm_62")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_70,code=sm_70")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_72,code=sm_72")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")'''
new = '''        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_120,code=sm_120")'''
if old not in text:
    raise SystemExit('causal-conv1d architecture block not found')
causal.write_text(text.replace(old, new, 1))

mamba = Path('/marimo/mamba-src/setup.py')
text = mamba.read_text()
start = text.index('        cc_flag.append("-gencode")', text.index('else:\n        check_if_cuda_home_none'))
end = text.index('\n\n\n    # HACK', start)
text = text[:start] + new + text[end:]
mamba.write_text(text)

reverse_scan = Path('/marimo/mamba-src/csrc/selective_scan/reverse_scan.cuh')
text = reverse_scan.read_text()
text = text.replace('cub::LaneId()', '(threadIdx.x & 31)')
text = text.replace('cub::CTA_SYNC()', '__syncthreads()')
reverse_scan.write_text(text)
PY

CAUSAL_CONV1D_FORCE_BUILD=TRUE "${PYTHON}" -m pip install --no-build-isolation --no-cache-dir /marimo/causal-conv1d-src
MAMBA_FORCE_BUILD=TRUE "${PYTHON}" -m pip install --no-build-isolation --no-cache-dir /marimo/mamba-src

mkdir -p "${DYNAMICVIS_DIR}/checkpoints"
wget -q -O "${DYNAMICVIS_DIR}/checkpoints/pretrain_dynamicvis_b_bf16_mamba_epoch_200.pth" \
  'https://huggingface.co/KyanChen/DynamicVis/resolve/main/pretrain_dynamicvis_b_bf16_mamba_epoch_200.pth?download=true'

PYTHONPATH="${DYNAMICVIS_DIR}:${MMDET_DIR}/mmdetection:${MMDET_DIR}" \
"${PYTHON}" - <<'PY'
import torch
import mmcv
from mmcv.ops import roi_align
import causal_conv1d
import mamba_ssm
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (12, 0)
assert hasattr(roi_align, '__call__')
assert causal_conv1d.__version__ == '1.5.0.post8'
assert mamba_ssm.__version__ == '2.2.4'

model = Mamba(d_model=96, d_state=16, d_conv=4, expand=2).cuda().half().eval()
x = torch.randn(1, 64, 96, device='cuda', dtype=torch.float16)
with torch.no_grad():
    y = model(x)
assert tuple(y.shape) == (1, 64, 96)
print('DYNAMICVIS_BLACKWELL_READY')
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('mmcv', mmcv.__version__)
print('gpu', torch.cuda.get_device_name(0))
print('mamba_output', tuple(y.shape), y.dtype)
PY

"${PYTHON}" -m pip check
printf 'ready\n' > "${VENV_DIR}/DYNAMICVIS_READY"
