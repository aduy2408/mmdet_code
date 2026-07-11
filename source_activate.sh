#!/usr/bin/env bash

VENV_DIR="${VENV_DIR:-/tmp/mmdet-venv}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  FOUND_NVCC="$(find /usr/local -name nvcc 2>/dev/null | head -n 1)"
  if [ -n "$FOUND_NVCC" ]; then
    export CUDA_HOME="$(dirname "$(dirname "$FOUND_NVCC")")"
  fi
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "VENV=$VENV_DIR"
echo "CUDA_HOME=$CUDA_HOME"
