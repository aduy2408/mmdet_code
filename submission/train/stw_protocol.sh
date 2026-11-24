#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${PYTHON:-/marimo/mmdet-venv/bin/python}" "$ROOT/run_set_stw_protocol_queue.py" "$@"
