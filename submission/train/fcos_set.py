"""Submission entrypoint for FCOS-SET training."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "run_set_fcos_mmdet3_queue.py"), run_name="__main__")
