"""Submission entrypoint for COCO metric evaluation."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "evaluate_coco_metrics.py"), run_name="__main__")
