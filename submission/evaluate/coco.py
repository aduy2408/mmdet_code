"""Submission entrypoint for COCO metric evaluation."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "evaluate_coco_metrics.py"), run_name="__main__")
