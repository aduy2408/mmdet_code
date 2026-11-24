"""Submission entrypoint for MMDetection baseline training."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "run_two_server_baselines.py"), run_name="__main__")
