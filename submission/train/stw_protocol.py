"""Submission entrypoint for the STW-matched FCOS protocol."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "run_set_stw_protocol_queue.py"), run_name="__main__")
