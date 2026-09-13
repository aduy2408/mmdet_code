"""Marimo-safe process, manifest, artifact, and HF verification helpers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence


FORBIDDEN_FLAGS = {"--no-hf-upload", "--no-upload", "--no-hf-upload"}


def preflight(
    command: Sequence[str],
    *,
    manifest: dict[str, Any],
    expected_python: str,
    dataset_root: str,
    split_seed: int,
    training_seed: int,
) -> None:
    """Reject commands that contradict the recorded experiment manifest."""
    command = list(command)
    if not command or command[0] != expected_python:
        raise ValueError(f"wrong Python: expected {expected_python}, got {command[:1]}")
    if any(flag in command for flag in FORBIDDEN_FLAGS):
        raise ValueError("upload is mandatory; forbidden upload-suppression flag present")
    if manifest.get("python_executable") != expected_python:
        raise ValueError("manifest Python does not match the launch command")
    if manifest.get("dataset_root") != str(Path(dataset_root).resolve()):
        raise ValueError("manifest dataset root does not match launch intent")
    if manifest.get("split_seed") != split_seed:
        raise ValueError("manifest split seed does not match launch intent")
    if manifest.get("training_seed") != training_seed:
        raise ValueError("manifest training seed does not match launch intent")
    if manifest.get("upload_required") is not True:
        raise ValueError("PH-DETR runs must require Hugging Face upload")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is absent from the live Marimo kernel")


def launch_detached(
    command: Sequence[str], *, cwd: str | Path, log_path: str | Path, env: dict[str, str]
) -> subprocess.Popen[str]:
    """Launch exactly one detached worker and return its process handle."""
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", buffering=1, encoding="utf-8")
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process._marimo_log_handle = handle  # type: ignore[attr-defined]
    return process


def status(process: subprocess.Popen[str]) -> dict[str, Any]:
    """Return non-secret process status for a Marimo status cell."""
    return {"pid": process.pid, "returncode": process.poll()}


def verify_artifacts(run_dir: str | Path, required: Sequence[str]) -> list[str]:
    root = Path(run_dir)
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing PH-DETR artifacts: {missing}")
    return [str(root / name) for name in required]


def verify_hf_remote(repo_id: str, remote_prefix: str, *, token: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="dataset")
    matched = [path for path in files if path.startswith(remote_prefix + "/")]
    if not matched:
        raise FileNotFoundError(f"HF remote prefix not found: {repo_id}:{remote_prefix}")
    return matched


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
