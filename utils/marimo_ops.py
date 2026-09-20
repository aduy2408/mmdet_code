#!/usr/bin/env python3
"""Deterministic, dependency-free operations for Marimo training jobs.

This module is intentionally about orchestration and evidence, not model code:
- preflight checks the effective environment and exact Git revision;
- launch starts a detached process with durable PID/log/state files;
- status distinguishes process liveness from observable progress;
- artifacts verifies the minimum local completion contract.

It is safe to import from a runner. It never prints secret environment values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from utils.marimo.contracts import (
    DatasetSpec,
    ExecutionSpec,
    ExperimentContract,
    OutputSpec,
    SourceSpec,
)


DEFAULT_REQUIRED_ARTIFACTS = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
)
COMPLETION_REQUIRED_ARTIFACTS = DEFAULT_REQUIRED_ARTIFACTS + (
    "experiment_manifest.json",
    "upload_complete.json",
)
REQUIRED_METRIC_KEYS = (
    "val/AP50",
    "val/mAP50-95",
    "test/AP50",
    "test/mAP50-95",
)
REQUIRED_CONTRACT_KEYS = (
    "dataset",
    "data_root",
    "dataset_yaml",
    "model_yaml",
    "seed",
    "split_seed",
    "workers",
    "epochs",
    "patience",
    "nms_iou",
    "hf_repo_id",
)
MARIMO_DATASET_ROOTS = {
    "levir": ("LevirShip", "LevirShip/LevirShipData"),
    "levir_ship": ("LevirShip", "LevirShip/LevirShipData"),
    "levirship": ("LevirShip", "LevirShip/LevirShipData"),
    "varroa": ("Varroa",),
    "tinyperson": ("TinyPerson", "TinyPersonData"),
}


class MarimoOpsError(RuntimeError):
    """A preflight, launch, status, or artifact contract failure."""


def resolve_hf_repo_id(repo_id: str | None = None, *, token: str | None = None) -> str:
    """Resolve an artifact repository without falling back to a shared bucket.

    Upload-required experiments must identify their task.  A caller may provide
    an explicit ``repo_id`` or configure ``MARIMO_HF_REPO_ID``/``HF_REPO_ID``.
    Otherwise ``MARIMO_TASK_NAME`` or ``MARIMO_EXPERIMENT_NAME`` is converted
    into a task-specific ``<user>/<task>-runs`` repository name.
    """
    configured = repo_id or os.environ.get("MARIMO_HF_REPO_ID") or os.environ.get("HF_REPO_ID")
    if configured and configured.strip():
        return configured.strip()
    if not token:
        raise MarimoOpsError(
            "HF_TOKEN is required to derive the default artifact repository"
        )
    try:
        from huggingface_hub import HfApi

        identity = HfApi(token=token).whoami()
        username = identity.get("name") or identity.get("user", {}).get("name")
    except Exception as exc:  # pragma: no cover - depends on remote auth/service
        raise MarimoOpsError(f"Unable to resolve the Hugging Face username: {exc}") from exc
    if not username:
        raise MarimoOpsError("Hugging Face identity did not include a username")
    task_name = os.environ.get("MARIMO_TASK_NAME") or os.environ.get("MARIMO_EXPERIMENT_NAME")
    if not task_name or not task_name.strip():
        raise MarimoOpsError(
            "HF repository is not configured. Set MARIMO_HF_REPO_ID/HF_REPO_ID "
            "or provide MARIMO_TASK_NAME for a task-specific repository."
        )
    slug = re.sub(r"[^a-z0-9]+", "-", task_name.lower()).strip("-")
    if not slug:
        raise MarimoOpsError("MARIMO_TASK_NAME must contain at least one alphanumeric character")
    return f"{username}/{slug}-runs"


def ensure_hf_repo(repo_id: str | None = None, *, repo_type: str = "dataset") -> str:
    """Create the default artifact repository when it does not already exist."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise MarimoOpsError("HF_TOKEN is required when uploads are enabled")
    resolved = resolve_hf_repo_id(repo_id, token=token)
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).create_repo(repo_id=resolved, repo_type=repo_type, exist_ok=True)
    except Exception as exc:  # pragma: no cover - depends on remote auth/service
        raise MarimoOpsError(f"Unable to create or access Hugging Face repo {resolved}: {exc}") from exc
    return resolved


def require_training_context(*, hf_repo_id: str | None = None) -> None:
    """Require the shared Marimo launch context for upload-required runners."""
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise MarimoOpsError(
            "Marimo training must be launched with `python -m utils.marimo_ops launch`"
        )
    if not os.environ.get("HF_TOKEN"):
        raise MarimoOpsError("HF_TOKEN is required for upload-required Marimo training")
    configured_repo = os.environ.get("MARIMO_HF_REPO_ID") or os.environ.get("HF_REPO_ID")
    if hf_repo_id and configured_repo and configured_repo.strip() != hf_repo_id.strip():
        raise MarimoOpsError(
            f"HF repository mismatch: configured {configured_repo.strip()}, got {hf_repo_id.strip()}"
        )
    expected_repo = resolve_hf_repo_id(hf_repo_id, token=os.environ.get("HF_TOKEN"))
    if expected_repo and hf_repo_id and expected_repo != hf_repo_id:
        raise MarimoOpsError(
            f"HF repository mismatch: expected {expected_repo}, got {hf_repo_id}"
        )


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise MarimoOpsError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MarimoOpsError(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MarimoOpsError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_contract(run_dir: Path, contract: Mapping[str, object]) -> dict[str, object]:
    """Write the immutable pre-launch contract used by completion checks."""
    missing = [key for key in REQUIRED_CONTRACT_KEYS if key not in contract]
    if missing:
        raise MarimoOpsError("Run contract is missing required fields: " + ", ".join(missing))
    path = run_dir / "run_contract.json"
    if path.exists():
        raise MarimoOpsError(f"Refusing to overwrite existing run contract: {path}")
    payload = {"created_at": now_utc(), **dict(contract)}
    write_json(path, payload)
    return payload


def validate_dataset(data_root: Path, dataset_yaml: Path) -> dict[str, str]:
    """Validate the declared dataset root, YAML, and train/val/test directories."""
    if not data_root.is_dir():
        raise MarimoOpsError(f"Dataset root does not exist: {data_root}")
    if not dataset_yaml.is_file():
        raise MarimoOpsError(f"Dataset YAML does not exist: {dataset_yaml}")
    text = dataset_yaml.read_text()
    values: dict[str, str] = {}
    for key in ("path", "train", "val", "test"):
        match = re.search(rf"^\s*{key}\s*:\s*['\"]?([^#\n'\"]+)['\"]?\s*$", text, re.MULTILINE)
        if match:
            values[key] = match.group(1).strip()
    missing = [key for key in ("train", "val", "test") if key not in values]
    if missing:
        raise MarimoOpsError(f"Dataset YAML is missing split paths: {', '.join(missing)}")
    base = Path(values.get("path", str(data_root)))
    if not base.is_absolute():
        base = (dataset_yaml.parent / base).resolve()
    for split in ("train", "val", "test"):
        split_path = Path(values[split])
        if not split_path.is_absolute():
            split_path = base / split_path
        if not split_path.exists():
            raise MarimoOpsError(f"Dataset {split} path does not exist: {split_path}")
    return {"data_root": str(data_root.resolve()), "dataset_yaml": str(dataset_yaml.resolve())}


def resolve_marimo_dataset_root(dataset: str, marimo_root: Path = Path("/marimo")) -> Path:
    """Resolve only the repository's canonical persistent Marimo dataset mounts."""
    candidates = MARIMO_DATASET_ROOTS.get(dataset.strip().lower())
    if not candidates:
        raise MarimoOpsError(
            f"Unknown dataset {dataset!r}; expected one of: {', '.join(sorted(MARIMO_DATASET_ROOTS))}"
        )
    for relative in candidates:
        candidate = marimo_root / relative
        if candidate.is_dir():
            return candidate.resolve()
    expected = ", ".join(str(marimo_root / item) for item in candidates)
    raise MarimoOpsError(f"Dataset {dataset!r} is not mounted at a canonical path: {expected}")


def run_checked(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            list(command), cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        output = (exc.output or "").strip()
        raise MarimoOpsError(
            f"Command failed ({exc.returncode}): {shlex.join(command)}\n{output}"
        ) from exc


def git_sha(repo: Path) -> str:
    return run_checked(["git", "rev-parse", "HEAD"], cwd=repo)


def git_dirty(repo: Path) -> str:
    return run_checked(["git", "status", "--porcelain"], cwd=repo)


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # ``kill -0`` also succeeds for a zombie. Treat a zombie as finished so
    # stale PID files cannot block a restart forever.
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text().split()
    except OSError:
        return True
    return len(fields) < 3 or fields[2] != "Z"


def process_command(pid: int) -> str:
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        raw = proc_cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""
    return raw.strip()


def command_option_values(command: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    for index, value in enumerate(command):
        if value == option and index + 1 < len(command):
            values.append(command[index + 1])
        elif value.startswith(option + "="):
            values.append(value.split("=", 1)[1])
    return values


def _path_value(value: str) -> str:
    """Accept dataset aliases such as ``varroa=/marimo/Varroa``."""
    return value.split("=", 1)[1] if "=" in value else value


def _same_contract_value(key: str, expected: object, actual: str) -> bool:
    if key in {"data_root", "model_yaml"}:
        return os.path.normpath(_path_value(str(expected))) == os.path.normpath(_path_value(actual))
    if key in {"epochs", "patience", "workers", "seed", "split_seed"}:
        try:
            return int(expected) == int(actual)
        except (TypeError, ValueError):
            return False
    if key == "nms_iou":
        try:
            return float(expected) == float(actual)
        except (TypeError, ValueError):
            return False
    return str(expected) == actual


def validate_command_contract(run_dir: Path, command: Sequence[str]) -> None:
    """Reject launch commands that omit or disagree with contract settings."""
    contract_path = run_dir / "run_contract.json"
    if not contract_path.is_file():
        raise MarimoOpsError(f"Missing mandatory run contract: {contract_path}")
    contract = read_json(contract_path)
    missing = [key for key in REQUIRED_CONTRACT_KEYS if key not in contract]
    if missing:
        raise MarimoOpsError("Run contract is missing required fields: " + ", ".join(missing))
    options = {
        "epochs": "--epochs",
        "patience": "--patience",
        "workers": "--workers",
        "seed": "--seed",
        "split_seed": "--split-seed",
        "hf_repo_id": "--hf-repo-id",
        "model_yaml": "--model-yaml",
        "data_root": "--data-root",
    }
    mismatches: list[str] = []
    for key, option in options.items():
        # A single supervisor may intentionally own a deterministic shard of
        # several datasets/models/seeds. In that case the contract records a
        # matrix wildcard and the runner's own manifest carries per-run values.
        if contract.get(key) == "matrix":
            continue
        values = command_option_values(command, option)
        if not values:
            mismatches.append(f"{option}: missing (contract={contract[key]!r})")
        elif not all(_same_contract_value(key, contract[key], value) for value in values):
            mismatches.append(f"{option}: contract={contract[key]!r}, command={values!r}")
    nms_values = command_option_values(command, "--nms-iou")
    if nms_values and not all(_same_contract_value("nms_iou", contract["nms_iou"], value) for value in nms_values):
        mismatches.append(f"--nms-iou: contract={contract['nms_iou']!r}, command={nms_values!r}")
    if mismatches:
        raise MarimoOpsError("Launch command disagrees with run contract:\n- " + "\n- ".join(mismatches))


def artifact_root_from_state(run_dir: Path, state: Mapping[str, object] | None) -> Path:
    """Find the project output root when the wrapper and artifact directories differ."""
    if state and state.get("artifact_root"):
        root = Path(str(state["artifact_root"]))
        if not root.is_absolute() and state.get("cwd"):
            root = Path(str(state["cwd"])) / root
        return root.resolve()
    command = state.get("command") if state else None
    if isinstance(command, list):
        for option in ("--project", "--output-dir", "--project-dir"):
            values = command_option_values([str(item) for item in command], option)
            if values:
                output = Path(_path_value(values[-1]))
                if not output.is_absolute() and state and state.get("cwd"):
                    output = Path(str(state["cwd"])) / output
                return output.resolve()
    return run_dir.resolve()


def artifact_root_and_state(run_dir: Path) -> tuple[Path, dict[str, object] | None]:
    state_path = run_dir / "state.json"
    state = read_json(state_path) if state_path.is_file() else None
    return artifact_root_from_state(run_dir, state), state


def newest_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_mtime
    files = [p for p in path.rglob("*") if p.is_file()]
    return max((p.stat().st_mtime for p in files), default=None)


def require_files(root: Path, relative_paths: Iterable[str]) -> list[str]:
    paths = list(relative_paths)
    missing = [relative for relative in paths if not (root / relative).is_file()]
    if missing:
        raise MarimoOpsError(
            "Missing required artifacts:\n" + "\n".join(f"- {item}" for item in missing)
        )
    return paths


def preflight(
    *,
    repo: Path,
    expected_sha: str | None = None,
    python: str | None = None,
    required_paths: Iterable[str] = (),
    epochs: int | None = None,
    patience: int | None = None,
    upload_required: bool = False,
    hf_repo_id: str | None = None,
    data_root: Path | None = None,
    dataset_yaml: Path | None = None,
    contract_json: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Run fail-closed checks before any expensive remote job."""
    contract: dict[str, object] | None = None
    if contract_json is not None:
        contract = read_json(contract_json)
        missing = [key for key in REQUIRED_CONTRACT_KEYS if key not in contract]
        if missing:
            raise MarimoOpsError("Run contract is missing required fields: " + ", ".join(missing))
        declared = {
            "epochs": contract["epochs"],
            "patience": contract["patience"],
            "hf_repo_id": contract["hf_repo_id"],
            "data_root": None if contract["data_root"] == "matrix" else Path(_path_value(str(contract["data_root"]))),
            "dataset_yaml": None if contract["dataset_yaml"] == "matrix" else Path(_path_value(str(contract["dataset_yaml"]))),
        }
        for key, actual in (("epochs", epochs), ("patience", patience), ("hf_repo_id", hf_repo_id),
                            ("data_root", data_root), ("dataset_yaml", dataset_yaml)):
            if declared[key] is None:
                if actual is None:
                    raise MarimoOpsError(f"Preflight must provide matrix contract value for {key}")
                continue
            if actual is not None and not _same_contract_value(key, declared[key], str(actual)):
                raise MarimoOpsError(f"Preflight disagrees with run contract for {key}: contract={declared[key]!r}, got={actual!r}")
        epochs = int(declared["epochs"])
        patience = int(declared["patience"])
        hf_repo_id = str(declared["hf_repo_id"])
        if declared["data_root"] is not None:
            data_root = declared["data_root"]
        if declared["dataset_yaml"] is not None:
            dataset_yaml = declared["dataset_yaml"]
    if not repo.is_dir():
        raise MarimoOpsError(f"Repository does not exist: {repo}")
    actual_sha = git_sha(repo)
    dirty = git_dirty(repo)
    if expected_sha and actual_sha != expected_sha:
        raise MarimoOpsError(f"Wrong Git SHA: expected {expected_sha}, got {actual_sha}")
    if dirty and not allow_dirty:
        raise MarimoOpsError(f"Remote worktree is dirty:\n{dirty}")
    if python:
        py = Path(python)
        if not py.is_file() or not os.access(py, os.X_OK):
            raise MarimoOpsError(f"Python executable is not executable: {python}")
        executable = run_checked([python, "-c", "import sys; print(sys.executable)"])
    else:
        executable = sys.executable
    if upload_required:
        hf_repo_id = ensure_hf_repo(hf_repo_id)
    dataset_info = {}
    if data_root is not None or dataset_yaml is not None:
        if data_root is None or dataset_yaml is None:
            raise MarimoOpsError("Both data_root and dataset_yaml are required for dataset validation")
        dataset_info = validate_dataset(data_root, dataset_yaml)
    if epochs is not None and epochs <= 0:
        raise MarimoOpsError(f"epochs must be positive, got {epochs}")
    if patience is not None and patience < 0:
        raise MarimoOpsError(f"patience must be non-negative, got {patience}")
    required = require_files(repo, required_paths)
    result = {
        "checked_at": now_utc(),
        "repo": str(repo),
        "git_sha": actual_sha,
        "worktree_dirty": bool(dirty),
        "python": executable,
        "epochs": epochs,
        "patience": patience,
        "upload_required": upload_required,
        "hf_repo_id": hf_repo_id,
        **dataset_info,
        "required_paths": required,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@dataclass
class LaunchResult:
    pid: int
    command: list[str]
    cwd: str
    log_path: str
    pid_path: str
    state_path: str
    started_at: str


def launch_detached(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    pid_path: Path,
    state_path: Path,
    artifact_root: Path | None = None,
    contract_path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> LaunchResult:
    """Launch one durable process and record enough evidence to inspect it later."""
    if not command:
        raise MarimoOpsError("Cannot launch an empty command")
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
        except ValueError:
            old_pid = -1
        if is_pid_alive(old_pid):
            raise MarimoOpsError(f"Refusing duplicate launch: PID {old_pid} is alive")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = now_utc()
    log_file = open(log_path, "ab", buffering=0)
    child_env = dict(os.environ)
    if env is not None:
        child_env.update(env)
    proc = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    # The child owns the duplicated descriptor after Popen returns. Keeping the
    # parent's descriptor open causes warnings and delays EOF on short jobs.
    log_file.close()
    pid_path.write_text(f"{proc.pid}\n")
    resolved_artifact_root = None
    if artifact_root is not None:
        resolved_artifact_root = artifact_root if artifact_root.is_absolute() else cwd / artifact_root
    effective_contract_path = contract_path or state_path.parent / "run_contract.json"
    state = {
        "status": "running",
        "pid": proc.pid,
        "command": list(command),
        "cwd": str(cwd),
        "log_path": str(log_path),
        "pid_path": str(pid_path),
        "state_path": str(state_path),
        "artifact_root": str(resolved_artifact_root.resolve()) if resolved_artifact_root else None,
        "contract_sha256": file_sha256(effective_contract_path)
        if effective_contract_path.is_file()
        else None,
        "started_at": started_at,
    }
    write_json(state_path, state)
    # Reap after state exists so short-lived jobs still record their exit code.
    def reap() -> None:
        returncode = proc.wait()
        finished = read_json(state_path) if state_path.is_file() else state
        finished.update({"status": "exited", "returncode": returncode, "finished_at": now_utc()})
        write_json(state_path, finished)

    threading.Thread(target=reap, name=f"marimo-reaper-{proc.pid}", daemon=True).start()
    print(json.dumps(state, indent=2, sort_keys=True))
    return LaunchResult(
        pid=proc.pid,
        command=list(command),
        cwd=str(cwd),
        log_path=str(log_path),
        pid_path=str(pid_path),
        state_path=str(state_path),
        started_at=started_at,
    )


def status(
    run_dir: Path,
    *,
    pid_file: str = "train.pid",
    log_file: str = "train.log",
    state_file: str = "state.json",
    emit: bool = True,
) -> dict[str, object]:
    """Report liveness and progress separately. No claim of completion is inferred."""
    pid_path = run_dir / pid_file
    log_path = run_dir / log_file
    pid: int | None = None
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pass
    alive = is_pid_alive(pid) if pid is not None else False
    state_path = run_dir / state_file
    state = read_json(state_path) if state_path.is_file() else None
    artifact_root = artifact_root_from_state(run_dir, state)
    actual_command = process_command(pid) if alive and pid else ""
    expected_command = state.get("command") if isinstance(state, dict) else None
    process_identity = "unknown"
    if alive and isinstance(expected_command, list):
        try:
            process_identity = "matched" if shlex.split(actual_command) == [str(item) for item in expected_command] else "mismatch"
        except ValueError:
            process_identity = "mismatch"
    elif alive:
        process_identity = "unverified"
    active = alive and process_identity == "matched"
    artifact_paths = [artifact_root / item for item in DEFAULT_REQUIRED_ARTIFACTS]
    latest = max(
        (value for value in [newest_mtime(run_dir), *(newest_mtime(path) for path in artifact_paths)] if value is not None),
        default=None,
    )
    required_present = all(path.is_file() for path in artifact_paths)
    upload_marker_path = artifact_root / "upload_complete.json"
    upload_marker_present = upload_marker_path.is_file()
    upload_verified = False
    if upload_marker_present:
        try:
            marker = read_json(upload_marker_path)
            verified_files = marker.get("verified")
            upload_verified = bool(
                marker.get("repo_id")
                and marker.get("remote_prefix")
                and isinstance(verified_files, list)
                and all(item in verified_files for item in COMPLETION_REQUIRED_ARTIFACTS)
            )
            contract_path = run_dir / "run_contract.json"
            if upload_verified and contract_path.is_file():
                contract = read_json(contract_path)
                upload_verified = marker["repo_id"] == contract.get("hf_repo_id")
            else:
                upload_verified = False
        except MarimoOpsError:
            upload_verified = False
    if active:
        observed_status = "running"
    elif alive and process_identity == "mismatch":
        observed_status = "stale_pid_or_pid_reuse"
    elif upload_verified and required_present:
        observed_status = "artifacts_present"
    elif latest is not None or state is not None:
        observed_status = "not_running_unverified"
    else:
        observed_status = "not_started_or_unknown"
    result = {
        "run_dir": str(run_dir),
        "artifact_root": str(artifact_root),
        "pid": pid,
        "process_alive": alive,
        "process_identity": process_identity,
        "process_active": active,
        "observed_status": observed_status,
        "process_command": actual_command,
        "latest_artifact_mtime": latest,
        "log_exists": log_path.is_file(),
        "log_mtime": log_path.stat().st_mtime if log_path.is_file() else None,
        "state": state,
        "upload_marker_present": upload_marker_present,
        "upload_verified": upload_verified,
        "required_artifacts": {
            item: (artifact_root / item).is_file() for item in DEFAULT_REQUIRED_ARTIFACTS
        },
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        contract = read_json(contract_path)
        result["dataset"] = contract.get("dataset")
        result["data_root"] = contract.get("data_root")
        result["dataset_yaml"] = contract.get("dataset_yaml")
    if active:
        result["continuation_state"] = "running"
    elif (artifact_root / "evaluation_metrics.json").is_file() and not (artifact_root / "upload_complete.json").is_file():
        result["continuation_state"] = "evaluation_or_upload_pending"
    elif (artifact_root / "weights/last.pt").is_file() and not (artifact_root / "evaluation_metrics.json").is_file():
        result["continuation_state"] = "checkpoint_present_evaluation_pending"
    elif not (artifact_root / "weights/last.pt").is_file():
        result["continuation_state"] = "no_checkpoint_unverified"
    else:
        result["continuation_state"] = "not_running_unverified"
    command_for_resume = state.get("command") if isinstance(state, dict) else []
    resume_values = command_option_values(command_for_resume, "--resume") if isinstance(command_for_resume, list) else []
    if resume_values:
        state_log_path = Path(str(state.get("log_path"))) if isinstance(state, dict) and state.get("log_path") else log_path
        if not state_log_path.is_absolute():
            state_log_path = run_dir / state_log_path
        log_text = state_log_path.read_text(errors="replace") if state_log_path.is_file() else ""
        result["resume_evidence"] = "log_hint" if re.search(
            r"(?:resum(?:ed|ing)|loaded checkpoint|from epoch)", log_text, re.I
        ) else "pending"
        if not alive and result["resume_evidence"] != "log_hint":
            result["continuation_state"] = "resume_blocked"
    if emit:
        print(json.dumps(result, indent=2, sort_keys=True))
    return result


def artifacts(
    run_dir: Path,
    required: Iterable[str] = DEFAULT_REQUIRED_ARTIFACTS,
    *,
    state_file: str = "state.json",
) -> dict[str, object]:
    state_path = run_dir / state_file
    state = read_json(state_path) if state_path.is_file() else None
    artifact_root = artifact_root_from_state(run_dir, state)
    present = require_files(artifact_root, required)
    result = {"run_dir": str(run_dir), "artifact_root": str(artifact_root), "required_artifacts": present}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def complete_verified(
    run_dir: Path,
    *,
    pid_file: str = "train.pid",
    log_file: str = "train.log",
    state_file: str = "state.json",
) -> dict[str, object]:
    """Fail closed unless local metrics, provenance, and upload evidence are complete."""
    report = status(
        run_dir,
        pid_file=pid_file,
        log_file=log_file,
        state_file=state_file,
        emit=False,
    )
    if report["process_active"]:
        raise MarimoOpsError("Run is still active; completion cannot be verified")
    if report["process_alive"]:
        raise MarimoOpsError(
            f"PID is alive but process identity is {report['process_identity']}; completion is blocked"
        )
    state = report.get("state")
    if not isinstance(state, dict):
        raise MarimoOpsError("Missing launch state; completion cannot be verified")
    if state.get("status") != "exited":
        raise MarimoOpsError("Run has no recorded clean process exit")
    if state.get("returncode") != 0:
        raise MarimoOpsError(f"Training exited with return code {state.get('returncode')!r}")
    artifact_root = Path(str(report["artifact_root"]))
    require_files(artifact_root, COMPLETION_REQUIRED_ARTIFACTS)
    contract = read_json(run_dir / "run_contract.json")
    missing_contract = [key for key in REQUIRED_CONTRACT_KEYS if key not in contract]
    if missing_contract:
        raise MarimoOpsError("Run contract is missing required fields: " + ", ".join(missing_contract))
    expected_hash = state.get("contract_sha256")
    if not expected_hash:
        raise MarimoOpsError("Launch state lacks the run contract hash")
    if file_sha256(run_dir / "run_contract.json") != expected_hash:
        raise MarimoOpsError("Run contract changed after launch")
    metrics = read_json(artifact_root / "evaluation_metrics.json")
    missing_metrics = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if missing_metrics:
        raise MarimoOpsError("Evaluation is missing split-qualified metrics: " + ", ".join(missing_metrics))
    for key in REQUIRED_METRIC_KEYS:
        try:
            value = float(metrics[key])
        except (TypeError, ValueError) as exc:
            raise MarimoOpsError(f"Metric {key} is not numeric") from exc
        if not (value == value and abs(value) != float("inf")):
            raise MarimoOpsError(f"Metric {key} is not finite")
    marker = read_json(artifact_root / "upload_complete.json")
    verified_files = marker.get("verified")
    if not marker.get("repo_id") or not marker.get("remote_prefix"):
        raise MarimoOpsError("Upload marker lacks remote verification evidence")
    if not isinstance(verified_files, list) or not all(isinstance(item, str) for item in verified_files):
        raise MarimoOpsError("Upload marker verified field must be a list of remote files")
    missing_remote_evidence = [item for item in COMPLETION_REQUIRED_ARTIFACTS if item not in verified_files]
    if missing_remote_evidence:
        raise MarimoOpsError(
            "Upload marker lacks verification for: " + ", ".join(missing_remote_evidence)
        )
    if marker["repo_id"] != contract["hf_repo_id"]:
        raise MarimoOpsError(
            f"Upload repository mismatch: contract={contract['hf_repo_id']!r}, marker={marker['repo_id']!r}"
        )
    result = {
        "run_dir": str(run_dir),
        "artifact_root": str(artifact_root),
        "status": "complete_verified",
        "required_artifacts": list(COMPLETION_REQUIRED_ARTIFACTS),
        "metrics": {key: float(metrics[key]) for key in REQUIRED_METRIC_KEYS},
        "hf_repo_id": marker["repo_id"],
        "remote_prefix": marker["remote_prefix"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Marimo training evidence helper")
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--expected-sha")
    p.add_argument("--python")
    p.add_argument("--required-path", action="append", default=[])
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--upload-required", action="store_true")
    p.add_argument("--hf-repo-id")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--dataset-yaml", type=Path)
    p.add_argument("--contract", dest="contract_json", type=Path)
    p.add_argument("--allow-dirty", action="store_true")

    p = sub.add_parser("status")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--pid-file", default="train.pid")
    p.add_argument("--log-file", default="train.log")
    p.add_argument("--state-file", default="state.json")

    p = sub.add_parser("artifacts")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--required-path", action="append", default=list(DEFAULT_REQUIRED_ARTIFACTS))
    p.add_argument("--state-file", default="state.json")

    p = sub.add_parser("contract")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--contract-json", type=Path, required=True)

    p = sub.add_parser("complete_verified")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--pid-file", default="train.pid")
    p.add_argument("--log-file", default="train.log")
    p.add_argument("--state-file", default="state.json")

    p = sub.add_parser("launch")
    p.add_argument("--cwd", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--log-file", default="train.log")
    p.add_argument("--pid-file", default="train.pid")
    p.add_argument("--state-file", default="state.json")
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "preflight":
            preflight(
                repo=args.repo,
                expected_sha=args.expected_sha,
                python=args.python,
                required_paths=args.required_path,
                epochs=args.epochs,
                patience=args.patience,
                upload_required=args.upload_required,
                hf_repo_id=args.hf_repo_id,
                data_root=args.data_root,
                dataset_yaml=args.dataset_yaml,
                contract_json=args.contract_json,
                allow_dirty=args.allow_dirty,
            )
        elif args.action == "status":
            status(args.run_dir, pid_file=args.pid_file, log_file=args.log_file, state_file=args.state_file)
        elif args.action == "artifacts":
            artifacts(args.run_dir, args.required_path, state_file=args.state_file)
        elif args.action == "contract":
            contract = read_json(args.contract_json)
            payload = write_run_contract(args.run_dir, contract)
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.action == "complete_verified":
            complete_verified(args.run_dir, pid_file=args.pid_file, log_file=args.log_file, state_file=args.state_file)
        elif args.action == "launch":
            command = list(args.command)
            if command and command[0] == "--":
                command = command[1:]
            validate_command_contract(args.run_dir, command)
            launch_env = os.environ.copy()
            launch_env["MARIMO_TRAIN_WORKFLOW"] = "1"
            if "--hf-repo-id" in command:
                index = command.index("--hf-repo-id")
                if index + 1 < len(command):
                    launch_env["MARIMO_HF_REPO_ID"] = command[index + 1]
            launch_detached(
                command,
                cwd=args.cwd,
                log_path=args.run_dir / args.log_file,
                pid_path=args.run_dir / args.pid_file,
                state_path=args.run_dir / args.state_file,
                artifact_root=args.artifact_root,
                contract_path=args.run_dir / "run_contract.json",
                env=launch_env,
            )
        return 0
    except MarimoOpsError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
