"""Explicit runtime mapping for MMDetection and the external RT-DETR fork.

The project checkout does not ship RT-DETR.  The RT-DETR implementation is pinned
by URL and commit in the launcher manifests and must be supplied as an exact
checkout through ``--rtdetr-root``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

RTDETR_MODEL = "rtdetr_r18"
RTDETR_CONFIG = "configs/rtdetr/rtdetr_r18vd_8xb2-72e_coco.py"
RTDETR_REPOSITORY = "https://github.com/flytocc/rtdetr-mmdet.git"
RTDETR_COMMIT = "66365c1553ffd121ca4ee2be9d091735faf3c182"


def is_rtdetr(model_name: str) -> bool:
    return model_name == RTDETR_MODEL


def runtime_root(project_root: Path, internal_root: Path, model_name: str, rtdetr_root: str) -> Path:
    if not is_rtdetr(model_name):
        return internal_root
    root = Path(rtdetr_root).expanduser()
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve()
    config = root / RTDETR_CONFIG
    train = root / "tools" / "train.py"
    if not config.is_file() or not train.is_file():
        raise FileNotFoundError(
            f"RT-DETR checkout is incomplete: expected {config} and {train}. "
            f"Clone {RTDETR_REPOSITORY} at commit {RTDETR_COMMIT}."
        )
    return root


def config_path(
    project_root: Path,
    internal_root: Path,
    model_name: str,
    internal_configs: dict[str, str],
    rtdetr_root: str,
) -> Path:
    root = runtime_root(project_root, internal_root, model_name, rtdetr_root)
    relative = RTDETR_CONFIG if is_rtdetr(model_name) else internal_configs[model_name]
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical model config: {path}")
    return path


def prepend_runtime_path(runtime: Path) -> None:
    import sys

    value = str(runtime)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def manifest_runtime(model_name: str, runtime: Path) -> dict[str, Any]:
    return {
        "runtime_code_root": str(runtime),
        "rt_detr_repository": RTDETR_REPOSITORY if is_rtdetr(model_name) else None,
        "rt_detr_commit": RTDETR_COMMIT if is_rtdetr(model_name) else None,
    }
