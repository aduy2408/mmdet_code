#!/usr/bin/env python3
"""Prepare the pinned external RT-DETR MMDetection checkout."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rtdetr_runtime import RTDETR_COMMIT, RTDETR_REPOSITORY, RTDETR_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="third_party/rtdetr-mmdet")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    if root.exists() and not (root / ".git").exists():
        raise FileExistsError(f"Refusing to use non-git RT-DETR path: {root}")
    if not root.exists():
        command = ["git", "clone", RTDETR_REPOSITORY, str(root)]
        print("SETUP", " ".join(command))
        if not args.dry_run:
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
    if root.exists():
        current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if current != RTDETR_COMMIT:
            command = ["git", "fetch", "--depth", "1", "origin", RTDETR_COMMIT]
            print("SETUP", " ".join(command))
            if not args.dry_run:
                subprocess.run(command, cwd=root, check=True)
                subprocess.run(["git", "checkout", "--detach", RTDETR_COMMIT], cwd=root, check=True)
                current = RTDETR_COMMIT
        print(f"RT-DETR checkout: {root}")
        print(f"RT-DETR commit: {current}")
        print(f"RT-DETR config: {root / RTDETR_CONFIG}")
        if not args.dry_run and current != RTDETR_COMMIT:
            raise RuntimeError(f"Could not pin RT-DETR checkout to {RTDETR_COMMIT}")


if __name__ == "__main__":
    main()
