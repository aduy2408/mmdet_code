#!/usr/bin/env python3
"""Run an MMDetection test script after loading external model registrations."""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: mmdet_test_wrapper.py TEST_SCRIPT [ARGS ...]")
    import mmdet.models  # noqa: F401

    target = sys.argv[1]
    sys.argv = [target, *sys.argv[2:]]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
