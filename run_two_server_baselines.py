#!/usr/bin/env python3
"""Run the six LEVIR-Ship/TinyPerson baselines across two servers."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


JOBS = {
    1: [
        ("levir", "retinanet"),
        ("tinyperson", "cascade_rcnn"),
        ("levir", "rtmdet"),
    ],
    2: [
        ("tinyperson", "retinanet"),
        ("levir", "cascade_rcnn"),
        ("tinyperson", "rtmdet"),
    ],
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=int, choices=(1, 2), required=True)
    parser.add_argument("--python", default="/marimo/mmdet-venv/bin/python")
    parser.add_argument("--install-script", default="/marimo/mmdet_code/install.sh")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--code-root", type=Path, default=root)
    parser.add_argument("--levir-root", default="LevirShipData")
    parser.add_argument("--tinyperson-root", default="../TinyPerson/tiny_set")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--levir-batch-size", type=int, default=4)
    parser.add_argument("--tinyperson-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_root = args.code_root.resolve()
    python = Path(args.python)
    if args.install:
        subprocess.run(["bash", args.install_script], check=True)
    if not python.is_file():
        raise FileNotFoundError(
            f"MMDetection Python not found: {python}. Run with --install or execute "
            f"bash {args.install_script} first."
        )

    jobs = JOBS[args.machine]
    print(f"Machine {args.machine} jobs: {jobs}")
    if args.list:
        return

    env = os.environ.copy()
    env["MMDET_PYTHON"] = str(python)
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(code_root / "mmdetection"), env.get("PYTHONPATH", ""))
        if value
    )
    for seed in args.seeds:
        for dataset, model in jobs:
            work_dir = code_root / "mmdetection" / "work_dirs" / (
                f"{dataset}_baseline_split{args.split_seed}_seed{seed}"
            )
            if dataset == "levir":
                command = [
                    str(python),
                    str(code_root / "train_all_levir_baseline.py"),
                    "--models", model, "--data-root", args.levir_root,
                    "--epochs", str(1 if args.smoke_test else args.epochs),
                    "--split-seed", str(args.split_seed), "--seed", str(seed),
                    "--work-dir", str(work_dir),
                    "--batch-size", str(args.levir_batch_size),
                    "--num-workers", str(args.num_workers),
                    "--python", str(python), "--no-hf-upload",
                ]
                if args.smoke_test:
                    command += ["--limit", "16"]
            else:
                command = [
                    str(python),
                    str(code_root / "train_all_tinyperson_baseline.py"),
                    "--models", model, "--data-root", args.tinyperson_root,
                    "--epochs", str(1 if args.smoke_test else args.epochs),
                    "--split-seed", str(args.split_seed), "--seed", str(seed),
                    "--work-dir", str(work_dir),
                    "--batch-size", str(args.tinyperson_batch_size),
                    "--num-workers", str(args.num_workers),
                    "--python", str(python),
                ]
                if args.smoke_test:
                    command += ["--limit", "32", "--skip-final-metrics"]
            if args.amp:
                command.append("--amp")
            if args.resume:
                command.append("--resume")
            if args.dry_run:
                command.append("--dry-run")
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, cwd=code_root, env=env, check=True)


if __name__ == "__main__":
    main()
