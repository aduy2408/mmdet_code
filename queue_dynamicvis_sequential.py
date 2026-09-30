#!/usr/bin/env python3
"""Sequential DynamicVis queue: launch LevirShip only after TinyPerson succeeds."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path

PYTHON = '/marimo/dynamicvis-blackwell-venv/bin/python'
ROOT = Path('/marimo/mmdet_code')

def status(run_dir: Path) -> dict:
    out = subprocess.check_output([PYTHON, '-m', 'utils.marimo_ops', 'status', '--run-dir', str(run_dir)], cwd=ROOT, text=True)
    return json.loads(out)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--model-yaml', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--nms-iou', type=float, default=0.5)
    p.add_argument('--hf-repo-id', required=True)
    p.add_argument('--queue-run-dir', type=Path, default=ROOT / 'runs/dynamicvis_sequential_queue')
    args = p.parse_args()
    first = ROOT / 'runs/dynamicvis_tinyperson_mosaic'
    second = ROOT / 'runs/dynamicvis_levirship_mosaic'
    args.queue_run_dir.mkdir(parents=True, exist_ok=True)
    marker = args.queue_run_dir / 'queue_state.json'
    while True:
        state = status(first)
        marker.write_text(json.dumps({'first_run': str(first), 'second_run': str(second), 'first_status': state, 'updated_at': time.time()}, indent=2))
        if state.get('process_alive'):
            time.sleep(60)
            continue
        log = first / 'train.log'
        text = log.read_text(errors='replace') if log.exists() else ''
        failed = 'Traceback (most recent call last)' in text or 'ERROR' in text or 'Error:' in text
        checkpoints = list(first.glob('*.pth')) + list(first.glob('**/*.pth'))
        if failed or not checkpoints:
            marker.write_text(json.dumps({'status': 'blocked', 'reason': 'TinyPerson did not finish successfully', 'checkpoints': [str(x) for x in checkpoints]}, indent=2))
            raise SystemExit('TinyPerson gate failed, LevirShip was not launched')
        cmd = [PYTHON, '-m', 'utils.marimo_ops', 'launch', '--cwd', str(ROOT), '--run-dir', str(second), '--artifact-root', str(second), '--', PYTHON, str(ROOT / 'train_dynamicvis_yolo_mosaic.py'), '--dataset', 'levirship', '--data-root', '/marimo/LevirShip/LevirShipData', '--work-dir', str(second), '--model-yaml', args.model_yaml, '--epochs', str(args.epochs), '--patience', str(args.patience), '--workers', str(args.workers), '--seed', str(args.seed), '--split-seed', str(args.split_seed), '--nms-iou', str(args.nms_iou), '--hf-repo-id', args.hf_repo_id]
        subprocess.run(cmd, cwd=ROOT, check=True)
        marker.write_text(json.dumps({'status': 'levirship_launched', 'first_run': str(first), 'second_run': str(second)}, indent=2))
        return

if __name__ == '__main__':
    main()
