#!/usr/bin/env python3
"""Train, evaluate, and upload DynamicVis on the fixed LEVIR-Ship split."""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DYNAMICVIS = ROOT / "DynamicVis"
BASE_CONFIG = DYNAMICVIS / "configs_DynamicVis" / "Levir-Ship" / "dynamicvis_b_levirship_mamba.py"
PRETRAINED_HF_FILE = "pretrain_dynamicvis_b_bf16_mamba_epoch_200.pth"
PRETRAINED_REPO_FILE = "work_dirs/fMoW/pretrain_dynamicvis_b_bf16_mamba/epoch_200.pth"
PUBLISHED_COUNTS = {"train": 2320, "val": 788, "test": 788}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read image size: {path}")
        height, width = image.shape[:2]
        return width, height


def prepare_coco(args: argparse.Namespace) -> Path:
    split_dir = args.dataset_root / "levir_ship_yolo_seed42"
    if not split_dir.exists():
        raise RuntimeError(f"Fixed split not found: {split_dir}")

    ann_dir = split_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for split, expected in PUBLISHED_COUNTS.items():
        image_dir = split_dir / "images" / split
        label_dir = split_dir / "labels" / split
        images = sorted(image_dir.glob("*.png"))
        if len(images) != expected:
            raise ValueError(f"{split} has {len(images)} images, expected {expected}")
        if not label_dir.is_dir():
            raise RuntimeError(f"Missing label directory: {label_dir}")

        out_json = ann_dir / f"{split}.json"
        if out_json.is_file():
            existing = json.loads(out_json.read_text(encoding="utf-8"))
            if len(existing.get("images", [])) == expected:
                continue

        coco_images: list[dict] = []
        coco_annotations: list[dict] = []
        annotation_id = 1
        for image_id, image_path in enumerate(images, start=1):
            width, height = image_size(image_path)
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                }
            )
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = raw.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    raise ValueError(f"Bad YOLO label at {label_path}:{line_number}: {raw}")
                cls_id, xc, yc, bw, bh = map(float, parts)
                if int(cls_id) != 0:
                    raise ValueError(f"Unexpected class id at {label_path}:{line_number}: {cls_id}")
                box_w, box_h = bw * width, bh * height
                x = (xc * width) - (box_w / 2)
                y = (yc * height) - (box_h / 2)
                x = max(0.0, min(float(width), x))
                y = max(0.0, min(float(height), y))
                box_w = max(0.0, min(float(width) - x, box_w))
                box_h = max(0.0, min(float(height) - y, box_h))
                if box_w <= 0 or box_h <= 0:
                    continue
                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x, y, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

        out_json.write_text(
            json.dumps(
                {
                    "images": coco_images,
                    "annotations": coco_annotations,
                    "categories": [{"id": 1, "name": "ship"}],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return split_dir


def ensure_pretrained(args: argparse.Namespace) -> Path:
    checkpoint = args.pretrained_ckpt or (DYNAMICVIS / PRETRAINED_REPO_FILE)
    checkpoint = checkpoint.resolve()
    if checkpoint.is_file():
        return checkpoint

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    flat_checkpoint = DYNAMICVIS / PRETRAINED_HF_FILE
    if flat_checkpoint.is_file():
        shutil.copy2(flat_checkpoint, checkpoint)
        return checkpoint

    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="KyanChen/DynamicVis",
            filename=PRETRAINED_HF_FILE,
            local_dir=str(DYNAMICVIS),
            local_dir_use_symlinks=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"Missing pretrained checkpoint {checkpoint}; automatic HF download failed: {error!r}"
        ) from error
    if flat_checkpoint.is_file() and not checkpoint.is_file():
        shutil.copy2(flat_checkpoint, checkpoint)
    if not checkpoint.is_file():
        raise RuntimeError(f"Pretrained checkpoint was not downloaded: {checkpoint}")
    return checkpoint


def write_config(run_dir: Path, data_root: Path, pretrained: Path, seed: int, args: argparse.Namespace) -> Path:
    config_path = run_dir / "dynamicvis_levir_config.py"
    text = f"""_base_ = {str(BASE_CONFIG)!r}

custom_imports = dict(imports='dynamicvis', allow_failed_imports=False)
default_scope = 'mmdet'
data_root = {str(data_root)!r}
code_root = {str(DYNAMICVIS)!r}
pretrained_ckpt = {str(pretrained)!r}
work_dir = {str(run_dir)!r}
batch_size = {args.batch_size}
img_size = {args.imgsz}
crop_size = (img_size, img_size)
num_workers = {args.workers}
persistent_workers = {bool(args.workers > 0)}
randomness = dict(seed={seed}, deterministic=False)
train_cfg = dict(by_epoch=True, max_epochs={args.epochs}, val_interval={args.val_interval})
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer', line_width=2)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval={args.checkpoint_interval}, by_epoch=True,
        max_keep_ckpts=5, save_last=True,
        save_best=['coco/bbox_mAP'],
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=False, score_thr=0.3, interval=1)
)
model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=pretrained_ckpt, prefix='backbone.')),
    neck=dict(init_cfg=dict(type='Pretrained', checkpoint=pretrained_ckpt, prefix='pre_neck.')),
    test_cfg=dict(nms=dict(type='nms', iou_threshold=0.5))
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        data_root=data_root,
        ann_file=data_root + '/annotations/train.json',
        data_prefix=dict(img='images/train')))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    dataset=dict(
        data_root=data_root,
        ann_file=data_root + '/annotations/val.json',
        data_prefix=dict(img='images/val')))
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    dataset=dict(
        data_root=data_root,
        ann_file=data_root + '/annotations/test.json',
        data_prefix=dict(img='images/test')))
val_evaluator = dict(iou_thrs=[0.5], ann_file=data_root + '/annotations/val.json')
test_evaluator = dict(iou_thrs=[0.5], ann_file=data_root + '/annotations/test.json')
"""
    config_path.write_text(text, encoding="utf-8")
    return config_path


def env() -> dict[str, str]:
    result = os.environ.copy()
    paths = [str(DYNAMICVIS), str(ROOT / "mmdetection")]
    if result.get("PYTHONPATH"):
        paths.append(result["PYTHONPATH"])
    result["PYTHONPATH"] = os.pathsep.join(paths)
    result.setdefault("WANDB_MODE", "disabled")
    result.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dynamicvis")
    result.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    result.setdefault("PYTHONUNBUFFERED", "1")
    return result


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> str:
    print("[cmd]", " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=str(cwd), env=env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(proc.stdout)
    print(proc.stdout[-3000:], flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")
    return proc.stdout


def complete(run_dir: Path) -> bool:
    return bool(find_best_checkpoint(run_dir)) and (run_dir / "last_checkpoint").is_file()


def find_best_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("best_*.pth"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def parse_metrics(output: str) -> dict:
    metrics: dict[str, float | str] = {}
    matches = re.findall(r"OrderedDict\((\[.*?\])\)", output, flags=re.S)
    for match in matches:
        try:
            for key, value in ast.literal_eval(match):
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
                else:
                    metrics[key] = value
        except Exception:
            continue
    for key, value in re.findall(r"(coco/[A-Za-z0-9_]+)\s*[:=]\s*([0-9.]+)", output):
        metrics[key] = float(value)
    return metrics


def train(run_dir: Path, config_path: Path, seed: int, args: argparse.Namespace) -> None:
    if complete(run_dir):
        print(f"[train] seed={seed} already has best checkpoint and last_checkpoint; skipping.", flush=True)
        return
    seed_everything(seed)
    cmd = [
        sys.executable,
        str(DYNAMICVIS / "tools_mmdet" / "train.py"),
        str(config_path),
        "--work-dir",
        str(run_dir),
    ]
    if args.amp:
        cmd.append("--amp")
    run_command(cmd, DYNAMICVIS, run_dir / "train.log")
    if not complete(run_dir):
        raise RuntimeError(f"Training finished without required DynamicVis checkpoints in {run_dir}")


def evaluate(run_dir: Path, config_path: Path, args: argparse.Namespace) -> dict:
    best = find_best_checkpoint(run_dir)
    if not best:
        raise RuntimeError(f"Cannot evaluate without best checkpoint in {run_dir}")
    all_metrics = {"checkpoint": str(best), "nms_iou": 0.5}
    for split in ("val", "test"):
        eval_dir = run_dir / "evaluation" / split
        eval_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(DYNAMICVIS / "tools_mmdet" / "test.py"),
            str(config_path),
            str(best),
            "--work-dir",
            str(eval_dir),
        ]
        output = run_command(cmd, DYNAMICVIS, run_dir / f"eval_{split}.log")
        metrics = parse_metrics(output)
        if not metrics:
            raise RuntimeError(f"No metrics parsed from {split} evaluation output")
        for key, value in metrics.items():
            all_metrics[f"{split}/{key}"] = value
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return all_metrics


def write_manifest(run_dir: Path, seed: int, args: argparse.Namespace, data_root: Path, pretrained: Path) -> None:
    manifest = {
        "model": "DynamicVis-B FCOS",
        "dynamicvis_repo": str(DYNAMICVIS),
        "base_config": str(BASE_CONFIG),
        "train_seed": seed,
        "split_seed": 42,
        "dataset": "LEVIR-Ship",
        "data_root": str(data_root),
        "pretrained_ckpt": str(pretrained),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class Uploader:
    def __init__(self, repo_id: str) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required before training starts")
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.api.whoami()
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    @staticmethod
    def retry(func):
        for attempt in range(3):
            try:
                return func()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    def upload_run(self, seed: int, run_dir: Path) -> None:
        best = find_best_checkpoint(run_dir)
        required = [
            best,
            run_dir / "last_checkpoint",
            run_dir / "dynamicvis_levir_config.py",
            run_dir / "evaluation_metrics.json",
            run_dir / "experiment_manifest.json",
            run_dir / "train.log",
            run_dir / "eval_val.log",
            run_dir / "eval_test.log",
        ]
        missing = [str(path) for path in required if path is None or not path.is_file()]
        if missing:
            raise RuntimeError(f"Refusing incomplete upload, missing: {missing}")

        remote = f"runs/dynamicvis/seed_{seed}"
        self.retry(lambda: self.api.upload_folder(folder_path=run_dir, path_in_repo=remote, repo_id=self.repo_id, repo_type="dataset"))
        expected = {f"{remote}/{path.relative_to(run_dir)}" for path in required if path is not None}
        uploaded = set(self.retry(lambda: self.api.list_repo_files(self.repo_id, repo_type="dataset")))
        missing_remote = sorted(expected - uploaded)
        if missing_remote:
            raise RuntimeError(f"HF verification failed, missing: {missing_remote}")
        marker = run_dir / "upload_complete.json"
        marker.write_text(
            json.dumps({"repo_id": self.repo_id, "seed": seed, "verified": sorted(expected)}, indent=2) + "\n",
            encoding="utf-8",
        )
        self.retry(
            lambda: self.api.upload_file(
                path_or_fileobj=marker,
                path_in_repo=f"{remote}/{marker.name}",
                repo_id=self.repo_id,
                repo_type="dataset",
            )
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "yolo_related" / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "levir_dynamicvis")
    parser.add_argument("--pretrained-ckpt", type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-dynamicvis-3seed")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not DYNAMICVIS.is_dir():
        raise RuntimeError(f"DynamicVis clone not found: {DYNAMICVIS}")
    if not BASE_CONFIG.is_file():
        raise RuntimeError(f"DynamicVis LEVIR config not found: {BASE_CONFIG}")

    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    args.project.mkdir(parents=True, exist_ok=True)

    uploader = Uploader(args.hf_repo_id)
    data_root = prepare_coco(args)
    pretrained = ensure_pretrained(args)

    for seed in args.seeds:
        print(f"\n{'=' * 70}\nDynamicVis seed={seed} split_seed=42\n{'=' * 70}", flush=True)
        run_dir = args.project / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = write_config(run_dir, data_root, pretrained, seed, args)
        shutil.copy2(BASE_CONFIG, run_dir / "base_dynamicvis_levirship_mamba.py")
        train(run_dir, config_path, seed, args)
        metrics = evaluate(run_dir, config_path, args)
        write_manifest(run_dir, seed, args, data_root, pretrained)
        print(f"[seed={seed}] test coco/bbox_mAP={metrics.get('test/coco/bbox_mAP', 'N/A')}", flush=True)
        uploader.upload_run(seed, run_dir)
        print(f"[seed={seed}] upload verified", flush=True)


if __name__ == "__main__":
    main()
