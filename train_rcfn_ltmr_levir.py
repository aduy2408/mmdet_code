#!/usr/bin/env python3
"""Prepare LEVIR-Ship and train the RCFN/PG-RCFN/LTMR ablations."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image


MODEL_CONFIGS = {
    "r2": "projects/rcfn_ltmr/configs/fcos_r2.py",
    "pg_aux": "projects/rcfn_ltmr/configs/fcos_pg_aux.py",
    "pg_aux_w01": "projects/rcfn_ltmr/configs/fcos_pg_aux_w01.py",
    "pg_h": "projects/rcfn_ltmr/configs/fcos_pg_h.py",
    "pg_ch": "projects/rcfn_ltmr/configs/fcos_pg_ch.py",
    "pg_ch_w01_floor":
        "projects/rcfn_ltmr/configs/fcos_pg_ch_w01_floor.py",
    "l1": "projects/rcfn_ltmr/configs/fcos_l1.py",
}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SCENE_RE = re.compile(r"^(.*)_(-?\d+)_(-?\d+)$")


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def mmdet_root() -> Path:
    root = repo_root() / "mmdetection"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing MMDetection checkout: {root}")
    return root


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root() / path).resolve()


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def scene_name(image_path: Path) -> str:
    match = SCENE_RE.match(image_path.stem)
    if not match:
        raise ValueError(f"Cannot extract source scene from {image_path.name}")
    return match.group(1)


def discover_samples(data_root: Path) -> list[tuple[Path, Path, str]]:
    image_dir = data_root / "All Images"
    annotation_dir = data_root / "All Annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"Expected 'All Images' and 'All Annotations' under {data_root}")
    samples = []
    for image_path in sorted(image_dir.glob("*.png")):
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation for {image_path.name}")
        samples.append((image_path, annotation_path, scene_name(image_path)))
    if not samples:
        raise ValueError(f"No PNG images found in {image_dir}")
    annotation_stems = {path.stem for path in annotation_dir.glob("*.txt")}
    image_stems = {sample[0].stem for sample in samples}
    orphans = sorted(annotation_stems - image_stems)
    if orphans:
        raise ValueError(
            f"Found {len(orphans)} annotations without images; "
            f"first: {orphans[0]}.txt")
    return samples


def split_by_scene(
        samples: list[tuple[Path, Path, str]],
        seed: int) -> dict[str, list[tuple[Path, Path, str]]]:
    groups: dict[str, list[tuple[Path, Path, str]]] = defaultdict(list)
    for sample in samples:
        groups[sample[2]].append(sample)
    scenes = list(groups)
    random.Random(seed).shuffle(scenes)
    scenes.sort(key=lambda name: len(groups[name]), reverse=True)
    targets = {
        split: len(samples) * ratio for split, ratio in SPLIT_RATIOS.items()
    }
    counts = {split: 0 for split in SPLIT_RATIOS}
    assignments = {}
    for scene in scenes:
        size = len(groups[scene])

        def cost(split: str) -> float:
            projected = dict(counts)
            projected[split] += size
            return sum(
                ((projected[name] - targets[name]) / targets[name])**2
                for name in SPLIT_RATIOS)

        split = min(SPLIT_RATIOS, key=cost)
        assignments[scene] = split
        counts[split] += size
    output = {split: [] for split in SPLIT_RATIOS}
    for scene, group in groups.items():
        output[assignments[scene]].extend(group)
    for split in output:
        output[split].sort(key=lambda sample: sample[0].name)
    return output


def yolo_boxes(annotation_path: Path, width: int,
               height: int) -> list[list[float]]:
    boxes = []
    for line_number, line in enumerate(
            annotation_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{annotation_path}:{line_number}: expected 5 YOLO values")
        class_id, cx, cy, box_width, box_height = map(float, parts)
        if class_id != 0:
            raise ValueError(
                f"{annotation_path}:{line_number}: expected class 0")
        x1 = max(0.0, (cx - box_width / 2) * width)
        y1 = max(0.0, (cy - box_height / 2) * height)
        x2 = min(float(width), (cx + box_width / 2) * width)
        y2 = min(float(height), (cy + box_height / 2) * height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"{annotation_path}:{line_number}: invalid bounding box")
        boxes.append([x1, y1, x2 - x1, y2 - y1])
    return boxes


def prepare_coco_dataset(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = resolve_path(args.data_root)
    dataset_out = resolve_path(args.dataset_out)
    splits = split_by_scene(discover_samples(data_root), args.seed)
    annotation_dir = dataset_out / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    scene_sets = {}
    for split, samples in splits.items():
        selected = samples[:args.limit] if args.limit > 0 else samples
        images, annotations = [], []
        annotation_id = 1
        for image_id, (image_path, label_path, _) in enumerate(selected, 1):
            with Image.open(image_path) as image:
                width, height = image.size
            images.append(
                dict(id=image_id, file_name=image_path.name,
                     width=width, height=height))
            for bbox in yolo_boxes(label_path, width, height):
                annotations.append(
                    dict(
                        id=annotation_id,
                        image_id=image_id,
                        category_id=1,
                        bbox=bbox,
                        area=bbox[2] * bbox[3],
                        iscrowd=0,
                        segmentation=[]))
                annotation_id += 1
        payload = dict(
            images=images,
            annotations=annotations,
            categories=[dict(id=1, name="ship", supercategory="ship")])
        (annotation_dir / f"{split}.json").write_text(
            json.dumps(payload), encoding="utf-8")
        scene_sets[split] = {sample[2] for sample in selected}
        positives = {ann["image_id"] for ann in annotations}
        print(
            f"{split}: images={len(images)} scenes={len(scene_sets[split])} "
            f"boxes={len(annotations)} negatives={len(images) - len(positives)}")
    names = list(scene_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = scene_sets[left] & scene_sets[right]
            if overlap:
                raise AssertionError(
                    f"Scene leakage between {left} and {right}: {overlap}")
    return dataset_out, data_root / "All Images"


def update_nested(obj: Any, callback) -> None:
    if isinstance(obj, dict):
        callback(obj)
        for value in obj.values():
            update_nested(value, callback)
    elif isinstance(obj, list):
        for value in obj:
            update_nested(value, callback)


def patch_dataset(dataset: Any, dataset_out: Path,
                  image_dir: Path, split: str) -> None:
    dataset.data_root = ""
    dataset.ann_file = str(dataset_out / "annotations" / f"{split}.json")
    dataset.data_prefix = dict(img=f"{image_dir}/")
    dataset.metainfo = dict(classes=("ship",))
    update_nested(
        dataset.pipeline,
        lambda item: item.update(scale=(512, 512))
        if item.get("type") == "Resize" else None)


def patch_config(cfg: Any, model_name: str, args: argparse.Namespace,
                 dataset_out: Path, image_dir: Path) -> Any:
    update_nested(
        cfg.model,
        lambda item: item.update(num_classes=1)
        if "num_classes" in item else None)
    cfg.val_dataloader = deepcopy(cfg.val_dataloader)
    cfg.test_dataloader = deepcopy(cfg.test_dataloader)
    patch_dataset(cfg.train_dataloader.dataset, dataset_out, image_dir, "train")
    patch_dataset(cfg.val_dataloader.dataset, dataset_out, image_dir, "val")
    patch_dataset(cfg.test_dataloader.dataset, dataset_out, image_dir, "test")
    for dataloader in (
            cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        dataloader.num_workers = args.num_workers
        dataloader.persistent_workers = args.num_workers > 0
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.val_evaluator.ann_file = str(
        dataset_out / "annotations" / "val.json")
    cfg.test_evaluator.ann_file = str(
        dataset_out / "annotations" / "test.json")
    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = 1
    cfg.work_dir = str(resolve_path(args.work_dir) / model_name)
    cfg.default_hooks.checkpoint.update(
        interval=1,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=1,
        save_last=True)
    cfg.randomness = dict(seed=args.seed)
    return cfg


def write_config(model_name: str, args: argparse.Namespace,
                 dataset_out: Path, image_dir: Path) -> Path:
    root = str(mmdet_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from mmengine.config import Config
    from mmdet.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(str(mmdet_root() / MODEL_CONFIGS[model_name]))
    cfg = patch_config(cfg, model_name, args, dataset_out, image_dir)
    output = Path(cfg.work_dir) / "patched_config.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    return output


def find_checkpoint(work_dir: Path) -> Path:
    best = sorted(work_dir.glob("best_*.pth"))
    if best:
        return best[0]
    latest = work_dir / "latest.pth"
    if latest.is_file():
        return latest
    raise FileNotFoundError(f"No best_*.pth or latest.pth in {work_dir}")


def find_test_metrics(test_dir: Path) -> Path:
    metrics = sorted(test_dir.glob("*/*.json"))
    if not metrics:
        raise FileNotFoundError(f"No test metrics JSON under {test_dir}")
    return metrics[-1]


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{mmdet_root()}{os.pathsep}{current}" if current
        else str(mmdet_root()))
    return env


def run(command: list[str]) -> None:
    print("RUN", " ".join(map(str, command)))
    subprocess.run(
        command, cwd=mmdet_root(), env=command_env(), check=True)


def upload_work_dir(model_name: str, args: argparse.Namespace) -> None:
    if args.no_hf_upload:
        return
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "Hugging Face upload requires --hf-token or HF_TOKEN; "
            "pass --no-hf-upload to skip.")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub or pass --no-hf-upload.") from exc
    work_dir = resolve_path(args.work_dir) / model_name
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        private=False,
        exist_ok=True)
    api.upload_folder(
        folder_path=str(work_dir),
        path_in_repo=model_name,
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type)


def run_job(model_name: str, args: argparse.Namespace,
            dataset_out: Path, image_dir: Path) -> None:
    config = write_config(model_name, args, dataset_out, image_dir)
    work_dir = resolve_path(args.work_dir) / model_name
    if not args.test_only:
        command = [
            sys.executable, str(mmdet_root() / "tools/train.py"),
            str(config), "--work-dir", str(work_dir), "--auto-scale-lr"
        ]
        if args.amp:
            command.append("--amp")
        run(command)
    checkpoint = find_checkpoint(work_dir)
    test_dir = work_dir / "test_results"
    run([
        sys.executable,
        str(mmdet_root() / "tools/test.py"),
        str(config),
        str(checkpoint),
        "--work-dir",
        str(test_dir),
        "--out",
        str(test_dir / "predictions.pkl"),
    ])
    if not args.skip_diagnostic:
        diagnostic_command = [
            sys.executable,
            str(mmdet_root() / "projects/rcfn_ltmr/tools/diagnose_ltmr.py"),
            str(config),
            str(checkpoint),
            "--variant",
            model_name,
            "--output-dir",
            str(work_dir / "diagnostics"),
        ]
        if model_name.startswith("pg_"):
            reference_test_dir = (
                resolve_path(args.work_dir) / "r2" / "test_results")
            reference_predictions = (
                reference_test_dir / "predictions.pkl")
            if not reference_predictions.is_file():
                raise FileNotFoundError(
                    "Paired PG diagnostics require the R2 predictions at "
                    f"{reference_predictions}")
            diagnostic_command.extend([
                "--reference-predictions", str(reference_predictions),
                "--candidate-predictions",
                str(test_dir / "predictions.pkl"),
                "--reference-test-metrics",
                str(find_test_metrics(reference_test_dir)),
                "--candidate-test-metrics",
                str(find_test_metrics(test_dir)),
            ])
        run(diagnostic_command)
    upload_work_dir(model_name, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="LevirShipData")
    parser.add_argument(
        "--dataset-out", default="mmdetection/data/levir_ship_coco")
    parser.add_argument(
        "--work-dir", default="mmdetection/work_dirs/levir_rcfn_ltmr")
    parser.add_argument(
        "--models", default="r2,pg_aux,pg_h,pg_ch",
        help=(
            "Comma-separated: r2,pg_aux,pg_aux_w01,pg_h,pg_ch,"
            "pg_ch_w01_floor,l1."))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Maximum images per split; 0 uses all images.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--skip-diagnostic", action="store_true")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument(
        "--hf-repo-id", default="duyle2408/levir_ship_mmdet_runs")
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--no-hf-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_machines < 1:
        raise ValueError("--num-machines must be >= 1")
    if not 0 <= args.machine_index < args.num_machines:
        raise ValueError("--machine-index must be in [0, num_machines)")
    models = comma_list(args.models)
    unknown = sorted(set(models) - set(MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")
    dataset_out, image_dir = prepare_coco_dataset(args)
    assigned = [
        model for index, model in enumerate(models)
        if index % args.num_machines == args.machine_index
    ]
    print(f"Assigned models ({args.machine_index}/{args.num_machines}): {assigned}")
    if args.dry_run:
        for model_name in assigned:
            config = write_config(model_name, args, dataset_out, image_dir)
            print(f"CONFIG {model_name}: {config}")
            work_dir = resolve_path(args.work_dir) / model_name
            print(
                "TRAIN",
                sys.executable,
                mmdet_root() / "tools/train.py",
                config,
                "--work-dir",
                work_dir,
                "--auto-scale-lr",
                *(["--amp"] if args.amp else []))
            print(
                "TEST",
                sys.executable,
                mmdet_root() / "tools/test.py",
                config,
                "<checkpoint>",
                "--work-dir",
                work_dir / "test_results")
            if not args.skip_diagnostic:
                paired_args = []
                if model_name.startswith("pg_"):
                    reference_dir = (
                        resolve_path(args.work_dir) / "r2" / "test_results")
                    paired_args = [
                        "--reference-predictions",
                        reference_dir / "predictions.pkl",
                        "--candidate-predictions",
                        work_dir / "test_results" / "predictions.pkl",
                        "--reference-test-metrics", "<r2-test-metrics.json>",
                        "--candidate-test-metrics",
                        "<candidate-test-metrics.json>",
                    ]
                print(
                    "DIAGNOSTIC",
                    sys.executable,
                    mmdet_root()
                    / "projects/rcfn_ltmr/tools/diagnose_ltmr.py",
                    config,
                    "<checkpoint>",
                    "--variant",
                    model_name,
                    "--output-dir",
                    work_dir / "diagnostics",
                    *paired_args)
        return
    for model_name in assigned:
        run_job(model_name, args, dataset_out, image_dir)


if __name__ == "__main__":
    main()
