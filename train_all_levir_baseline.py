#!/usr/bin/env python3
"""Prepare LEVIR-Ship as COCO and train MMDetection baselines."""

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
    "atss": "configs/atss/atss_r50_fpn_1x_coco.py",
    "retinanet": "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
    "faster_rcnn": "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    "fcos": "configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
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
        raise ValueError(f"Cannot extract source scene from image name: {image_path.name}")
    return match.group(1)


def discover_samples(data_root: Path) -> list[tuple[Path, Path, str]]:
    image_dir = data_root / "All Images"
    annotation_dir = data_root / "All Annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"Expected 'All Images' and 'All Annotations' under {data_root}"
        )

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
    orphan_annotations = sorted(annotation_stems - image_stems)
    if orphan_annotations:
        raise ValueError(
            f"Found {len(orphan_annotations)} annotations without images; "
            f"first: {orphan_annotations[0]}.txt"
        )
    return samples


def split_by_scene(
    samples: list[tuple[Path, Path, str]], seed: int
) -> dict[str, list[tuple[Path, Path, str]]]:
    groups: dict[str, list[tuple[Path, Path, str]]] = defaultdict(list)
    for sample in samples:
        groups[sample[2]].append(sample)

    rng = random.Random(seed)
    scenes = list(groups)
    rng.shuffle(scenes)
    scenes.sort(key=lambda name: len(groups[name]), reverse=True)

    targets = {split: len(samples) * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: 0 for split in SPLIT_RATIOS}
    assignments: dict[str, str] = {}
    for scene in scenes:
        size = len(groups[scene])

        def cost(split: str) -> float:
            projected = dict(counts)
            projected[split] += size
            return sum(
                ((projected[name] - targets[name]) / targets[name]) ** 2
                for name in SPLIT_RATIOS
            )

        split = min(SPLIT_RATIOS, key=cost)
        assignments[scene] = split
        counts[split] += size

    output = {split: [] for split in SPLIT_RATIOS}
    for scene, group in groups.items():
        output[assignments[scene]].extend(group)
    for split in output:
        output[split].sort(key=lambda sample: sample[0].name)
    return output


def yolo_boxes(annotation_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line_number, line in enumerate(
        annotation_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{annotation_path}:{line_number}: expected 5 YOLO values"
            )
        class_id, cx, cy, box_width, box_height = map(float, parts)
        if class_id != 0:
            raise ValueError(
                f"{annotation_path}:{line_number}: expected class 0, got {class_id:g}"
            )
        x1 = max(0.0, (cx - box_width / 2) * width)
        y1 = max(0.0, (cy - box_height / 2) * height)
        x2 = min(float(width), (cx + box_width / 2) * width)
        y2 = min(float(height), (cy + box_height / 2) * height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{annotation_path}:{line_number}: invalid bounding box")
        boxes.append([x1, y1, x2 - x1, y2 - y1])
    return boxes


def prepare_coco_dataset(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = resolve_path(args.data_root)
    dataset_out = resolve_path(args.dataset_out)
    split_samples = split_by_scene(discover_samples(data_root), args.seed)
    annotation_dir = dataset_out / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    scene_sets: dict[str, set[str]] = {}
    for split, samples in split_samples.items():
        selected = samples[: args.limit] if args.limit > 0 else samples
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        annotation_id = 1
        for image_id, (image_path, label_path, _) in enumerate(selected, 1):
            with Image.open(image_path) as image:
                width, height = image.size
            images.append(
                dict(id=image_id, file_name=image_path.name, width=width, height=height)
            )
            for bbox in yolo_boxes(label_path, width, height):
                annotations.append(
                    dict(
                        id=annotation_id,
                        image_id=image_id,
                        category_id=1,
                        bbox=bbox,
                        area=bbox[2] * bbox[3],
                        iscrowd=0,
                        segmentation=[],
                    )
                )
                annotation_id += 1
        payload = dict(
            images=images,
            annotations=annotations,
            categories=[dict(id=1, name="ship", supercategory="ship")],
        )
        (annotation_dir / f"{split}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        scene_sets[split] = {sample[2] for sample in selected}
        negatives = len(images) - len({ann["image_id"] for ann in annotations})
        print(
            f"{split}: images={len(images)} scenes={len(scene_sets[split])} "
            f"boxes={len(annotations)} negatives={negatives}"
        )

    split_names = list(scene_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = scene_sets[left] & scene_sets[right]
            if overlap:
                raise AssertionError(f"Scene leakage between {left} and {right}: {overlap}")
    return dataset_out, data_root / "All Images"


def set_num_classes(obj: Any) -> None:
    if isinstance(obj, dict):
        if "num_classes" in obj:
            obj["num_classes"] = 1
        for value in obj.values():
            set_num_classes(value)
    elif isinstance(obj, list):
        for value in obj:
            set_num_classes(value)


def set_resize_scale(obj: Any) -> None:
    if isinstance(obj, dict):
        if obj.get("type") == "Resize":
            obj["scale"] = (512, 512)
        for value in obj.values():
            set_resize_scale(value)
    elif isinstance(obj, list):
        for value in obj:
            set_resize_scale(value)


def patch_dataset(dataset: Any, dataset_out: Path, image_dir: Path, split: str) -> None:
    dataset.data_root = ""
    dataset.ann_file = str(dataset_out / "annotations" / f"{split}.json")
    dataset.data_prefix = dict(img=f"{image_dir}/")
    dataset.metainfo = dict(classes=("ship",))
    set_resize_scale(dataset.pipeline)


def patch_config(
    cfg: Any,
    model_name: str,
    args: argparse.Namespace,
    dataset_out: Path,
    image_dir: Path,
) -> Any:
    set_num_classes(cfg.model)
    cfg.val_dataloader = deepcopy(cfg.val_dataloader)
    cfg.test_dataloader = deepcopy(cfg.test_dataloader)
    patch_dataset(cfg.train_dataloader.dataset, dataset_out, image_dir, "train")
    patch_dataset(cfg.val_dataloader.dataset, dataset_out, image_dir, "val")
    patch_dataset(cfg.test_dataloader.dataset, dataset_out, image_dir, "test")

    for dataloader in (
        cfg.train_dataloader,
        cfg.val_dataloader,
        cfg.test_dataloader,
    ):
        dataloader.num_workers = args.num_workers
        dataloader.persistent_workers = args.num_workers > 0
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.val_evaluator.ann_file = str(dataset_out / "annotations" / "val.json")
    cfg.test_evaluator.ann_file = str(dataset_out / "annotations" / "test.json")
    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = 1
    cfg.work_dir = str(resolve_path(args.work_dir) / model_name)
    cfg.default_hooks.checkpoint.update(
        interval=1,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=1,
        save_last=True,
    )
    cfg.randomness = dict(seed=args.seed)
    return cfg


def write_config(
    model_name: str,
    args: argparse.Namespace,
    dataset_out: Path,
    image_dir: Path,
) -> Path:
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


def run(command: list[str]) -> None:
    print("RUN", " ".join(map(str, command)))
    subprocess.run(command, cwd=mmdet_root(), check=True)


def upload_work_dir_to_hf(model_name: str, args: argparse.Namespace) -> None:
    if args.no_hf_upload:
        return
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "Hugging Face upload requires --hf-token or HF_TOKEN; "
            "pass --no-hf-upload to skip."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Hugging Face upload requires `huggingface_hub`; "
            "install it or pass --no-hf-upload."
        ) from exc

    work_dir = resolve_path(args.work_dir) / model_name
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        private=False,
        exist_ok=True,
    )
    print(f"UPLOAD {work_dir} -> hf://{args.hf_repo_type}/{args.hf_repo_id}/{model_name}")
    api.upload_folder(
        folder_path=str(work_dir),
        path_in_repo=model_name,
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
    )


def run_job(
    model_name: str,
    args: argparse.Namespace,
    dataset_out: Path,
    image_dir: Path,
) -> None:
    config_path = write_config(model_name, args, dataset_out, image_dir)
    work_dir = resolve_path(args.work_dir) / model_name
    if not args.test_only:
        command = [
            sys.executable,
            str(mmdet_root() / "tools" / "train.py"),
            str(config_path),
            "--work-dir",
            str(work_dir),
            "--auto-scale-lr",
        ]
        if args.amp:
            command.append("--amp")
        run(command)
    checkpoint = find_checkpoint(work_dir)
    run(
        [
            sys.executable,
            str(mmdet_root() / "tools" / "test.py"),
            str(config_path),
            str(checkpoint),
            "--work-dir",
            str(work_dir / "test_results"),
            "--out",
            str(work_dir / "test_results" / "predictions.pkl"),
        ]
    )
    upload_work_dir_to_hf(model_name, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="LevirShipData")
    parser.add_argument("--dataset-out", default="mmdetection/data/levir_ship_coco")
    parser.add_argument("--work-dir", default="mmdetection/work_dirs/levir_baseline")
    parser.add_argument(
        "--models",
        default="atss,retinanet,faster_rcnn,fcos",
        help="Comma-separated: atss, retinanet, faster_rcnn, fcos.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum images per split after scene-safe splitting; 0 uses all images.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir_ship_mmdet_runs")
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument(
        "--hf-token",
        default="",
        help="Hugging Face token. Defaults to HF_TOKEN from the environment.",
    )
    parser.add_argument(
        "--no-hf-upload",
        action="store_true",
        help="Skip uploading each completed model to Hugging Face.",
    )
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
        model
        for index, model in enumerate(models)
        if index % args.num_machines == args.machine_index
    ]
    print(f"Assigned models ({args.machine_index}/{args.num_machines}): {assigned}")
    if args.dry_run:
        for model_name in assigned:
            config_path = write_config(
                model_name, args, dataset_out, image_dir
            )
            print(f"CONFIG {model_name}: {config_path}")
        return
    for model_name in assigned:
        run_job(model_name, args, dataset_out, image_dir)


if __name__ == "__main__":
    main()
