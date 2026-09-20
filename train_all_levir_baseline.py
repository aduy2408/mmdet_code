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
import threading
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image


MODEL_CONFIGS = {
    "fcos_set": "configs/set/fcos_r50_set.py",
    "atss": "configs/atss/atss_r50_fpn_1x_coco.py",
    "retinanet": "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
    "faster_rcnn": "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    "fcos": "configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
    "cascade_rcnn": "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
    "rtmdet": "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
    "detr": "configs/detr/detr_r50_8xb2-150e_coco.py",
    "dino": "configs/dino/dino-4scale_r50_8xb2-12e_coco.py",
}
# ``dino`` selects MMDetection's DINO-DETR implementation.
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


def default_python() -> str:
    configured = os.environ.get("MMDET_PYTHON")
    if configured:
        return configured
    marimo_python = Path("/marimo/mmdet-venv/bin/python")
    return str(marimo_python) if marimo_python.is_file() else sys.executable


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
    split_samples = split_by_scene(discover_samples(data_root), args.split_seed)
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


def set_resize_scale(obj: Any, image_size: int) -> None:
    if isinstance(obj, dict):
        if obj.get("type") == "Resize":
            obj["scale"] = (image_size, image_size)
        for value in obj.values():
            set_resize_scale(value, image_size)
    elif isinstance(obj, list):
        for value in obj:
            set_resize_scale(value, image_size)


def yolo_pipeline(image_size: int, train: bool, mosaic: bool = True) -> list[dict[str, Any]]:
    """Build the bbox-only equivalent of Ultralytics' default train path.

    Ultralytics defaults are mosaic=1, translate=.1, scale=.5, degrees=0,
    shear=0, HSV=(.015,.7,.4), and fliplr=.5.  MMDetection's YOLOX
    transforms expose the same operations, with HSV expressed in OpenCV
    integer units (hue is 0..180, saturation/value are 0..255).
    """
    scale = (image_size, image_size)
    pipeline: list[dict[str, Any]] = []
    if train and mosaic:
        pipeline.append(dict(type="Mosaic", img_scale=scale, pad_val=114.0))
    if train:
        pipeline.extend([
            dict(
                type="RandomAffine",
                max_rotate_degree=0.0,
                max_translate_ratio=0.1,
                scaling_ratio_range=(0.5, 1.5),
                max_shear_degree=0.0,
                border=(-image_size // 2, -image_size // 2) if mosaic else (0, 0),
            ),
            dict(type="YOLOXHSVRandomAug", hue_delta=3,
                 saturation_delta=179, value_delta=102),
            dict(type="RandomFlip", prob=0.5),
        ])
    pipeline.extend([
        dict(type="Resize", scale=scale, keep_ratio=True),
        dict(type="Pad", pad_to_square=True,
             pad_val=dict(img=(114.0, 114.0, 114.0))),
    ])
    if train:
        pipeline.append(dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1),
                             keep_empty=False))
    pipeline.append(dict(type="PackDetInputs"))
    return pipeline


def patch_dataset(
    dataset: Any,
    dataset_out: Path,
    image_dir: Path,
    split: str,
    image_size: int,
    train: bool,
) -> None:
    """Patch a dataset and wrap train data for MMDetection's Mosaic support."""
    # Copy plain datasets before assigning them under ``dataset.dataset``.
    # Without this, non-YOLO configs make the wrapper point to itself and
    # Config.dump() recurses forever.
    base = deepcopy(dataset.get("dataset", dataset)) if train else dataset
    base.data_root = ""
    base.ann_file = str(dataset_out / "annotations" / f"{split}.json")
    base.data_prefix = dict(img=f"{image_dir}/")
    base.metainfo = dict(classes=("ship",))
    base.pipeline = [
        dict(type="LoadImageFromFile", backend_args=None),
        dict(type="LoadAnnotations", with_bbox=True),
    ]
    if train:
        # MultiImageMixDataset only accepts the nested dataset and wrapper
        # pipeline.  Retaining CocoDataset-only keys such as ann_file causes
        # construction to fail before training starts.
        dataset.clear()
        dataset.type = "MultiImageMixDataset"
        dataset.dataset = base
        dataset.pipeline = yolo_pipeline(image_size, train=True)
    else:
        dataset.pipeline = base.pipeline + yolo_pipeline(image_size, train=False)


def patch_config(
    cfg: Any,
    model_name: str,
    args: argparse.Namespace,
    dataset_out: Path,
    image_dir: Path,
) -> Any:
    set_num_classes(cfg.model)
    # MMDetection 3.3.0 requires both keys when tools/train.py receives
    # --auto-scale-lr. Stock DETR/DINO configs only declare base_batch_size.
    cfg.auto_scale_lr = deepcopy(cfg.get("auto_scale_lr", {}))
    cfg.auto_scale_lr.enable = False
    cfg.auto_scale_lr.setdefault("base_batch_size", 16)
    cfg.val_dataloader = deepcopy(cfg.val_dataloader)
    cfg.test_dataloader = deepcopy(cfg.test_dataloader)
    patch_dataset(
        cfg.train_dataloader.dataset, dataset_out, image_dir, "train", args.image_size, True
    )
    patch_dataset(
        cfg.val_dataloader.dataset, dataset_out, image_dir, "val", args.image_size, False
    )
    patch_dataset(
        cfg.test_dataloader.dataset, dataset_out, image_dir, "test", args.image_size, False
    )

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
    cfg.custom_imports = dict(imports=["mmdet.engine.hooks"], allow_failed_imports=False)
    cfg.custom_hooks = [
        hook for hook in cfg.get("custom_hooks", [])
        if hook.get("type") != "PipelineSwitchHook"
    ]
    cfg.custom_hooks.append(dict(
        type="PipelineSwitchHook",
        switch_epoch=max(0, args.epochs - 10),
        switch_pipeline=yolo_pipeline(args.image_size, train=True, mosaic=False),
    ))
    cfg.custom_hooks.append(dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        rule="greater",
        patience=args.patience,
        min_delta=0.001,
    ))
    cfg.optim_wrapper.optimizer = dict(
        type="MuSGD", lr=0.01, momentum=0.9, nesterov=True,
        weight_decay=0.0005, muon=0.2, sgd=1.0,
    )
    if model_name == "rtmdet":
        milestones = sorted({
            epoch
            for epoch in (
                max(1, round(args.epochs * 2 / 3)),
                max(1, round(args.epochs * 11 / 12)),
            )
            if epoch < args.epochs
        })
        cfg.train_cfg.pop("dynamic_intervals", None)
        cfg.param_scheduler = [
            dict(
                type="LinearLR",
                start_factor=0.001,
                by_epoch=False,
                begin=0,
                end=500,
            ),
            dict(
                type="MultiStepLR",
                by_epoch=True,
                begin=0,
                end=args.epochs,
                milestones=milestones,
                gamma=0.1,
            ),
        ]
    cfg.work_dir = str(resolve_path(args.work_dir) / model_name)
    cfg.test_evaluator.outfile_prefix = str(
        Path(cfg.work_dir) / "test_results" / "test" / "levir_ship"
    )
    cfg.default_hooks.checkpoint.update(
        interval=1,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=1,
        save_last=True,
    )
    imports = list(cfg.get("custom_imports", {}).get("imports", []))
    if "projects.set" not in imports:
        imports.append("projects.set")
    cfg.custom_imports = dict(imports=imports, allow_failed_imports=False)
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

    cfg = Config.fromfile(str(mmdet_root() / MODEL_CONFIGS[model_name]))
    cfg = patch_config(cfg, model_name, args, dataset_out, image_dir)
    output = Path(cfg.work_dir) / "patched_config.py"
    val_output = Path(cfg.work_dir) / "patched_config_val.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    val_cfg = deepcopy(cfg)
    val_cfg.test_dataloader = deepcopy(cfg.val_dataloader)
    val_cfg.test_evaluator = deepcopy(cfg.val_evaluator)
    val_cfg.test_evaluator.outfile_prefix = str(
        Path(cfg.work_dir) / "test_results" / "validation" / "levir_ship"
    )
    val_cfg.dump(str(val_output))
    return output


def find_checkpoint(work_dir: Path) -> Path:
    best = sorted(work_dir.glob("best_*.pth"))
    if best:
        return best[0]
    latest = work_dir / "latest.pth"
    if latest.is_file():
        return latest
    epochs = sorted(
        work_dir.glob("epoch_*.pth"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if epochs:
        return epochs[-1]
    raise FileNotFoundError(
        f"No best_*.pth, latest.pth, or epoch_*.pth in {work_dir}"
    )


def run(command: list[str]) -> None:
    print("RUN", " ".join(map(str, command)))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(mmdet_root()), env.get("PYTHONPATH", ""))
        if value
    )
    # MMEngine checkpoints contain HistoryBuffer objects. PyTorch 2.6+ defaults
    # torch.load() to weights_only=True, which rejects these trusted objects.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    subprocess.run(command, cwd=mmdet_root(), env=env, check=True)


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
    remote = f"{args.remote_prefix}/{model_name}".strip("/")
    print(f"UPLOAD {work_dir} -> hf://{args.hf_repo_type}/{args.hf_repo_id}/{remote}")
    api.upload_folder(
        folder_path=str(work_dir),
        path_in_repo=remote,
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
    )


def periodic_upload(model_name: str, args: argparse.Namespace, stop: threading.Event) -> None:
    interval = args.upload_interval_hours * 3600
    while interval > 0 and not stop.wait(interval):
        try:
            upload_work_dir_to_hf(model_name, args)
        except Exception as exc:
            print(f"PERIODIC UPLOAD FAILED {model_name}: {exc}", flush=True)


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
            args.python,
            str(mmdet_root() / "tools" / "train.py"),
            str(config_path),
            "--work-dir",
            str(work_dir),
            "--auto-scale-lr",
        ]
        if args.amp:
            command.append("--amp")
        if args.resume:
            command.append("--resume")
        stop = threading.Event()
        uploader = threading.Thread(target=periodic_upload, args=(model_name, args, stop), daemon=True)
        uploader.start()
        try:
            run(command)
        finally:
            stop.set()
            uploader.join(timeout=30)
    if args.skip_test:
        upload_work_dir_to_hf(model_name, args)
        return
    checkpoint = find_checkpoint(work_dir)
    configs = {
        "validation": config_path.with_name("patched_config_val.py"),
        "test": config_path,
    }
    annotations = {
        "validation": dataset_out / "annotations" / "val.json",
        "test": dataset_out / "annotations" / "test.json",
    }
    metrics: dict[str, Any] = {}
    for split, split_config in configs.items():
        result_dir = work_dir / "test_results" / split
        run(
            [
                args.python,
                str(mmdet_root() / "tools" / "test.py"),
                str(split_config),
                str(checkpoint),
                "--work-dir",
                str(result_dir),
                "--out",
                str(result_dir / "predictions.pkl"),
            ]
        )
        result_file = result_dir / "levir_ship.bbox.json"
        if not result_file.is_file():
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text("[]\n", encoding="utf-8")
        metric_file = result_dir / "metrics.json"
        run(
            [
                args.python,
                str(repo_root() / "evaluate_coco_metrics.py"),
                "--gt",
                str(annotations[split]),
                "--res",
                str(result_file),
                "--out",
                str(metric_file),
            ]
        )
        metrics[split] = json.loads(metric_file.read_text(encoding="utf-8"))
    final = {
        "dataset": "LEVIR-Ship",
        "model": model_name,
        "seed": args.seed,
        **metrics,
    }
    (work_dir / "final_results.json").write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )
    upload_work_dir_to_hf(model_name, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="LevirShipData")
    parser.add_argument("--dataset-out", default="mmdetection/data/levir_ship_coco")
    parser.add_argument("--work-dir", default="mmdetection/work_dirs/levir_baseline")
    parser.add_argument(
        "--models",
        default="retinanet,cascade_rcnn,rtmdet",
        help=(
            "Comma-separated: atss, retinanet, faster_rcnn, fcos, "
            "cascade_rcnn, rtmdet, detr, dino."
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Square resize used by the original baseline launcher.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--python",
        default=default_python(),
        help="Python executable used for MMDetection train/test subprocesses.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed seed for the scene-safe dataset split; independent of training seed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum images per split after scene-safe splitting; 0 uses all images.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir_ship_mmdet_runs")
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument("--remote-prefix", default="levirship_mmdet_yolo_protocol")
    parser.add_argument("--upload-interval-hours", type=float, default=1.0)
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
    print(f"Dataset split seed: {args.split_seed}; training seed: {args.seed}")
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
