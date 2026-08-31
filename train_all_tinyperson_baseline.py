#!/usr/bin/env python3
"""Train RetinaNet, Cascade R-CNN, and RTMDet on TinyPerson tiles."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import train_all_levir_baseline as common


MODEL_CONFIGS = {
    "atss": "configs/atss/atss_r50_fpn_1x_coco.py",
    "fcos": "configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
    "faster_rcnn": "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    "retinanet": "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
    "cascade_rcnn": "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
    "rtmdet": "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
}
TRAIN_ANN = (
    "erase_with_uncertain_dataset/annotations/corner/task/"
    "tiny_set_train_sw640_sh512_all.json"
)
TEST_ANN = "annotations/corner/task/tiny_set_test_sw640_sh512_all.json"
MERGED_TEST_ANN = "annotations/task/tiny_set_test_all.json"
MERGED_TRAIN_ANN = (
    "erase_with_uncertain_dataset/annotations/task/tiny_set_train_all.json"
)


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a trusted dataset archive after preventing path traversal."""
    destination = destination.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination)


def ensure_erased_train_images(dataset_root: Path, dry_run: bool) -> Path:
    erased_root = dataset_root / "erase_with_uncertain_dataset"
    image_dir = erased_root / "train"
    if image_dir.is_dir():
        return image_dir
    archive = erased_root / "train.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(
            f"Missing both extracted TinyPerson images and archive: {archive}"
        )
    if dry_run:
        print(f"PREPARE {archive} -> {erased_root}")
        return image_dir
    print(f"Extracting {archive} -> {erased_root}")
    safe_extract(archive, erased_root)
    return image_dir


def prepare_validation_split(
    dataset_root: Path,
    output_dir: Path,
    seed: int,
    val_ratio: float,
) -> dict[str, Path]:
    """Split by source image so overlapping tiles cannot leak across splits."""
    corner = json.loads((dataset_root / TRAIN_ANN).read_text(encoding="utf-8"))
    merged = json.loads(
        (dataset_root / MERGED_TRAIN_ANN).read_text(encoding="utf-8")
    )
    sources = sorted(corner["old_images"], key=lambda item: item["file_name"])
    rng = random.Random(seed)
    rng.shuffle(sources)
    val_count = max(1, round(len(sources) * val_ratio))
    val_names = {item["file_name"] for item in sources[:val_count]}

    def corner_subset(use_val: bool) -> dict[str, Any]:
        images = [
            item
            for item in corner["images"]
            if (item["file_name"] in val_names) == use_val
        ]
        image_ids = {item["id"] for item in images}
        return {
            key: value
            for key, value in corner.items()
            if key not in {"images", "annotations", "old_images"}
        } | {
            "images": images,
            "annotations": [
                item for item in corner["annotations"] if item["image_id"] in image_ids
            ],
            "old_images": [
                item
                for item in corner["old_images"]
                if (item["file_name"] in val_names) == use_val
            ],
        }

    merged_val_ids = {
        item["id"] for item in merged["images"] if item["file_name"] in val_names
    }
    merged_val = {
        key: value
        for key, value in merged.items()
        if key not in {"images", "annotations"}
    } | {
        "images": [item for item in merged["images"] if item["id"] in merged_val_ids],
        "annotations": [
            item for item in merged["annotations"] if item["image_id"] in merged_val_ids
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train_corner.json",
        "val": output_dir / "val_corner.json",
        "val_merged": output_dir / "val_merged.json",
    }
    for path, data in (
        (paths["train"], corner_subset(False)),
        (paths["val"], corner_subset(True)),
        (paths["val_merged"], merged_val),
    ):
        path.write_text(json.dumps(data), encoding="utf-8")
    print(
        f"TinyPerson split: {len(sources) - val_count} train source images, "
        f"{val_count} validation source images"
    )
    return paths


def pipeline(train: bool) -> list[dict[str, Any]]:
    transforms: list[dict[str, Any]] = [
        dict(type="LoadTinyPersonImageFromFile", backend_args=None),
        dict(type="LoadAnnotations", with_bbox=True),
    ]
    if train:
        transforms.append(dict(type="RandomFlip", prob=0.5))
    transforms.append(
        dict(
            type="PackDetInputs",
            meta_keys=(
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
                "corner",
            ),
        )
    )
    return transforms


def dataset_config(
    ann_file: Path,
    image_dir: Path,
    train: bool,
    limit: int,
) -> dict[str, Any]:
    dataset = dict(
        type="TinyPersonDataset",
        data_root="",
        ann_file=str(ann_file),
        data_prefix=dict(img=f"{image_dir}/"),
        metainfo=dict(classes=("person",)),
        filter_cfg=dict(filter_empty_gt=train, min_size=1),
        test_mode=not train,
        pipeline=pipeline(train),
    )
    if limit > 0:
        dataset["indices"] = limit
    return dataset


def set_max_per_image(obj: Any, maximum: int) -> None:
    if isinstance(obj, dict):
        if "max_per_img" in obj:
            obj["max_per_img"] = maximum
        for value in obj.values():
            set_max_per_image(value, maximum)
    elif isinstance(obj, list):
        for value in obj:
            set_max_per_image(value, maximum)


def patch_config(
    cfg: Any,
    model_name: str,
    args: argparse.Namespace,
    train_images: Path,
    test_images: Path,
    train_ann: Path,
    val_ann: Path,
    test_ann: Path,
) -> Any:
    common.set_num_classes(cfg.model)
    set_max_per_image(cfg.model, 200)
    # TinyPerson windows are loaded and cropped at native resolution. There is
    # no Resize transform, so predictions are already in the evaluator's tile
    # coordinates and must not be rescaled during validation or testing.
    cfg.model.test_cfg = deepcopy(cfg.model.get("test_cfg", {}))
    cfg.model.test_cfg.rescale = False
    if model_name == "retinanet":
        # TinyBenchmark starts RetinaNet anchors at 8 px on the stride-8 level.
        cfg.model.bbox_head.anchor_generator.octave_base_scale = 1
    elif model_name == "cascade_rcnn":
        # The stock scale 8 produces 32 px anchors on P2. Scale 2 starts at 8 px.
        cfg.model.rpn_head.anchor_generator.scales = [2]
    cfg.custom_imports = dict(
        imports=["projects.tinyperson_baselines"], allow_failed_imports=False
    )
    cfg.train_dataloader = deepcopy(cfg.train_dataloader)
    cfg.val_dataloader = deepcopy(cfg.val_dataloader)
    cfg.test_dataloader = deepcopy(cfg.test_dataloader)
    cfg.train_dataloader.dataset = dataset_config(
        train_ann, train_images, True, args.limit
    )
    cfg.test_dataloader.dataset = dataset_config(
        test_ann, test_images, False, args.limit
    )
    cfg.val_dataloader.dataset = dataset_config(
        val_ann, train_images, False, args.limit
    )
    for dataloader in (
        cfg.train_dataloader,
        cfg.val_dataloader,
        cfg.test_dataloader,
    ):
        dataloader.batch_size = args.batch_size if dataloader is cfg.train_dataloader else 1
        dataloader.num_workers = args.num_workers
        dataloader.persistent_workers = args.num_workers > 0

    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.pop("dynamic_intervals", None)
    cfg.train_cfg.val_interval = 1

    milestones = sorted({
        epoch
        for epoch in (
            max(1, round(args.epochs * 2 / 3)),
            max(1, round(args.epochs * 11 / 12)),
        )
        if epoch < args.epochs
    })
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
    if model_name == "rtmdet":
        cfg.custom_hooks = [
            hook for hook in cfg.get("custom_hooks", [])
            if hook.get("type") != "PipelineSwitchHook"
        ]

    work_dir = common.resolve_path(args.work_dir) / model_name
    cfg.work_dir = str(work_dir)
    cfg.default_hooks.checkpoint.update(
        interval=1,
        max_keep_ckpts=1,
        save_last=True,
        save_best="coco/bbox_mAP",
        rule="greater",
    )
    cfg.val_evaluator = dict(
        type="CocoMetric", ann_file=str(val_ann), metric="bbox"
    )
    cfg.test_evaluator = dict(
        type="CocoMetric",
        ann_file=str(test_ann),
        metric="bbox",
        format_only=False,
        outfile_prefix=str(work_dir / "test_results" / "test" / "tinyperson"),
    )
    cfg.randomness = dict(seed=args.seed)
    return cfg


def write_configs(model_name: str, args: argparse.Namespace) -> dict[str, Path]:
    dataset_root = common.resolve_path(args.data_root)
    test_ann = dataset_root / TEST_ANN
    for path in (
        dataset_root / TRAIN_ANN,
        dataset_root / MERGED_TRAIN_ANN,
        test_ann,
        dataset_root / MERGED_TEST_ANN,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    prepared = prepare_validation_split(
        dataset_root,
        common.resolve_path(args.prepared_ann_dir),
        args.split_seed,
        args.val_ratio,
    )
    train_images = ensure_erased_train_images(dataset_root, args.dry_run)
    test_images = dataset_root / "test"
    if not test_images.is_dir():
        raise FileNotFoundError(test_images)

    root = str(common.mmdet_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from mmengine.config import Config

    cfg = Config.fromfile(str(common.mmdet_root() / MODEL_CONFIGS[model_name]))
    cfg = patch_config(
        cfg,
        model_name,
        args,
        train_images,
        test_images,
        prepared["train"],
        prepared["val"],
        test_ann,
    )
    work_dir = Path(cfg.work_dir)
    output = work_dir / "patched_config.py"
    val_output = work_dir / "patched_config_val.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    val_cfg = deepcopy(cfg)
    val_cfg.test_dataloader = deepcopy(cfg.val_dataloader)
    val_cfg.test_evaluator = dict(
        type="CocoMetric",
        ann_file=str(prepared["val"]),
        metric="bbox",
        outfile_prefix=str(work_dir / "test_results" / "validation" / "tinyperson"),
    )
    val_cfg.dump(str(val_output))
    return {
        "train": output,
        "validation": val_output,
        "test": output,
        "val_corner": prepared["val"],
        "val_merged": prepared["val_merged"],
        "test_corner": test_ann,
        "test_merged": dataset_root / MERGED_TEST_ANN,
    }


def run_final_evaluation(
    model_name: str,
    split: str,
    config_paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, float]:
    work_dir = common.resolve_path(args.work_dir) / model_name
    result_dir = work_dir / "test_results" / split
    result = result_dir / "tinyperson.bbox.json"
    evaluator = common.repo_root() / "evaluate_tinyperson_metrics.py"
    if not result.is_file():
        raise FileNotFoundError(result)
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)
    prefix = "val" if split == "validation" else "test"
    corner_gt = config_paths[f"{prefix}_corner"]
    merged_gt = config_paths[f"{prefix}_merged"]
    output = result_dir / "metrics.json"
    subprocess.run(
        [
            args.python,
            str(evaluator),
            "--res",
            str(result),
            "--corner-gt",
            str(corner_gt),
            "--merged-gt",
            str(merged_gt),
            "--out",
            str(output),
        ],
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def run_job(
    model_name: str, config_paths: dict[str, Path], args: argparse.Namespace
) -> None:
    config = config_paths["train"]
    work_dir = common.resolve_path(args.work_dir) / model_name
    if not args.test_only:
        command = [
            args.python,
            str(common.mmdet_root() / "tools" / "train.py"),
            str(config),
            "--work-dir",
            str(work_dir),
            "--auto-scale-lr",
        ]
        if args.amp:
            command.append("--amp")
        if args.resume:
            command.append("--resume")
        common.run(command)
    if args.skip_test:
        return
    checkpoint = common.find_checkpoint(work_dir)
    for split in ("validation", "test"):
        result_dir = work_dir / "test_results" / split
        common.run(
            [
                args.python,
                str(common.mmdet_root() / "tools" / "test.py"),
                str(config_paths[split]),
                str(checkpoint),
                "--work-dir",
                str(result_dir),
                "--out",
                str(result_dir / "predictions.pkl"),
            ]
        )
        result_file = result_dir / "tinyperson.bbox.json"
        if not result_file.is_file():
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text("[]\n", encoding="utf-8")
    if not args.skip_final_metrics:
        final = {
            "dataset": "TinyPerson",
            "model": model_name,
            "seed": args.seed,
            "validation": run_final_evaluation(
                model_name, "validation", config_paths, args
            ),
            "test": run_final_evaluation(model_name, "test", config_paths, args),
        }
        output = work_dir / "final_results.json"
        output.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
        print(f"FINAL RESULTS {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../TinyPerson/tiny_set")
    parser.add_argument(
        "--work-dir", default="mmdetection/work_dirs/tinyperson_baseline"
    )
    parser.add_argument(
        "--prepared-ann-dir",
        default="mmdetection/data/tinyperson_baseline_seed42",
    )
    parser.add_argument(
        "--models", default="retinanet,cascade_rcnn,rtmdet"
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--python",
        default=common.default_python(),
        help="Python executable used for MMDetection and metric subprocesses.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed source-image split seed; keep constant across training seeds.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-final-metrics", action="store_true")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.num_machines < 1:
        raise ValueError("--num-machines must be >= 1")
    if not 0 <= args.machine_index < args.num_machines:
        raise ValueError("--machine-index must be in [0, num_machines)")
    models = common.comma_list(args.models)
    unknown = sorted(set(models) - set(MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")
    assigned = [
        model
        for index, model in enumerate(models)
        if index % args.num_machines == args.machine_index
    ]
    print(f"Assigned models ({args.machine_index}/{args.num_machines}): {assigned}")
    for model_name in assigned:
        config_paths = write_configs(model_name, args)
        print(f"CONFIG {model_name}: {config_paths['train']}")
        if not args.dry_run:
            run_job(model_name, config_paths, args)


if __name__ == "__main__":
    main()
