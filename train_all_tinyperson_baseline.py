#!/usr/bin/env python3
"""Train MMDetection baselines, including DETR and DINO, on TinyPerson tiles."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tarfile
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import train_all_levir_baseline as common


MODEL_CONFIGS = {
    "fcos_set": "configs/set/fcos_r50_set.py",
    "atss": "configs/atss/atss_r50_fpn_1x_coco.py",
    "fcos": "configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
    "faster_rcnn": "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    "retinanet": "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
    "cascade_rcnn": "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
    "rtmdet": "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
    "detr": "configs/detr/detr_r50_8xb2-150e_coco.py",
    "dino": "configs/dino/dino-4scale_r50_8xb2-12e_coco.py",
}
# Keep the public launcher name short while documenting that this is the
# denoising DINO-DETR implementation shipped with MMDetection.
TRAIN_ANN = (
    "erase_with_uncertain_dataset/annotations/corner/task/"
    "tiny_set_train_sw640_sh512_all.json"
)
TEST_ANN = "annotations/corner/task/tiny_set_test_sw640_sh512_all.json"
MERGED_TEST_ANN = "annotations/task/tiny_set_test_all.json"
MERGED_TRAIN_ANN = (
    "erase_with_uncertain_dataset/annotations/task/tiny_set_train_all.json"
)


def pipeline(train: bool, image_size: int = 640) -> list[dict[str, Any]]:
    """Return the complete TinyPerson pipeline for inspection and reuse."""
    return [
        dict(type="LoadTinyPersonImageFromFile", backend_args=None),
        dict(type="LoadAnnotations", with_bbox=True),
    ] + common.yolo_pipeline(image_size, train=train)


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


def dataset_config(
    ann_file: Path,
    image_dir: Path,
    train: bool,
    limit: int,
    image_size: int = 640,
    mosaic: bool = True,
) -> dict[str, Any]:
    base = dict(
        type="TinyPersonDataset",
        data_root="",
        ann_file=str(ann_file),
        data_prefix=dict(img=f"{image_dir}/"),
        metainfo=dict(classes=("person",)),
        filter_cfg=dict(filter_empty_gt=train, min_size=1),
        test_mode=not train,
        pipeline=[
            dict(type="LoadTinyPersonImageFromFile", backend_args=None),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
    )
    if limit > 0:
        base["indices"] = limit
    if not train:
        base["pipeline"] = base["pipeline"] + common.yolo_pipeline(
            image_size, train=False
        )
        return base
    return dict(
        type="MultiImageMixDataset",
        dataset=base,
        pipeline=common.yolo_pipeline(image_size, train=True, mosaic=mosaic),
    )


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
    use_mosaic = args.variant == "mosaic"
    common.set_num_classes(cfg.model)
    # MMDetection 3.3.0 requires both keys when tools/train.py receives
    # --auto-scale-lr. Stock DETR/DINO configs only declare base_batch_size.
    cfg.auto_scale_lr = deepcopy(cfg.get("auto_scale_lr", {}))
    cfg.auto_scale_lr.enable = False
    cfg.auto_scale_lr.setdefault("base_batch_size", 16)
    # DETR has only 100 object queries by default, so asking top-k for 200
    # predictions makes MMDetection fail during validation. DINO has 900
    # queries and keeps the larger TinyPerson cap.
    max_per_image = min(200, int(cfg.model.get("num_queries", 200)))
    set_max_per_image(cfg.model, max_per_image)
    # Keep evaluator coordinates consistent with the resized/padded tile path.
    cfg.model.test_cfg = deepcopy(cfg.model.get("test_cfg", {}))
    cfg.model.test_cfg.rescale = False
    if model_name == "rtmdet":
        # Edge windows can have dimensions smaller than the nominal tile size.
        # Pad batches to a stride-compatible shape before CSPNeXt/PAN-FPN.
        cfg.model.data_preprocessor.pad_size_divisor = 32
    if model_name == "retinanet":
        # TinyBenchmark starts RetinaNet anchors at 8 px on the stride-8 level.
        cfg.model.bbox_head.anchor_generator.octave_base_scale = 1
    elif model_name == "cascade_rcnn":
        # The stock scale 8 produces 32 px anchors on P2. Scale 2 starts at 8 px.
        cfg.model.rpn_head.anchor_generator.scales = [2]
    cfg.custom_imports = dict(
        imports=["projects.tinyperson_baselines", "projects.set", "mmdet.engine.hooks"],
        allow_failed_imports=False,
    )
    cfg.train_dataloader = deepcopy(cfg.train_dataloader)
    cfg.val_dataloader = deepcopy(cfg.val_dataloader)
    cfg.test_dataloader = deepcopy(cfg.test_dataloader)
    cfg.train_dataloader.dataset = dataset_config(
        train_ann, train_images, True, args.limit, args.image_size, use_mosaic
    )
    cfg.test_dataloader.dataset = dataset_config(
        test_ann, test_images, False, args.limit, args.image_size
    )
    cfg.val_dataloader.dataset = dataset_config(
        val_ann, train_images, False, args.limit, args.image_size
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
    cfg.custom_hooks = [
        hook for hook in cfg.get("custom_hooks", [])
        if hook.get("type") != "PipelineSwitchHook"
    ]
    if use_mosaic:
        cfg.custom_hooks.append(dict(
            type="PipelineSwitchHook",
            switch_epoch=max(0, args.epochs - 10),
            switch_pipeline=common.yolo_pipeline(
                args.image_size, train=True, mosaic=False
            ),
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
        stop = threading.Event()
        uploader = threading.Thread(
            target=periodic_upload, args=(model_name, args, stop), daemon=True
        )
        uploader.start()
        try:
            common.run(command)
        finally:
            stop.set()
            uploader.join(timeout=30)
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
        upload_work_dir_to_hf(model_name, args)


def upload_work_dir_to_hf(model_name: str, args: argparse.Namespace) -> None:
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("TinyPerson upload requires HF_TOKEN")
    from huggingface_hub import HfApi
    work_dir = common.resolve_path(args.work_dir) / model_name
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.hf_repo_id, repo_type=args.hf_repo_type,
        private=False, exist_ok=True,
    )
    remote = f"{args.remote_prefix}/{model_name}".strip("/")
    print(f"UPLOAD {work_dir} -> hf://{args.hf_repo_type}/{args.hf_repo_id}/{remote}")
    api.upload_folder(
        folder_path=str(work_dir), path_in_repo=remote,
        repo_id=args.hf_repo_id, repo_type=args.hf_repo_type,
    )
    files = api.list_repo_files(repo_id=args.hf_repo_id, repo_type=args.hf_repo_type)
    if not any(path.startswith(remote + "/") for path in files):
        raise FileNotFoundError(f"HF remote prefix not found: {remote}")


def periodic_upload(model_name: str, args: argparse.Namespace, stop: threading.Event) -> None:
    interval = args.upload_interval_hours * 3600
    while interval > 0 and not stop.wait(interval):
        try:
            upload_work_dir_to_hf(model_name, args)
        except Exception as exc:
            print(f"PERIODIC UPLOAD FAILED {model_name}: {exc}", flush=True)


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
        "--models",
        default="retinanet,cascade_rcnn,rtmdet",
        help=(
            "Comma-separated: atss, fcos, faster_rcnn, retinanet, "
            "cascade_rcnn, rtmdet, detr, dino."
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--variant", choices=("mosaic", "no_mosaic"), default="mosaic")
    parser.add_argument(
        "--image-size", type=int, default=640,
        help="Square YOLO-style training size used by Mosaic/Resize.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--model-yaml", default="explicit MMDetection registry")
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
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed model and continue with the remaining models.",
    )
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument("--hf-repo-id", default="duyle2408/set_fcos_runs")
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--remote-prefix", default="tinyperson_mmdet_yolo_protocol")
    parser.add_argument("--upload-interval-hours", type=float, default=1.0)
    parser.set_defaults(no_hf_upload=False)
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
        try:
            config_paths = write_configs(model_name, args)
            print(f"CONFIG {model_name}: {config_paths['train']}")
            if not args.dry_run:
                run_job(model_name, config_paths, args)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            work_dir = common.resolve_path(args.work_dir) / model_name
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "failure.json").write_text(
                json.dumps({"model": model_name, "seed": args.seed, "error": repr(exc)}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            print(f"MODEL FAILED, CONTINUING: {model_name}: {exc}", flush=True)


if __name__ == "__main__":
    main()
