#!/usr/bin/env python3
"""Train MMDetection Varroa baselines and DGFE/API variants."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


SPLITS = ("train", "val", "test")
GT_SOURCES = ("gt_one", "gt_filtered")
CLASS_POLICIES = ("all", "only-class-1", "drop-class-3", "map-3-to-1")
VARIANTS = ("base", "dgfe_api")

MODEL_CONFIGS = {
    "atss": "configs/atss/atss_r50_fpn_1x_coco.py",
    "retinanet": "configs/retinanet/retinanet_r18_fpn_1x_coco.py",
    "faster_rcnn": "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    "cascade_rcnn": "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
    "dyhead": "configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py",
    "sabl_retinanet": "configs/sabl/sabl-retinanet_r50_fpn_1x_coco.py",
    "sabl_faster": "configs/sabl/sabl-faster-rcnn_r50_fpn_1x_coco.py",
    "fsaf": "configs/fsaf/fsaf_r50_fpn_1x_coco.py",
    "gfl": "configs/gfl/gfl_r50_fpn_1x_coco.py",
    "vfnet": "configs/vfnet/vfnet_r50_fpn_1x_coco.py",
    "tood": "configs/tood/tood_r50_fpn_1x_coco.py",
    "fcos": "configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
    "reppoints": "configs/reppoints/reppoints-moment_r50_fpn-gn_head-gn_1x_coco.py",
}
MASK_MODEL_CONFIGS = {
    "mask_rcnn": "configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py",
    "htc": "configs/htc/htc_r50_fpn_1x_coco.py",
}


@dataclass(frozen=True)
class SampleRecord:
    source_split: str
    rel_image_path: str
    class_token: str
    boxes: list[list[float]]
    is_positive: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def mmdet_root() -> Path:
    root = repo_root() / "mmdetection"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing MMDetection checkout: {root}")
    return root


def ensure_mmdet_imports() -> None:
    root = str(mmdet_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (repo_root() / path).resolve()


def parse_gt_records(csv_path: Path) -> list[SampleRecord]:
    if not csv_path.exists():
        return []

    records = []
    source_split = csv_path.parent.name
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        values = [float(value) for value in parts[2:]]
        boxes = [values[idx : idx + 4] for idx in range(0, len(values) - 3, 4)]
        class_token = parts[1]
        if csv_path.name == "gt_filtered.csv" and boxes:
            class_token = "1"
        records.append(
            SampleRecord(
                source_split=source_split,
                rel_image_path=parts[0],
                class_token=class_token,
                boxes=boxes,
                is_positive=bool(boxes) and class_token != "0",
            )
        )
    return records


def load_records(data_root: Path, gt_source: str) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for split in SPLITS:
        records.extend(parse_gt_records(data_root / split / f"{gt_source}.csv"))
    return records


def filter_records(
    records: list[SampleRecord],
    *,
    only_positives: bool,
    class_policy: str,
) -> list[SampleRecord]:
    filtered = []
    for record in records:
        if only_positives and not record.is_positive:
            continue
        if class_policy == "only-class-1" and record.class_token != "1":
            continue
        if class_policy == "drop-class-3" and record.class_token == "3":
            continue
        if class_policy == "map-3-to-1" and record.class_token == "3":
            record = SampleRecord(
                source_split=record.source_split,
                rel_image_path=record.rel_image_path,
                class_token="1",
                boxes=record.boxes,
                is_positive=record.is_positive,
            )
        filtered.append(record)
    return filtered


def dedupe_records(records: list[SampleRecord]) -> list[SampleRecord]:
    seen = set()
    deduped = []
    for record in records:
        key = (record.source_split, record.rel_image_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def assign_output_splits(records: list[SampleRecord], seed: int) -> dict[str, list[SampleRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_end = round(len(shuffled) * 0.15)
    test_end = val_end + round(len(shuffled) * 0.15)
    return {
        "train": shuffled[test_end:],
        "val": shuffled[:val_end],
        "test": shuffled[val_end:test_end],
    }


def source_image_path(data_root: Path, record: SampleRecord) -> Path:
    return data_root / record.source_split / record.rel_image_path


def output_image_name(record: SampleRecord) -> str:
    return f"{record.source_split}__{'__'.join(Path(record.rel_image_path).parts[1:])}"


def clamp_box(box: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    left, right = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    top, bottom = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    if right <= left or bottom <= top:
        return None
    return [left, top, right - left, bottom - top]


def prepare_coco_dataset(args: argparse.Namespace) -> Path:
    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.dataset_out)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing data root: {data_root}")

    records = load_records(data_root, args.gt_source)
    records = filter_records(records, only_positives=args.only_positives, class_policy=args.class_policy)
    records = dedupe_records(records)
    split_records = assign_output_splits(records, args.seed)

    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}

    for split in SPLITS:
        selected = split_records[split]
        if args.limit > 0:
            selected = selected[: args.limit]
        split_counts[split] = len(selected)

        img_dir = out_dir / "images" / split
        if args.rebuild_dataset and img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)

        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        ann_id = 1
        for img_id, record in enumerate(selected, 1):
            src = source_image_path(data_root, record)
            if not src.exists():
                continue
            dst_name = output_image_name(record)
            dst = img_dir / dst_name
            if args.copy_images and (args.rebuild_dataset or not dst.exists()):
                shutil.copy2(src, dst)
            image_path = dst if args.copy_images else src
            with Image.open(image_path) as image:
                width, height = image.size
            images.append(dict(id=img_id, file_name=dst_name, width=width, height=height))
            for box in record.boxes:
                bbox = clamp_box(box, width, height)
                if bbox is None:
                    continue
                annotations.append(
                    dict(
                        id=ann_id,
                        image_id=img_id,
                        category_id=1,
                        bbox=bbox,
                        area=bbox[2] * bbox[3],
                        iscrowd=0,
                        segmentation=[],
                    )
                )
                ann_id += 1

        coco = dict(
            images=images,
            annotations=annotations,
            categories=[dict(id=1, name="varroa")],
        )
        (ann_dir / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")

    print(f"COCO dataset ready: {out_dir} ({split_counts})")
    return out_dir


def set_num_classes(obj: Any, num_classes: int = 1) -> None:
    if isinstance(obj, dict):
        if "num_classes" in obj:
            obj["num_classes"] = num_classes
        for value in obj.values():
            set_num_classes(value, num_classes)
    elif isinstance(obj, list):
        for value in obj:
            set_num_classes(value, num_classes)


def disable_pretrained(obj: Any) -> None:
    if isinstance(obj, dict):
        if "init_cfg" in obj:
            obj["init_cfg"] = None
        if "pretrained" in obj:
            obj["pretrained"] = None
        for value in obj.values():
            disable_pretrained(value)
    elif isinstance(obj, list):
        for value in obj:
            disable_pretrained(value)


def checkpoint_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "ema"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")


def candidate_checkpoint_keys(key: str, variant: str) -> list[str]:
    keys = [key]
    if variant == "dgfe_api" and key.startswith("neck."):
        keys.insert(0, f"neck.base_neck.{key[len('neck.'):]}")  # wrapped neck
    return keys


def load_compatible_checkpoint(model: Any, checkpoint_path: Path, variant: str) -> None:
    source_state = checkpoint_state_dict(checkpoint_path)
    target_state = model.state_dict()
    compatible = {}
    skipped_shape = []

    for src_key, value in source_state.items():
        for dst_key in candidate_checkpoint_keys(src_key, variant):
            if dst_key not in target_state:
                continue
            if tuple(target_state[dst_key].shape) != tuple(value.shape):
                skipped_shape.append((src_key, dst_key, tuple(value.shape), tuple(target_state[dst_key].shape)))
                continue
            compatible[dst_key] = value
            break

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded compatible checkpoint: {checkpoint_path} "
        f"matched={len(compatible)} missing_after_partial={len(missing)} "
        f"unexpected_after_partial={len(unexpected)} skipped_shape={len(skipped_shape)}"
    )
    for src_key, dst_key, src_shape, dst_shape in skipped_shape[:20]:
        print(f"  skip shape {src_key} -> {dst_key}: {src_shape} != {dst_shape}")


def wrap_dgfe_api_neck(model: Any, args: argparse.Namespace) -> None:
    if "neck" not in model:
        raise ValueError("DGFE/API variant requires model.neck")
    base_neck = deepcopy(model.neck)
    if isinstance(base_neck, list):
        out_channels = base_neck[-1].get("out_channels", base_neck[0].get("out_channels", 256))
    else:
        out_channels = base_neck.get("out_channels", 256)
    model.neck = dict(
        type="FeatureAugmentNeck",
        base_neck=base_neck,
        out_channels=out_channels,
        levels=tuple(args.dgfe_levels),
        dgfe=dict(type="FeatureDGFE", upsample_steps=args.dgfe_upsample_steps),
        api=dict(
            type="AdversarialPerturbationInjection",
            api_weight=args.api_weight,
            rho=args.api_rho,
            target_mode=args.api_target_mode,
        ),
    )


def dataset_cfg(dataset_out: Path, split: str, pipeline: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(
        type="CocoDataset",
        data_root=str(dataset_out),
        ann_file=f"annotations/{split}.json",
        data_prefix=dict(img=f"images/{split}/"),
        metainfo=dict(classes=("varroa",)),
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=pipeline,
    )


def patch_config(cfg: Any, model_name: str, variant: str, args: argparse.Namespace, dataset_out: Path) -> Any:
    if not args.keep_pretrained_init:
        disable_pretrained(cfg.model)
    set_num_classes(cfg.model, 1)
    if variant == "dgfe_api":
        wrap_dgfe_api_neck(cfg.model, args)

    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="Resize", scale=tuple(args.img_scale), keep_ratio=True),
        dict(type="PackDetInputs"),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=tuple(args.img_scale), keep_ratio=True),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="PackDetInputs"),
    ]

    cfg.train_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        sampler=dict(type="DefaultSampler", shuffle=True),
        batch_sampler=None,
        dataset=dataset_cfg(dataset_out, "train", train_pipeline),
    )
    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dataset_cfg(dataset_out, "val", test_pipeline),
    )
    cfg.test_dataloader = cfg.val_dataloader
    cfg.val_evaluator = dict(
        type="CocoMetric",
        ann_file=str(dataset_out / "annotations" / "val.json"),
        metric="bbox",
    )
    cfg.test_evaluator = cfg.val_evaluator
    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=args.epochs, val_interval=args.val_interval)
    cfg.val_cfg = dict(type="ValLoop")
    cfg.test_cfg = dict(type="TestLoop")
    cfg.optim_wrapper = dict(
        type="AmpOptimWrapper" if args.amp else "OptimWrapper",
        optimizer=dict(type="SGD", lr=args.lr, momentum=0.9, weight_decay=args.weight_decay),
    )
    cfg.param_scheduler = []
    cfg.auto_scale_lr = dict(enable=False)
    cfg.work_dir = str(resolve_path(args.work_dir) / model_name / variant)
    cfg.default_hooks.logger.interval = args.log_interval
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=args.checkpoint_interval)
    cfg.log_processor = dict(type="LogProcessor", window_size=1, by_epoch=True)
    cfg.randomness = dict(seed=args.seed)
    return cfg


def resolve_checkpoint_template(template: str, model_name: str, variant: str) -> Path:
    return resolve_path(template.format(model=model_name, variant=variant))


def compatible_checkpoint_for(model_name: str, variant: str, args: argparse.Namespace) -> Path | None:
    if not args.load_compatible_from:
        return None
    path = resolve_checkpoint_template(args.load_compatible_from, model_name, variant)
    if path.is_file():
        return path
    if "{variant}" in args.load_compatible_from and variant != "base":
        base_path = resolve_checkpoint_template(args.load_compatible_from, model_name, "base")
        if base_path.is_file():
            return base_path
    raise FileNotFoundError(f"Compatible checkpoint not found: {path}")


def build_jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    available = dict(MODEL_CONFIGS)
    if args.include_mask_models:
        raise ValueError(
            "Mask models are disabled for this bbox-only Varroa COCO export. "
            "Add mask annotations or a mask-head-disabling patch before enabling them."
        )
        available.update(MASK_MODEL_CONFIGS)

    models = list(available) if args.models == "default" else comma_list(args.models)
    variants = list(VARIANTS) if args.variants == "all" else comma_list(args.variants)
    unknown_models = sorted(set(models) - set(available))
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if unknown_models:
        raise ValueError(f"Unknown/disabled models: {', '.join(unknown_models)}")
    if unknown_variants:
        raise ValueError(f"Unknown variants: {', '.join(unknown_variants)}")
    return [(model, variant) for model in sorted(models) for variant in variants]


def run_job(model_name: str, variant: str, args: argparse.Namespace, dataset_out: Path) -> None:
    ensure_mmdet_imports()
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(str(mmdet_root() / MODEL_CONFIGS.get(model_name, MASK_MODEL_CONFIGS.get(model_name, ""))))
    cfg = patch_config(cfg, model_name, variant, args, dataset_out)
    print(f"RUN {model_name}/{variant} -> {cfg.work_dir}")
    runner = Runner.from_cfg(cfg)
    checkpoint_path = compatible_checkpoint_for(model_name, variant, args)
    if checkpoint_path is not None:
        load_compatible_checkpoint(runner.model, checkpoint_path, variant)
    runner.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-out", default="mmdetection/data/varroa_coco")
    parser.add_argument("--work-dir", default="mmdetection/work_dirs/varroa_train_all")
    parser.add_argument("--gt-source", default="gt_one", choices=GT_SOURCES)
    parser.add_argument("--class-policy", default="map-3-to-1", choices=CLASS_POLICIES)
    parser.add_argument("--only-positives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0, help="Max samples per split after shuffling; 0 means full data.")
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--models", default="default", help="'default' or comma-separated model names.")
    parser.add_argument("--variants", default="all", help="'all' or comma-separated: base,dgfe_api.")
    parser.add_argument("--include-mask-models", action="store_true")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--keep-pretrained-init",
        action="store_true",
        help="Keep init_cfg/pretrained entries from the original configs. May download weights if checkpoints are URLs.",
    )
    parser.add_argument(
        "--load-compatible-from",
        default="",
        help=(
            "Optional local checkpoint path/template loaded best-effort after model build. "
            "Supports {model} and {variant}; e.g. checkpoints/{model}/base/latest.pth. "
            "For dgfe_api, neck.* keys are also tried as neck.base_neck.*."
        ),
    )
    parser.add_argument("--img-scale", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--api-weight", type=float, default=0.01)
    parser.add_argument("--api-rho", type=float, default=0.001)
    parser.add_argument("--api-target-mode", default="foreground")
    parser.add_argument("--dgfe-levels", type=int, nargs="+", default=[0])
    parser.add_argument("--dgfe-upsample-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_machines < 1:
        raise ValueError("--num-machines must be >= 1")
    if not 0 <= args.machine_index < args.num_machines:
        raise ValueError("--machine-index must be in [0, num_machines)")

    dataset_out = prepare_coco_dataset(args)
    jobs = build_jobs(args)
    assigned = [(idx, job) for idx, job in enumerate(jobs) if idx % args.num_machines == args.machine_index]
    print(f"Jobs total={len(jobs)} assigned={len(assigned)} machine={args.machine_index}/{args.num_machines}")
    for idx, (model_name, variant) in assigned:
        print(f"  [{idx:02d}] {model_name}/{variant}")
    if args.dry_run:
        return
    for _, (model_name, variant) in assigned:
        run_job(model_name, variant, args, dataset_out)


if __name__ == "__main__":
    main()
