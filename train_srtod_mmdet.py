#!/usr/bin/env python3
"""Train SR-TOD Varroa DGFE trials and an FCOS baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


SPLITS = ("train", "val", "test")
GT_SOURCES = ("gt_one", "gt_filtered")
CLASS_POLICIES = ("all", "only-class-1", "drop-class-3", "map-3-to-1")

MODEL_CONFIGS = {
    "srtod_faster": "srtod_project/srtod_faster_rcnn/config/srtod-faster-rcnn_r50_fpn_1x_coco.py",
    "srtod_cascade": "srtod_project/srtod_detectors/config/srtod-cascade-rcnn_r50-rfp_1x_coco.py",
    "fcos": "build/lib/mmdet/.mim/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
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


def srtod_root() -> Path:
    root = repo_root() / "SR-TOD"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing SR-TOD checkout: {root}")
    return root


def ensure_srtod_imports() -> None:
    root = str(srtod_root())
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


def fix_custom_imports(obj: Any) -> None:
    if "custom_imports" not in obj:
        return
    custom_imports = obj.get("custom_imports")
    if "imports" not in custom_imports:
        return
    imports = custom_imports.get("imports")
    if not isinstance(imports, list):
        return
    custom_imports["imports"] = [
        "srtod_project.srtod_faster_rcnn.srtod_fasterrcnn"
        if item == "srtod_project.srtod_faster_rcnn.srtod_faster"
        else item
        for item in imports
    ]


def patch_dataset_cfg(dataset: Any, dataset_out: Path, split: str) -> None:
    dataset.type = "CocoDataset"
    dataset.data_root = str(dataset_out)
    dataset.ann_file = f"annotations/{split}.json"
    dataset.data_prefix = dict(img=f"images/{split}/")
    dataset.metainfo = dict(classes=("varroa",))


def patch_evaluator(evaluator: Any, dataset_out: Path, split: str) -> None:
    evaluator.type = "CocoMetric"
    evaluator.ann_file = str(dataset_out / "annotations" / f"{split}.json")
    evaluator.metric = "bbox"
    evaluator.format_only = False


def add_photometric_distortion(train_pipeline: list[Any]) -> None:
    if any(step.get("type") == "PhotoMetricDistortion" for step in train_pipeline):
        return
    insert_at = next(
        (idx for idx, step in enumerate(train_pipeline) if step.get("type") == "RandomFlip"),
        len(train_pipeline) - 1,
    )
    train_pipeline.insert(insert_at, dict(type="PhotoMetricDistortion"))


def set_resize_scale(pipeline: list[Any], scale: tuple[int, int]) -> None:
    for step in pipeline:
        if step.get("type") == "Resize":
            step["scale"] = scale


def set_score_threshold(obj: Any, score_thr: float) -> None:
    if isinstance(obj, dict):
        if "score_thr" in obj:
            obj["score_thr"] = score_thr
        for value in obj.values():
            set_score_threshold(value, score_thr)
    elif isinstance(obj, list):
        for value in obj:
            set_score_threshold(value, score_thr)


def set_soft_nms(obj: Any, iou_thr: float, min_score: float) -> None:
    if isinstance(obj, dict):
        nms = obj.get("nms")
        if isinstance(nms, dict) and nms.get("type") == "nms":
            obj["nms"] = dict(type="soft_nms", iou_threshold=iou_thr, min_score=min_score)
        for value in obj.values():
            set_soft_nms(value, iou_thr, min_score)
    elif isinstance(obj, list):
        for value in obj:
            set_soft_nms(value, iou_thr, min_score)


def patch_config(cfg: Any, model_name: str, args: argparse.Namespace, dataset_out: Path) -> Any:
    fix_custom_imports(cfg)
    if not args.keep_pretrained_init:
        disable_pretrained(cfg.model)
    set_num_classes(cfg.model, 1)

    cfg.val_dataloader = deepcopy(cfg.val_dataloader)
    cfg.test_dataloader = deepcopy(cfg.test_dataloader)
    patch_dataset_cfg(cfg.train_dataloader.dataset, dataset_out, "train")
    patch_dataset_cfg(cfg.val_dataloader.dataset, dataset_out, "val")
    patch_dataset_cfg(cfg.test_dataloader.dataset, dataset_out, "test")
    set_resize_scale(cfg.train_dataloader.dataset.pipeline, tuple(args.img_scale))
    set_resize_scale(cfg.val_dataloader.dataset.pipeline, tuple(args.img_scale))
    set_resize_scale(cfg.test_dataloader.dataset.pipeline, tuple(args.img_scale))
    if args.photometric:
        add_photometric_distortion(cfg.train_dataloader.dataset.pipeline)
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.train_dataloader.num_workers = args.num_workers
    cfg.train_dataloader.persistent_workers = args.num_workers > 0
    cfg.val_dataloader.num_workers = args.num_workers
    cfg.val_dataloader.persistent_workers = args.num_workers > 0
    cfg.test_dataloader.num_workers = args.num_workers
    cfg.test_dataloader.persistent_workers = args.num_workers > 0
    patch_evaluator(cfg.val_evaluator, dataset_out, "val")
    patch_evaluator(cfg.test_evaluator, dataset_out, "test")
    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = args.val_interval
    if args.lr is not None:
        cfg.optim_wrapper.optimizer.lr = args.lr
    if args.weight_decay is not None:
        cfg.optim_wrapper.optimizer.weight_decay = args.weight_decay
    set_score_threshold(cfg.model, args.score_thr)
    if args.soft_nms:
        set_soft_nms(cfg.model.test_cfg, args.soft_nms_iou_thr, args.soft_nms_min_score)
    cfg.work_dir = str(resolve_path(args.work_dir) / model_name)
    if "visualizer" in cfg:
        cfg.visualizer.save_dir = str(Path(cfg.work_dir) / "vis")
    cfg.default_hooks.logger.interval = args.log_interval
    cfg.default_hooks.checkpoint.update(
        interval=args.checkpoint_interval,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=1,
        save_last=True,
    )
    if args.early_stop_patience > 0:
        cfg.custom_hooks = list(cfg.get("custom_hooks", []))
        cfg.custom_hooks.append(
            dict(
                type="EarlyStoppingHook",
                monitor="coco/bbox_mAP",
                rule="greater",
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
            )
        )
    cfg.log_processor = dict(type="LogProcessor", window_size=1, by_epoch=True)
    cfg.randomness = dict(seed=args.seed)
    return cfg


def write_patched_config(model_name: str, args: argparse.Namespace, dataset_out: Path) -> Path:
    ensure_srtod_imports()
    from mmengine.config import Config

    config_path = srtod_root() / MODEL_CONFIGS[model_name]
    cfg = Config.fromfile(str(config_path), import_custom_modules=False)
    cfg = patch_config(cfg, model_name, args, dataset_out)
    out_path = Path(cfg.work_dir) / "patched_config.py"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(out_path))
    return out_path


def build_jobs(args: argparse.Namespace) -> list[str]:
    models = list(MODEL_CONFIGS) if args.models == "default" else comma_list(args.models)
    unknown_models = sorted(set(models) - set(MODEL_CONFIGS))
    if unknown_models:
        raise ValueError(f"Unknown models: {', '.join(unknown_models)}")
    return sorted(models)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_trusted_checkpoint_command(command: list[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env["PYTHONPATH"] = f"{cwd}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def find_trained_checkpoint(work_dir: Path) -> Path:
    best_checkpoints = sorted(work_dir.glob("best_*.pth"))
    if best_checkpoints:
        return best_checkpoints[0]
    latest = work_dir / "latest.pth"
    if latest.is_file():
        return latest
    raise FileNotFoundError(f"No best_*.pth or latest.pth checkpoint found in {work_dir}")


def resolve_checkpoint_template(template: str, model_name: str) -> Path:
    return resolve_path(template.format(model=model_name))


def resolve_test_checkpoint(model_name: str, args: argparse.Namespace, work_dir: Path) -> Path:
    if not args.test_checkpoint:
        return find_trained_checkpoint(work_dir)
    checkpoint_path = resolve_checkpoint_template(args.test_checkpoint, model_name)
    if checkpoint_path.is_file():
        return checkpoint_path
    raise FileNotFoundError(f"Test checkpoint not found: {checkpoint_path}")


def run_final_test(config_path: Path, checkpoint_path: Path, work_dir: Path) -> Path:
    result_dir = work_dir / "test_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(srtod_root() / "tools" / "test.py"),
        str(config_path),
        str(checkpoint_path),
        "--work-dir",
        str(result_dir),
        "--out",
        str(result_dir / "predictions.pkl"),
    ]
    print(f"TEST {checkpoint_path} -> {result_dir}")
    run_trusted_checkpoint_command(command, cwd=srtod_root())
    return result_dir


def write_job_summary(
    model_name: str,
    config_path: Path,
    checkpoint_path: Path,
    result_dir: Path,
    work_dir: Path,
    started_at: str,
) -> Path:
    summary_path = work_dir / "job_summary.json"
    summary = {
        "model": model_name,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "test_result_dir": str(result_dir),
        "started_at": started_at,
        "finished_at": utc_now(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def upload_work_dir_to_hf(args: argparse.Namespace) -> None:
    if args.no_hf_upload:
        return
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Hugging Face upload requires --hf-token or HF_TOKEN; pass --no-hf-upload to skip.")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("Hugging Face upload requires `huggingface_hub`; install it or pass --no-hf-upload.") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo_id, repo_type=args.hf_repo_type, private=False, exist_ok=True)
    work_dir = resolve_path(args.work_dir)
    print(f"UPLOAD {work_dir} -> hf://{args.hf_repo_type}/{args.hf_repo_id}")
    api.upload_large_folder(
        folder_path=str(work_dir),
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
    )


def run_job(model_name: str, args: argparse.Namespace, dataset_out: Path) -> None:
    started_at = utc_now()
    work_dir = resolve_path(args.work_dir) / model_name
    existing_config_path = work_dir / "patched_config.py"
    if args.test_only and existing_config_path.is_file() and not args.soft_nms:
        config_path = existing_config_path
    else:
        config_path = write_patched_config(model_name, args, dataset_out)
    if args.test_only:
        checkpoint_path = resolve_test_checkpoint(model_name, args, work_dir)
        result_dir = run_final_test(config_path, checkpoint_path, work_dir)
        write_job_summary(model_name, config_path, checkpoint_path, result_dir, work_dir, started_at)
        upload_work_dir_to_hf(args)
        return

    command = [
        sys.executable,
        str(srtod_root() / "tools" / "train.py"),
        str(config_path),
        "--work-dir",
        str(work_dir),
    ]
    if args.amp:
        command.append("--amp")
    print(f"RUN {model_name} -> {work_dir}")
    run_trusted_checkpoint_command(command, cwd=srtod_root())
    checkpoint_path = find_trained_checkpoint(work_dir)
    result_dir = run_final_test(config_path, checkpoint_path, work_dir)
    write_job_summary(model_name, config_path, checkpoint_path, result_dir, work_dir, started_at)
    upload_work_dir_to_hf(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-out", default="SR-TOD/data/varroa_coco")
    parser.add_argument("--work-dir", default="SR-TOD/work_dirs/varroa_srtod")
    parser.add_argument("--gt-source", default="gt_one", choices=GT_SOURCES)
    parser.add_argument("--class-policy", default="map-3-to-1", choices=CLASS_POLICIES)
    parser.add_argument("--only-positives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0, help="Max samples per split after shuffling; 0 means full data.")
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--models", default="default", help="'default' or comma-separated model names.")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None, help="Override config optimizer lr; default keeps SR-TOD config.")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--keep-pretrained-init",
        action="store_true",
        help="Keep init_cfg/pretrained entries from the original configs. May download weights if checkpoints are URLs.",
    )
    parser.add_argument("--img-scale", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--photometric", action="store_true", help="Add PhotoMetricDistortion to the train pipeline.")
    parser.add_argument("--score-thr", type=float, default=0.001, help="Detection score threshold used during validation/test.")
    parser.add_argument("--soft-nms", action="store_true", help="Use Soft-NMS in validation/test NMS configs.")
    parser.add_argument("--soft-nms-iou-thr", type=float, default=0.5)
    parser.add_argument("--soft-nms-min-score", type=float, default=0.05)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--hf-repo-id", default="duyle2408/varroa_srtod_runs")
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument("--hf-token", default="", help="Hugging Face token. Defaults to HF_TOKEN from the environment.")
    parser.add_argument("--no-hf-upload", action="store_true", help="Skip uploading each completed job to Hugging Face.")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run final test/upload for existing job work dirs.")
    parser.add_argument("--test-checkpoint", default="", help="Optional checkpoint path/template for --test-only. Supports {model}.")
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
    for idx, model_name in assigned:
        print(f"  [{idx:02d}] {model_name}")
    if args.dry_run:
        return
    for _, model_name in assigned:
        run_job(model_name, args, dataset_out)


if __name__ == "__main__":
    main()
