#!/usr/bin/env python3
"""Train matched MMDetection baselines on VisDrone2019-DET.

The launcher keeps the official train/val/test-dev split and patches only the
common dataset and training protocol around explicit MMDetection configs.
Always run ``--dry-run`` first. Full jobs are intended for the Marimo
MMDetection environment, not the notebook Python.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MMDET_ROOT = ROOT / "mmdetection"
PYTHON_DEFAULT = "/marimo/mmdet-venv/bin/python"
CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)

# Explicit registry. Do not infer architecture from checkpoint or output names.
MODEL_CONFIGS = {
    "fcos": "configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
    "faster_rcnn": "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    "atss": "configs/atss/atss_r50_fpn_1x_coco.py",
    "cascade_rcnn": "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
    "rtmdet": "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
    "retinanet": "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=os.environ.get("MMDET_PYTHON", PYTHON_DEFAULT))
    parser.add_argument("--data-root", default="/marimo/VisDrone2019")
    parser.add_argument("--dataset-out", default="mmdetection/data/visdrone2019_coco")
    parser.add_argument("--work-dir", default="mmdetection/work_dirs/visdrone2019_baselines")
    parser.add_argument("--models", default=",".join(MODEL_CONFIGS))
    parser.add_argument("--model-yaml", default="", help="Explicit canonical config path for Marimo contract validation.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", "--patience", dest="patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", "--num-workers", dest="workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=(640, 640), metavar=("W", "H"))
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed protocol seed recorded for the official split.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--rebuild-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--hf-repo-id", required=False, default="")
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument("--remote-prefix", default="visdrone2019_mmdet_baselines")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def ensure_mmdet_imports() -> None:
    """Load the repository compatibility shim before importing MMDetection."""
    compat = ROOT / "blackwell_compat"
    shim_path = compat / "sitecustomize.py"
    if shim_path.is_file():
        spec = importlib.util.spec_from_file_location("_varroa_blackwell_compat", shim_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load compatibility shim: {shim_path}")
        shim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shim)
    for path in (ROOT, MMDET_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def visdrone_source(data_root: Path, split: str) -> Path:
    names = {
        "train": "VisDrone2019-DET-train",
        "val": "VisDrone2019-DET-val",
        "test": "VisDrone2019-DET-test-dev",
    }
    path = data_root / names[split]
    if not (path / "images").is_dir() or not (path / "annotations").is_dir():
        raise FileNotFoundError(f"Expected VisDrone images/annotations under {path}")
    return path


def convert_split(data_root: Path, output: Path, split: str, copy_images: bool) -> dict[str, Any]:
    """Convert official VisDrone text labels to COCO, preserving ignored boxes."""
    from PIL import Image

    source = visdrone_source(data_root, split)
    image_out = output / "images" / split
    image_out.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    for image_id, ann_path in enumerate(sorted((source / "annotations").glob("*.txt")), start=1):
        image_path = source / "images" / f"{ann_path.stem}.jpg"
        if not image_path.is_file():
            image_path = source / "images" / f"{ann_path.stem}.png"
        if not image_path.is_file():
            continue
        destination = image_out / image_path.name
        if copy_images:
            if not destination.exists():
                shutil.copy2(image_path, destination)
            file_name = image_path.name
        else:
            file_name = str(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

        for line in ann_path.read_text(encoding="utf-8").splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) < 6:
                continue
            x, y, box_w, box_h, score, category = map(int, values[:6])
            # VisDrone score 0, category 0, and category 11 are ignored.
            if score == 0 or category not in range(1, 11):
                continue
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            box_w = max(0, min(box_w, width - x))
            box_h = max(0, min(box_h, height - y))
            if box_w == 0 or box_h == 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(CLASSES, start=1)],
    }


def prepare_dataset(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root).expanduser().resolve()
    output = Path(args.dataset_out).expanduser()
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    if args.rebuild_dataset and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    ann_dir = output / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        coco = convert_split(data_root, output, split, args.copy_images)
        (ann_dir / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")
        print(f"DATASET {split}: images={len(coco['images'])} boxes={len(coco['annotations'])}")
    (output / "visdrone2019.yaml").write_text(
        "path: " + str(output) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n",
        encoding="utf-8",
    )
    return output


def patch_dataset(dataset: Any, dataset_out: Path, split: str) -> None:
    if isinstance(dataset, dict) and dataset.get("type") == "MultiImageMixDataset":
        patch_dataset(dataset["dataset"], dataset_out, split)
        return
    if not isinstance(dataset, dict):
        raise TypeError(f"Unsupported dataset config for {split}: {type(dataset).__name__}")
    dataset["type"] = "CocoDataset"
    dataset["ann_file"] = str(dataset_out / "annotations" / f"{split}.json")
    dataset["data_prefix"] = {"img": ""}
    dataset["metainfo"] = {"classes": CLASSES}


def replace_resize(obj: Any, scale: tuple[int, int]) -> None:
    if isinstance(obj, list):
        for value in obj:
            replace_resize(value, scale)
    elif isinstance(obj, dict):
        if obj.get("type") in {"Resize", "RandomResize"}:
            obj["scale"] = scale
            obj.pop("scales", None)
            obj["keep_ratio"] = True
        for value in obj.values():
            replace_resize(value, scale)


def set_num_classes(obj: Any) -> None:
    if isinstance(obj, dict):
        if "num_classes" in obj:
            obj["num_classes"] = len(CLASSES)
        for value in obj.values():
            set_num_classes(value)
    elif isinstance(obj, list):
        for value in obj:
            set_num_classes(value)


def set_nms_iou(obj: Any, threshold: float) -> None:
    if isinstance(obj, dict):
        nms = obj.get("nms")
        if isinstance(nms, dict) and nms.get("type") == "nms":
            nms["iou_threshold"] = threshold
        for value in obj.values():
            set_nms_iou(value, threshold)
    elif isinstance(obj, list):
        for value in obj:
            set_nms_iou(value, threshold)


def patch_config(cfg: Any, model: str, args: argparse.Namespace, dataset_out: Path, work_dir: Path) -> Any:
    cfg.custom_imports = {"imports": ["projects.set", "mmdet.engine.hooks"], "allow_failed_imports": False}
    cfg.model = deepcopy(cfg.model)
    set_num_classes(cfg.model)
    set_nms_iou(cfg.model, args.nms_iou)
    cfg.train_dataloader = deepcopy(cfg.train_dataloader)
    cfg.val_dataloader = deepcopy(cfg.val_dataloader)
    cfg.test_dataloader = deepcopy(cfg.test_dataloader)
    patch_dataset(cfg.train_dataloader.dataset, dataset_out, "train")
    patch_dataset(cfg.val_dataloader.dataset, dataset_out, "val")
    patch_dataset(cfg.test_dataloader.dataset, dataset_out, "test")
    for dataloader in (cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        dataloader.num_workers = args.workers
        dataloader.persistent_workers = args.workers > 0
        replace_resize(dataloader.dataset.pipeline, tuple(args.image_size))
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.val_evaluator.ann_file = str(dataset_out / "annotations" / "val.json")
    cfg.test_evaluator.ann_file = str(dataset_out / "annotations" / "test.json")
    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=args.epochs, val_interval=1)
    cfg.val_cfg = dict(type="ValLoop")
    cfg.test_cfg = dict(type="TestLoop")
    cfg.optim_wrapper = dict(
        type="OptimWrapper",
        optimizer=dict(
            type="MuSGD", lr=args.lr, momentum=0.9, nesterov=True,
            weight_decay=0.0005, muon=0.2, sgd=1.0,
        ),
        clip_grad=dict(max_norm=35, norm_type=2),
    )
    cfg.param_scheduler = [
        dict(type="LinearLR", start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
        dict(type="CosineAnnealingLR", T_max=args.epochs, by_epoch=True, begin=0, end=args.epochs),
    ]
    cfg.custom_hooks = list(cfg.get("custom_hooks", [])) + [
        dict(type="EarlyStoppingHook", monitor="coco/bbox_mAP", rule="greater", patience=args.patience, min_delta=0.001)
    ]
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook", interval=args.checkpoint_interval, save_best="coco/bbox_mAP",
        rule="greater", max_keep_ckpts=1, save_last=True,
    )
    cfg.default_hooks.logger.interval = args.log_interval
    cfg.randomness = dict(seed=args.seed)
    cfg.work_dir = str(work_dir)
    return cfg


def write_config(model: str, args: argparse.Namespace, dataset_out: Path, work_dir: Path) -> Path:
    ensure_mmdet_imports()
    from mmengine.config import Config
    from mmdet.utils import register_all_modules

    register_all_modules()
    config_path = MMDET_ROOT / MODEL_CONFIGS[model]
    cfg = Config.fromfile(str(config_path))
    cfg = patch_config(cfg, model, args, dataset_out, work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    patched = work_dir / "patched_config.py"
    cfg.dump(str(patched))
    resolved_model = {"type": cfg.model.get("type")}
    for key in ("backbone", "neck", "bbox_head"):
        component = cfg.model.get(key)
        if isinstance(component, dict):
            resolved_model[key] = component.get("type")
    manifest = {
        "experiment_id": f"visdrone2019_{model}_seed{args.seed}",
        "baseline": {"control_config_or_baseline_config": str(config_path), "model": model},
        "variant": {"variant_config_or_explicit_change": "VisDrone COCO adapter + common MuSGD protocol", "patched_config": str(patched)},
        "source_commit": source_commit(),
        "runner": str(Path(__file__).resolve()),
        "python_executable": args.python,
        "dataset_root": str(Path(args.data_root).expanduser().resolve()),
        "split": "official VisDrone2019-DET train/val/test-dev",
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "model_backbone_pretrained_source": str(config_path),
        "resolved_model_components": resolved_model,
        "image_size": list(args.image_size), "batch_size": args.batch_size,
        "epochs": args.epochs, "patience": args.patience, "amp": args.amp,
        "optimizer": {"type": "MuSGD", "lr": args.lr}, "workers": args.workers,
        "nms_iou": args.nms_iou, "hf_repo": args.hf_repo_id or "unknown",
        "remote_prefix": f"{args.remote_prefix}/{model}/seed{args.seed}",
        "required_artifacts": ["patched_config.py", "experiment_manifest.json", "checkpoint", "test_results"],
        "upload_required": True,
    }
    (work_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return patched


def upload(work_dir: Path, args: argparse.Namespace, model: str) -> None:
    if not args.hf_repo_id:
        raise ValueError("--hf-repo-id is required for upload-required VisDrone runs")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required in the detached Marimo child environment")
    from huggingface_hub import HfApi

    prefix = f"{args.remote_prefix}/{model}/seed{args.seed}"
    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo_id, repo_type=args.hf_repo_type, exist_ok=True)
    api.upload_folder(folder_path=str(work_dir), path_in_repo=prefix, repo_id=args.hf_repo_id, repo_type=args.hf_repo_type)
    files = api.list_repo_files(repo_id=args.hf_repo_id, repo_type=args.hf_repo_type)
    if not any(path.startswith(prefix + "/") for path in files):
        raise RuntimeError(f"Upload verifier did not find remote prefix {prefix}")
    marker = work_dir / "upload_complete.json"
    marker.write_text(
        json.dumps({"repo_id": args.hf_repo_id, "remote_prefix": prefix, "verified": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=f"{prefix}/upload_complete.json",
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
    )
    print(f"UPLOAD VERIFIED hf://{args.hf_repo_type}/{args.hf_repo_id}/{prefix}")


def validate_upload_requirements(args: argparse.Namespace) -> None:
    """Fail before training when the required remote artifact path is unavailable."""
    if not args.hf_repo_id:
        raise ValueError("--hf-repo-id is required for upload-required VisDrone runs")
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN is required in the detached Marimo child environment")
    from huggingface_hub import HfApi

    HfApi(token=os.environ["HF_TOKEN"]).create_repo(
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        exist_ok=True,
    )


def run_model(model: str, args: argparse.Namespace, dataset_out: Path) -> None:
    work_dir = Path(args.work_dir).expanduser()
    if not work_dir.is_absolute():
        work_dir = (ROOT / work_dir).resolve()
    work_dir = work_dir / model / f"seed{args.seed}"
    config_path = write_config(model, args, dataset_out, work_dir)
    command = [
        args.python, str(MMDET_ROOT / "tools" / "train.py"), str(config_path), "--work-dir", str(work_dir),
    ]
    if args.amp:
        command.append("--amp")
    print("RUN", " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=MMDET_ROOT, check=True)
    checkpoints = sorted(work_dir.glob("best_*.pth")) or [work_dir / "latest.pth"]
    checkpoint = checkpoints[0]
    result_dir = work_dir / "test_results"
    test_env = os.environ.copy()
    # MMEngine 0.10.7 checkpoints contain trusted HistoryBuffer objects.
    # PyTorch 2.6 defaults torch.load to weights_only=True, which prevents
    # MMDetection's test.py from loading these locally generated checkpoints.
    test_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    subprocess.run([
        args.python, str(MMDET_ROOT / "tools" / "test.py"), str(config_path), str(checkpoint),
        "--work-dir", str(result_dir), "--out", str(result_dir / "predictions.pkl"),
    ], cwd=MMDET_ROOT, check=True, env=test_env)
    (work_dir / "completion.json").write_text(
        json.dumps({"finished_at": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint)}),
        encoding="utf-8",
    )
    upload(work_dir, args, model)


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.models.split(",")) - set(MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")
    if args.epochs <= 0 or args.patience < 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs, batch-size must be positive and patience/workers cannot be negative")
    if not args.dry_run:
        sys.path.insert(0, str(ROOT))
        from utils.marimo_ops import require_training_context

        require_training_context(hf_repo_id=args.hf_repo_id)
        validate_upload_requirements(args)
    dataset_out = prepare_dataset(args)
    print(json.dumps({
        "control": {model: MODEL_CONFIGS[model] for model in args.models.split(",")},
        "variant": "official VisDrone split + MuSGD + common runtime settings",
        "source_commit": source_commit(), "dataset_root": str(Path(args.data_root).expanduser().resolve()),
        "split_seed": args.split_seed, "training_seed": args.seed,
        "epochs": args.epochs, "patience": args.patience, "lr": args.lr,
        "batch_size": args.batch_size, "workers": args.workers, "amp": args.amp,
        "nms_iou": args.nms_iou, "upload_required": True,
    }, indent=2))
    for model in args.models.split(","):
        run_model(model, args, dataset_out)


if __name__ == "__main__":
    main()
