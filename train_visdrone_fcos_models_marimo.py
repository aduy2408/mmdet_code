#!/usr/bin/env python3
"""Run one-seed VisDrone FCOS experiments for SET and SR-TOD via Marimo."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _candidate in (Path("/marimo/yolo_code"), ROOT.parent / "yolo_code"):
    if (_candidate / "utils").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
SEED = 42
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
CONFIGS = {
    "fcos_set": ROOT / "mmdetection/configs/set/fcos_r50_set.py",
    "fcos_srtod": ROOT / "SR-TOD/srtod_project/srtod_faster_rcnn/config/srtod-faster-rcnn_r50_fpn_1x_coco.py",
}


def convert(data_root: Path, out: Path) -> Path:
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    ann_dir = out / "annotations"
    ann_dir.mkdir(exist_ok=True)
    for split, source_name in SPLITS.items():
        source = data_root / source_name
        image_root = source / "images"
        records, annotations = [], []
        for image_id, image in enumerate(sorted(image_root.glob("*")), 1):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            with Image.open(image) as im:
                width, height = im.size
            image_dir = out / split
            image_dir.mkdir(parents=True, exist_ok=True)
            image_link = image_dir / image.name
            if not image_link.exists():
                image_link.symlink_to(image)
            records.append({"id": image_id, "file_name": image.name, "width": width, "height": height})
            annotation_file = source / "annotations" / f"{image.stem}.txt"
            if annotation_file.is_file():
                for ann_id, line in enumerate(annotation_file.read_text(encoding="utf-8").splitlines(), 1):
                    fields = line.split(",")
                    if len(fields) < 8:
                        continue
                    x, y, w, h = map(int, fields[:4])
                    score, category = map(int, fields[4:6])
                    if score <= 0 or not 1 <= category <= 10 or w <= 0 or h <= 0:
                        continue
                    annotations.append({"id": image_id * 100000 + ann_id, "image_id": image_id, "category_id": category, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
        payload = {"images": records, "annotations": annotations, "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]}
        (ann_dir / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")
    return out


def patch_cfg(config_path: Path, dataset: Path, work_dir: Path, epochs: int, batch_size: int, workers: int, seed: int = SEED):
    from mmengine.config import Config
    cfg = Config.fromfile(str(config_path), import_custom_modules=False)
    if "custom_imports" not in cfg:
        cfg.custom_imports = dict(imports=[], allow_failed_imports=False)
    imports = list(cfg.custom_imports.get("imports", []))
    if config_path.name == "fcos_r50_set.py" and "projects.set" not in imports:
        imports.append("projects.set")
    if config_path.name.startswith("srtod-"):
        imports = [item.replace("srtod_project.srtod_faster_rcnn.srtod_faster", "srtod_project.srtod_faster_rcnn.srtod_fasterrcnn") for item in imports]
    if config_path.name.startswith("srtod-"):
        imports.extend(["srtod_project.srtod_faster_rcnn.srtod_fasterrcnn", "srtod_project.srtod_faster_rcnn.srtod_datapreprocessor", "srtod_project.srtod_faster_rcnn.srtod_twostagedetector"])
    cfg.custom_imports = dict(imports=sorted(set(imports)), allow_failed_imports=False)
    if config_path.name == "fcos_r50_set.py":
        cfg.model.bbox_head.num_classes = 10
    else:
        cfg.model.roi_head.bbox_head.num_classes = 10
    for split in ("train", "val", "test"):
        loader = cfg.train_dataloader if split == "train" else cfg.val_dataloader if split == "val" else cfg.test_dataloader
        loader.dataset.type = "CocoDataset"
        loader.dataset.data_root = str(dataset)
        loader.dataset.ann_file = f"annotations/{split}.json"
        loader.dataset.data_prefix = dict(img=f"{split}/")
        loader.dataset.metainfo = dict(classes=tuple(CLASSES))
        for step in loader.dataset.pipeline:
            if step.get("type") == "Resize":
                step["scale"] = (640, 640)
    for evaluator, split in ((cfg.val_evaluator, "val"), (cfg.test_evaluator, "test")):
        evaluator.type = "CocoMetric"
        evaluator.ann_file = str(dataset / "annotations" / f"{split}.json")
        evaluator.metric = "bbox"
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = workers
    cfg.val_dataloader.num_workers = workers
    cfg.test_dataloader.num_workers = workers
    cfg.train_cfg.max_epochs = epochs
    cfg.train_cfg.val_interval = 1
    cfg.randomness = dict(seed=seed)
    cfg.work_dir = str(work_dir)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(CONFIGS), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--hf-repo-id", required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--split-seed", type=int, default=SEED)
    p.add_argument("--model-yaml", type=Path, default=None)
    args = p.parse_args()
    from utils.marimo_ops import require_training_context
    require_training_context(hf_repo_id=args.hf_repo_id)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "mmdetection"))
    dataset = convert(args.data_root.resolve(), args.dataset_root.resolve())
    run_dir = args.work_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = (args.model_yaml or CONFIGS[args.model]).resolve()
    config = patch_cfg(config_path, dataset, run_dir, args.epochs, args.batch_size, args.workers, args.seed)
    patched = run_dir / "patched_config.py"
    config.dump(str(patched))
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings
    import_modules_from_strings(config.custom_imports.get("imports", []), allow_failed_imports=False)
    runner = Runner.from_cfg(config)
    runner.train()
    checkpoint = run_dir / "latest.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"Missing checkpoint: {checkpoint}")
    runner.test()
    metrics = {"val/AP50": None, "val/mAP50-95": None, "test/AP50": None, "test/mAP50-95": None}
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest = {"experiment_id": f"visdrone-{args.model}-seed42", "baseline": {"config": str(config_path)}, "variant": {"explicit_change": "official VisDrone2019-DET data adapter; one training seed"}, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "runner": str(Path(__file__).resolve()), "python_executable": sys.executable, "dataset_root": str(args.data_root.resolve()), "dataset_yaml": str(dataset), "split_seed": args.split_seed, "training_seed": args.seed, "model/backbone/pretrained source": "FCOS-SET or SR-TOD Faster R-CNN / ResNet-50 / torchvision://resnet50", "image_size, batch_size, epochs, patience, AMP": [640, args.batch_size, args.epochs, 0, False], "NMS IoU": 0.5, "HF repo and remote prefix": [args.hf_repo_id, f"runs/{args.model}/seed_42"], "required artifacts": ["patched_config.py", "latest.pth", "evaluation_metrics.json", "experiment_manifest.json"], "upload_required": True, **metrics}
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset", exist_ok=True)
    remote = f"runs/{args.model}/seed_42"
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=args.hf_repo_id, repo_type="dataset")
    required = ["patched_config.py", "latest.pth", "evaluation_metrics.json", "experiment_manifest.json"]
    files = set(api.list_repo_files(args.hf_repo_id, repo_type="dataset"))
    expected = {f"{remote}/{item}" for item in required}
    if not expected.issubset(files):
        raise RuntimeError(f"Remote verification failed: {sorted(expected - files)}")
    marker = {"repo_id": args.hf_repo_id, "remote_prefix": remote, "verified": sorted(expected)}
    (run_dir / "upload_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(run_dir / "upload_complete.json"), path_in_repo=f"{remote}/upload_complete.json", repo_id=args.hf_repo_id, repo_type="dataset")


if __name__ == "__main__":
    main()
