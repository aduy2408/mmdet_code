#!/usr/bin/env python3
"""Train Hugging Face DETR/RT-DETR on the four project datasets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from hf_dataset_adapters import DatasetBundle, DetectionSample, load_dataset
from hf_musgd import MuSGD
from hf_yolo_augment import YoloHFTransform
from train_hf_detr_levirship import upload_artifacts


MODEL_REGISTRY = {
    "detr_r50": {
        "model_id": "facebook/detr-resnet-50",
        "model_cls": "DetrForObjectDetection",
        "processor_cls": "DetrImageProcessor",
        "backbone": "ResNet-50",
    },
    "rtdetr_r18": {
        "model_id": "PekingU/rtdetr_r18vd",
        "model_cls": "RTDetrForObjectDetection",
        "processor_cls": "RTDetrImageProcessor",
        "backbone": "ResNet-18",
    },
}

DATASET_DEFAULTS = {
    "varroa": {"image_size": 640, "batch_size": 8, "workers": 8},
    "levirship": {"image_size": 512, "batch_size": 4, "workers": 4},
    "tinyperson": {"image_size": 640, "batch_size": 8, "workers": 8},
    "visdrone": {"image_size": 1536, "batch_size": 8, "workers": 8},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HFDetectionDataset(Dataset):
    def __init__(self, samples: tuple[DetectionSample, ...], processor, image_size: int, *, train: bool, mosaic: bool, num_labels: int, mosaic_close_epoch: int):
        self.samples = samples
        self.processor = processor
        self.transform = YoloHFTransform(image_size, train=train, mosaic=mosaic, mosaic_close_epoch=mosaic_close_epoch)
        self.num_labels = num_labels

    def set_epoch(self, epoch: int) -> None:
        self.transform.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _raw(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        return image, [list(box) for box in sample.boxes], list(sample.labels)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image, boxes, labels = self._raw(index)
        original_size = (image.height, image.width)
        extras = []
        if self.transform._mosaic_enabled() and len(self.samples) > 1:
            choices = random.sample(range(len(self.samples)), k=min(3, len(self.samples) - 1))
            extras = [self._raw(choice) for choice in choices if choice != index]
        image, boxes, labels = self.transform(image, boxes, labels, extras)
        annotations = [
            {"bbox": box, "category_id": int(label), "area": box[2] * box[3], "iscrowd": 0}
            for box, label in zip(boxes, labels)
        ]
        encoded = self.processor(
            images=image,
            annotations={"image_id": sample.image_id, "annotations": annotations},
            return_tensors="pt",
        )
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"][0],
            "image_id": sample.image_id,
            "original_size": original_size,
        }


def collate_fn(batch):
    height = max(item["pixel_values"].shape[-2] for item in batch)
    width = max(item["pixel_values"].shape[-1] for item in batch)
    values = batch[0]["pixel_values"].new_zeros((len(batch), 3, height, width))
    mask = torch.zeros((len(batch), height, width), dtype=torch.bool)
    for index, item in enumerate(batch):
        h, w = item["pixel_values"].shape[-2:]
        values[index, :, :h, :w] = item["pixel_values"]
        mask[index, :h, :w] = True
    return {
        "pixel_values": values,
        "pixel_mask": mask,
        "labels": [item["labels"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "original_sizes": [item["original_size"] for item in batch],
    }


def move_labels(labels, device):
    return [{key: value.to(device) for key, value in item.items()} for item in labels]


def coco_ground_truth(samples: tuple[DetectionSample, ...], label_names: tuple[str, ...]):
    images, annotations = [], []
    ann_id = 1
    for sample in samples:
        width, height = Image.open(sample.image_path).size
        images.append({"id": sample.image_id, "file_name": str(sample.image_path), "width": width, "height": height})
        for box, label in zip(sample.boxes, sample.labels):
            annotations.append({"id": ann_id, "image_id": sample.image_id, "category_id": int(label) + 1, "bbox": list(box), "area": box[2] * box[3], "iscrowd": 0})
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": [{"id": i + 1, "name": name} for i, name in enumerate(label_names)]}


@torch.no_grad()
def predict(model, processor, loader, device, threshold: float, image_size: int):
    model.eval()
    detections = []
    for batch in loader:
        outputs = model(pixel_values=batch["pixel_values"].to(device), pixel_mask=batch["pixel_mask"].to(device))
        transformed_sizes = torch.tensor([(image_size, image_size)] * len(batch["image_ids"]), device=device)
        results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=transformed_sizes)
        for image_id, original_size, result in zip(batch["image_ids"], batch["original_sizes"], results):
            original_height, original_width = original_size
            scale = min(image_size / original_width, image_size / original_height)
            for score, label, box in zip(result["scores"].cpu().tolist(), result["labels"].cpu().tolist(), result["boxes"].cpu().tolist()):
                x1, y1, x2, y2 = [float(value) / scale for value in box]
                detections.append({"image_id": int(image_id), "category_id": int(label) + 1, "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": float(score)})
    return detections


def evaluate(samples, label_names, detections, output_dir: Path, split: str):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_path = output_dir / f"{split}_ground_truth.json"
    pred_path = output_dir / f"{split}_predictions.json"
    metric_path = output_dir / f"{split}_metrics.json"
    gt_path.write_text(json.dumps(coco_ground_truth(samples, label_names)), encoding="utf-8")
    pred_path.write_text(json.dumps(detections), encoding="utf-8")
    coco_gt = COCO(str(gt_path))
    if detections:
        coco_dt = coco_gt.loadRes(str(pred_path))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = [sample.image_id for sample in samples]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        values = evaluator.stats
        metrics = {"map_50_95": float(values[0]), "ap50": float(values[1]), "ap75": float(values[2])}
    else:
        metrics = {"map_50_95": 0.0, "ap50": 0.0, "ap75": 0.0}
    metric_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), default="detr_r50")
    parser.add_argument("--variant", choices=("no_mosaic", "mosaic"), default="no_mosaic")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--optimizer", choices=("musgd", "adamw"), default="musgd")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--muon", type=float, default=0.2)
    parser.add_argument("--sgd-scale", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-repo-id", default="duyle2408/detr_r18_rtdetr_r18_matrix_runs")
    parser.add_argument("--remote-prefix", required=True)
    parser.add_argument("--upload-required", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def build_optimizer(model, args):
    if args.optimizer == "adamw":
        backbone, other = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            (backbone if "backbone" in name else other).append(parameter)
        return torch.optim.AdamW(
            [
                {"params": other, "lr": args.learning_rate},
                {"params": backbone, "lr": args.backbone_learning_rate},
            ],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    return MuSGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
        muon=args.muon,
        sgd=args.sgd_scale,
    )


def build_scheduler(optimizer, args, steps_per_epoch: int):
    if args.optimizer != "adamw":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs)), False
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = min(max(0, args.warmup_steps), total_steps)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule), True


def main():
    args = parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    if args.smoke_test:
        args.epochs, args.limit, args.workers = 1, args.limit or 2, 0
    args.image_size = args.image_size or defaults["image_size"]
    args.batch_size = args.batch_size or defaults["batch_size"]
    args.workers = defaults["workers"] if args.workers < 0 else args.workers
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset(args.dataset, args.data_root, args.split_seed, args.limit)
    model_info = MODEL_REGISTRY[args.model]
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    processor_cls = getattr(__import__("transformers", fromlist=[model_info["processor_cls"]]), model_info["processor_cls"])
    model_cls = getattr(__import__("transformers", fromlist=[model_info["model_cls"]]), model_info["model_cls"])
    processor = processor_cls.from_pretrained(model_info["model_id"], size={"shortest_edge": args.image_size, "longest_edge": args.image_size})
    id2label = {index: name for index, name in enumerate(bundle.label_names)}
    label2id = {name: index for index, name in id2label.items()}
    model = model_cls.from_pretrained(model_info["model_id"], num_labels=bundle.num_labels, ignore_mismatched_sizes=True, id2label=id2label, label2id=label2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_set = HFDetectionDataset(bundle.splits["train"], processor, args.image_size, train=True, mosaic=args.variant == "mosaic", num_labels=bundle.num_labels, mosaic_close_epoch=max(0, args.epochs - 10))
    val_set = HFDetectionDataset(bundle.splits["val"], processor, args.image_size, train=False, mosaic=False, num_labels=bundle.num_labels, mosaic_close_epoch=0)
    test_set = HFDetectionDataset(bundle.splits["test"], processor, args.image_size, train=False, mosaic=False, num_labels=bundle.num_labels, mosaic_close_epoch=0)
    loader = lambda dataset, shuffle: DataLoader(dataset, batch_size=args.batch_size if shuffle else 1, shuffle=shuffle, num_workers=args.workers, pin_memory=device.type == "cuda", collate_fn=collate_fn)
    train_loader, val_loader, test_loader = loader(train_set, True), loader(val_set, False), loader(test_set, False)
    optimizer_info = {
        "type": "AdamW" if args.optimizer == "adamw" else "MuSGD",
        "lr": args.learning_rate,
        "backbone_lr": args.backbone_learning_rate if args.optimizer == "adamw" else None,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps if args.optimizer == "adamw" else 0,
        "momentum": args.momentum if args.optimizer == "musgd" else None,
        "muon": args.muon if args.optimizer == "musgd" else None,
        "sgd": args.sgd_scale if args.optimizer == "musgd" else None,
    }
    manifest = {"experiment_id": f"hf_{args.dataset}_{args.model}_{args.variant}_seed{args.seed}", "baseline": "dataset-specific baseline protocol", "variant": args.variant, "runner": "mmdetection/train_hf_detection_matrix.py", "python_executable": sys.executable, "dataset_root": str(Path(args.data_root).resolve()), "split_seed": args.split_seed, "training_seed": args.seed, "model": model_info["model_cls"], "model_id": model_info["model_id"], "backbone": model_info["backbone"], "image_size": args.image_size, "batch_size": args.batch_size, "workers": args.workers, "epochs": args.epochs, "patience": args.patience, "optimizer": optimizer_info, "grad_clip": args.grad_clip, "amp": args.amp, "hf_repo": args.hf_repo_id, "remote_prefix": args.remote_prefix, "upload_required": args.upload_required}
    (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    optimizer = build_optimizer(model, args)
    scheduler, scheduler_per_step = build_scheduler(optimizer, args, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_map, stale = -1.0, 0
    checkpoint = output_dir / "best"
    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch)
        model.train(); losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    outputs = model(pixel_values=batch["pixel_values"].to(device), pixel_mask=batch["pixel_mask"].to(device), labels=move_labels(batch["labels"], device))
            except ValueError as exc:
                message = str(exc)
                if "must be in [x0,y0,x1,y1]" not in message and "generalized_box_iou" not in message:
                    raise
                print("nonfinite_giou=skip_batch", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            if not torch.isfinite(outputs.loss):
                print("nonfinite_loss=skip_batch", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(outputs.loss).backward(); scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(grad_norm):
                    print("nonfinite_grad_norm=skip_batch", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue
            scaler.step(optimizer); scaler.update(); losses.append(float(outputs.loss.detach().cpu()))
            if scheduler_per_step:
                scheduler.step()
        if not scheduler_per_step:
            scheduler.step()
        val_metrics = evaluate(bundle.splits["val"], bundle.label_names, predict(model, processor, val_loader, device, args.score_threshold, args.image_size), output_dir, "val")
        print(f"epoch={epoch} loss={sum(losses)/max(1,len(losses)):.4f} val_map={val_metrics['map_50_95']:.4f}", flush=True)
        if val_metrics["map_50_95"] > best_map or not checkpoint.exists():
            best_map = val_metrics["map_50_95"]; stale = 0; model.save_pretrained(checkpoint); processor.save_pretrained(checkpoint)
        else:
            stale += 1
            if stale >= args.patience: break
    model = model_cls.from_pretrained(checkpoint); model.to(device)
    test_metrics = evaluate(bundle.splits["test"], bundle.label_names, predict(model, processor, test_loader, device, args.score_threshold, args.image_size), output_dir, "test")
    result = {"dataset": args.dataset, "model": model_info["model_id"], "variant": args.variant, "seed": args.seed, "test": test_metrics}
    (output_dir / "final_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.upload_required:
        upload_artifacts(output_dir, args.hf_repo_id, args.remote_prefix)
        result["upload"] = {"repo_id": args.hf_repo_id, "remote_prefix": args.remote_prefix}
        (output_dir / "final_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
