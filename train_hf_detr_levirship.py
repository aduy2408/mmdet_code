#!/usr/bin/env python3
"""Train/evaluate Hugging Face ``DetrForObjectDetection`` on LEVIR-Ship.

This is intentionally separate from the MMDetection DETR runner.  It uses the
same scene-safe split and COCO annotations as ``train_all_levir_baseline.py``
but bypasses MMDetection's config, dataset wrapper, and result conversion.
That makes it useful for diagnosing a suspicious all-zero DETR result.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from statistics import fmean
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_all_levir_baseline import (
    discover_samples,
    prepare_coco_dataset,
    split_by_scene,
    yolo_boxes,
)
from hf_musgd import MuSGD
from hf_yolo_augment import YoloHFTransform


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LevirShipDataset(Dataset):
    def __init__(self, samples, processor, image_size: int, *, train: bool, mosaic: bool):
        self.samples = samples
        self.processor = processor
        self.image_size = image_size
        self.transform = YoloHFTransform(
            image_size,
            train=train,
            mosaic=mosaic,
            mosaic_close_epoch=0,
        )

    def set_epoch(self, epoch: int) -> None:
        self.transform.set_epoch(epoch)

    def _raw(self, index: int):
        image_path, annotation_path, _ = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes = yolo_boxes(annotation_path, width, height)
        labels = [0] * len(boxes)
        return image, boxes, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image, boxes, labels = self._raw(index)
        extras = []
        if self.transform._mosaic_enabled() and len(self.samples) > 1:
            choices = random.sample(range(len(self.samples)), k=min(3, len(self.samples) - 1))
            extras = [self._raw(choice) for choice in choices if choice != index]
        image, boxes, labels = self.transform(image, boxes, labels, extras)
        annotations = [
            {
                "bbox": bbox,
                "category_id": label,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            for bbox, label in zip(boxes, labels)
        ]
        encoded = self.processor(
            images=image,
            annotations={"image_id": index, "annotations": annotations},
            return_tensors="pt",
        )
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"][0],
            "image_id": index + 1,
            "original_size": (image.height, image.width),
        }


def collate_fn(batch, processor):
    heights = [item["pixel_values"].shape[-2] for item in batch]
    widths = [item["pixel_values"].shape[-1] for item in batch]
    height, width = max(heights), max(widths)
    padded_values = batch[0]["pixel_values"].new_zeros(
        (len(batch), 3, height, width)
    )
    pixel_mask = torch.zeros((len(batch), height, width), dtype=torch.bool)
    for index, item in enumerate(batch):
        item_height, item_width = item["pixel_values"].shape[-2:]
        padded_values[index, :, :item_height, :item_width] = item["pixel_values"]
        pixel_mask[index, :item_height, :item_width] = True
    return {
        "pixel_values": padded_values,
        "pixel_mask": pixel_mask,
        "labels": [item["labels"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "original_sizes": [item["original_size"] for item in batch],
    }


def move_labels(labels, device):
    return [{key: value.to(device) for key, value in item.items()} for item in labels]


@torch.no_grad()
def predict(model, processor, loader, device, score_threshold: float):
    model.eval()
    detections = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"]
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = torch.tensor(batch["original_sizes"], device=device)
        results = processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes
        )
        for image_id, result in zip(batch["image_ids"], results):
            for score, label, box in zip(
                result["scores"].cpu().tolist(),
                result["labels"].cpu().tolist(),
                result["boxes"].cpu().tolist(),
            ):
                x1, y1, x2, y2 = box
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label) + 1,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": score,
                    }
                )
    return detections


def upload_artifacts(output_dir: Path, repo_id: str, remote_prefix: str) -> list[str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for upload-required HF runs")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=remote_prefix,
        commit_message=f"Upload {remote_prefix}",
    )
    files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
    prefix = remote_prefix.rstrip("/") + "/"
    required_names = ("experiment_manifest.json", "final_results.json", "best/config.json")
    verified = [path for path in files if path.startswith(prefix)]
    missing = [name for name in required_names if prefix + name not in verified]
    if missing:
        raise RuntimeError(f"HF upload verification missing {missing} under {remote_prefix}")
    marker = {
        "repo_id": repo_id,
        "remote_prefix": remote_prefix,
        "verified": [prefix + name for name in required_names],
    }
    marker_path = output_dir / "upload_complete.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(marker_path),
        path_in_repo=prefix + "upload_complete.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Verify upload {remote_prefix}",
    )
    remote_files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
    if prefix + "upload_complete.json" not in remote_files:
        raise RuntimeError(f"HF upload marker was not verified under {remote_prefix}")
    return [prefix + name for name in required_names] + [prefix + "upload_complete.json"]


def evaluate(repo_root: Path, gt_path: Path, detections, output_dir: Path, split: str):
    result_path = output_dir / f"{split}_predictions.json"
    metric_path = output_dir / f"{split}_metrics.json"
    result_path.write_text(json.dumps(detections), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "evaluate_coco_metrics.py"),
            "--gt",
            str(gt_path),
            "--res",
            str(result_path),
            "--out",
            str(metric_path),
        ],
        check=True,
    )
    return json.loads(metric_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="LevirShipData")
    parser.add_argument("--dataset-out", default="work_dirs/hf_detr_levirship/data")
    parser.add_argument("--output-dir", default="work_dirs/hf_detr_levirship/run")
    parser.add_argument("--model-name", default="facebook/detr-resnet-50")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("musgd", "adamw"), default="musgd")
    parser.add_argument("--variant", choices=("no_mosaic", "mosaic"), default="no_mosaic")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--muon", type=float, default=0.2)
    parser.add_argument("--sgd-scale", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--hf-repo-id", default="duyle2408/detr_r18_rtdetr_r18_matrix_runs")
    parser.add_argument("--remote-prefix", default="hf_detr_levirship")
    parser.add_argument("--upload-required", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.limit = args.limit or 8
        args.num_workers = 0
    seed_everything(args.seed)
    repo_root = Path(__file__).resolve().parent
    data_root = Path(args.data_root).expanduser().resolve()
    dataset_out = Path(args.dataset_out).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(data_root)
    split_samples = split_by_scene(samples, args.split_seed)
    for split, values in split_samples.items():
        if args.limit > 0:
            split_samples[split] = values[: args.limit]
    prepare_coco_dataset(
        argparse.Namespace(
            data_root=str(data_root),
            dataset_out=str(dataset_out),
            split_seed=args.split_seed,
            limit=args.limit,
        )
    )

    from transformers import DetrForObjectDetection, DetrImageProcessor

    processor = DetrImageProcessor.from_pretrained(
        args.model_name,
        size={"shortest_edge": args.image_size, "longest_edge": args.image_size},
    )
    model_source = args.checkpoint or args.model_name
    model = DetrForObjectDetection.from_pretrained(
        model_source,
        num_labels=1,
        ignore_mismatched_sizes=True,
        id2label={0: "ship"},
        label2id={"ship": 0},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_set = LevirShipDataset(
        split_samples["train"], processor, args.image_size,
        train=True, mosaic=args.variant == "mosaic",
    )
    val_set = LevirShipDataset(
        split_samples["val"], processor, args.image_size,
        train=False, mosaic=False,
    )
    test_set = LevirShipDataset(
        split_samples["test"], processor, args.image_size,
        train=False, mosaic=False,
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda batch: collate_fn(batch, processor),
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    manifest = {
        "experiment_id": "hf_detr_levirship",
        "baseline": "unknown",
        "variant": f"Transformers DetrForObjectDetection + {args.variant} YOLO protocol",
        "source_commit": "unknown until remote checkout",
        "runner": "mmdetection/train_hf_detr_levirship.py",
        "python_executable": sys.executable,
        "dataset_root": str(data_root),
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "model": args.model_name,
        "pretrained_source": model_source,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "muon": args.muon,
        "sgd_scale": args.sgd_scale,
        "grad_clip": args.grad_clip,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "amp": args.amp,
        "nms_iou": "not applicable, DETR set-based decoding",
        "hf_repo": args.hf_repo_id,
        "remote_prefix": args.remote_prefix,
        "upload_required": args.upload_required,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    if args.optimizer == "musgd":
        optimizer = MuSGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
            muon=args.muon,
            sgd=args.sgd_scale,
        )
    else:
        backbone_params = []
        other_params = []
        for name, parameter in model.named_parameters():
            (backbone_params if "backbone" in name else other_params).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": args.learning_rate},
                {"params": backbone_params, "lr": args.backbone_learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
    best_map = -1.0
    stale = 0
    checkpoint_path = output_dir / "best"
    if not args.test_only:
        for epoch in range(1, args.epochs + 1):
            train_set.set_epoch(epoch)
            train_set.transform.mosaic_close_epoch = max(0, args.epochs - 10)
            model.train()
            losses = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"]
                if pixel_mask is not None:
                    pixel_mask = pixel_mask.to(device)
                with torch.autocast(
                    device_type=device.type,
                    enabled=args.amp and device.type == "cuda",
                ):
                    outputs = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        labels=move_labels(batch["labels"], device),
                    )
                scaler.scale(outputs.loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(outputs.loss.detach().cpu()))
            detections = predict(model, processor, val_loader, device, args.score_threshold)
            metrics = evaluate(
                repo_root,
                dataset_out / "annotations" / "val.json",
                detections,
                output_dir,
                "val",
            )
            current_map = metrics["map_50_95"]
            print(
                f"epoch={epoch} loss={fmean(losses):.4f} "
                f"val_map={current_map:.4f} val_ap50={metrics['ap50']:.4f}",
                flush=True,
            )
            if current_map > best_map or not checkpoint_path.exists():
                best_map = current_map
                stale = 0
                model.save_pretrained(checkpoint_path)
                processor.save_pretrained(checkpoint_path)
            else:
                stale += 1
                if stale >= args.patience:
                    break
    else:
        model = DetrForObjectDetection.from_pretrained(args.checkpoint or checkpoint_path)
        model.to(device)

    if checkpoint_path.is_dir() and not args.test_only:
        model = DetrForObjectDetection.from_pretrained(checkpoint_path)
        model.to(device)
    test_detections = predict(model, processor, test_loader, device, args.score_threshold)
    test_metrics = evaluate(
        repo_root,
        dataset_out / "annotations" / "test.json",
        test_detections,
        output_dir,
        "test",
    )
    result = {"dataset": "LEVIR-Ship", "model": model_source, "test": test_metrics}
    (output_dir / "final_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if args.upload_required:
        uploaded = upload_artifacts(output_dir, args.hf_repo_id, args.remote_prefix)
        result["upload"] = {
            "repo_id": args.hf_repo_id,
            "remote_prefix": args.remote_prefix,
            "verified_files": len(uploaded),
        }
        (output_dir / "final_results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        upload_artifacts(output_dir, args.hf_repo_id, args.remote_prefix)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
