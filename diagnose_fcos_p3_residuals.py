#!/usr/bin/env python3
"""Frozen FCOS/P3 residual diagnostic for LEVIR-Ship.

This is deliberately an offline probe: it never trains or modifies the model.
Fit artifacts are learned from the train split, then frozen for validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EPS = 1e-8
MAP_NAMES = (
    "feature_norm",
    "local_contrast",
    "spatial",
    "channel",
    "agreement",
    "energy",
    "hardness",
    "minimum",
    "harmonic",
)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def ring_predict(feature: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Channel-wise ring average with a zero center."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("ring kernel size must be odd and >= 3")
    channels = feature.shape[1]
    kernel = feature.new_ones(channels, 1, kernel_size, kernel_size)
    kernel[:, :, kernel_size // 2, kernel_size // 2] = 0
    kernel /= kernel_size * kernel_size - 1
    return F.conv2d(
        feature, kernel, padding=kernel_size // 2, groups=channels
    )


def grouped_leave_one_out(feature: torch.Tensor, group_size: int) -> torch.Tensor:
    """Predict each channel from the other channels in its contiguous group."""
    _, channels, _, _ = feature.shape
    if group_size < 2 or channels % group_size:
        raise ValueError("group_size must divide channels and be >= 2")
    grouped = feature.reshape(
        feature.shape[0], channels // group_size, group_size, *feature.shape[2:]
    )
    prediction = (grouped.sum(2, keepdim=True) - grouped) / (group_size - 1)
    return prediction.reshape_as(feature)


def pca_reconstruct(
    feature: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    """Reconstruct NCHW features from a channel PCA basis."""
    vectors = feature.permute(0, 2, 3, 1)
    centered = vectors - mean
    return (
        mean + torch.matmul(torch.matmul(centered, basis), basis.T)
    ).permute(0, 3, 1, 2)


def residual_maps(
    feature: torch.Tensor,
    spatial_prediction: torch.Tensor,
    channel_prediction: torch.Tensor,
    spatial_stats: tuple[float, float],
    channel_stats: tuple[float, float],
) -> dict[str, torch.Tensor]:
    rs = feature.float() - spatial_prediction.float()
    rc = feature.float() - channel_prediction.float()
    es = rs.abs().mean(1)
    ec = rc.abs().mean(1)
    es_norm = torch.sigmoid(
        (es - spatial_stats[0]) / (1.4826 * spatial_stats[1] + EPS)
    )
    ec_norm = torch.sigmoid(
        (ec - channel_stats[0]) / (1.4826 * channel_stats[1] + EPS)
    )
    agreement = F.relu(
        (rs * rc).sum(1)
        / (torch.linalg.vector_norm(rs, dim=1)
           * torch.linalg.vector_norm(rc, dim=1) + EPS)
    )
    energy = torch.sqrt(es_norm * ec_norm)
    hardness = agreement * energy
    minimum = torch.minimum(es_norm, ec_norm)
    harmonic = 2 * es_norm * ec_norm / (es_norm + ec_norm + EPS)
    local = torch.linalg.vector_norm(
        feature.float() - F.avg_pool2d(feature.float(), 3, 1, 1), dim=1
    )
    return {
        "feature_norm": torch.linalg.vector_norm(feature.float(), dim=1),
        "local_contrast": local,
        "spatial": es,
        "channel": ec,
        "agreement": agreement,
        "energy": energy,
        "hardness": hardness,
        "minimum": minimum,
        "harmonic": harmonic,
    }


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(mad, EPS)


def softmax_centroid(
    heatmap: torch.Tensor, center_xy: tuple[float, float], window: int, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return peak and centroid in feature coordinates."""
    height, width = heatmap.shape
    radius = window // 2
    cx = int(round(center_xy[0]))
    cy = int(round(center_xy[1]))
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    patch = heatmap[y0:y1, x0:x1].float()
    flat_peak = int(patch.argmax())
    py, px = np.unravel_index(flat_peak, patch.shape)
    peak = np.array([x0 + px, y0 + py], dtype=np.float64)
    weights = torch.softmax(patch.flatten() / tau, dim=0).reshape_as(patch)
    ys, xs = torch.meshgrid(
        torch.arange(y0, y1, device=heatmap.device),
        torch.arange(x0, x1, device=heatmap.device),
        indexing="ij",
    )
    centroid = np.array(
        [(weights * xs).sum().item(), (weights * ys).sum().item()]
    )
    return peak, centroid


def size_bin(box: np.ndarray) -> str:
    radius = math.sqrt(max(0.0, (box[2] - box[0]) * (box[3] - box[1])))
    if radius <= 8:
        return "tiny-1"
    if radius <= 16:
        return "tiny-2"
    if radius <= 32:
        return "small"
    return "larger"


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(rb - lt, 0), axis=1)
    area_a = np.prod(np.maximum(box[2:] - box[:2], 0))
    area_b = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return intersection / (area_a + area_b - intersection + EPS)


def gt_metrics(
    box: np.ndarray, pred_boxes: np.ndarray, pred_scores: np.ndarray
) -> dict[str, float | bool]:
    ious = box_iou(box, pred_boxes)
    if not len(ious):
        return {
            "baseline_score": 0.0,
            "baseline_iou": 0.0,
            "baseline_center_error": float("nan"),
            "is_missed": True,
            "candidate_x": float("nan"),
            "candidate_y": float("nan"),
        }
    index = int(ious.argmax())
    pred = pred_boxes[index]
    gt_center = (box[:2] + box[2:]) / 2
    pred_center = (pred[:2] + pred[2:]) / 2
    scale = math.sqrt(
        max(EPS, (box[2] - box[0]) * (box[3] - box[1]))
    )
    return {
        "baseline_score": float(pred_scores[index]),
        "baseline_iou": float(ious[index]),
        "baseline_center_error": float(np.linalg.norm(pred_center - gt_center) / scale),
        "is_missed": bool(ious[index] < 0.5),
        "candidate_x": float(pred_center[0]),
        "candidate_y": float(pred_center[1]),
    }


def raster_bounds(
    box: np.ndarray, scale_x: float, scale_y: float, width: int, height: int
) -> tuple[int, int, int, int]:
    x0 = max(0, min(width - 1, int(math.floor(box[0] * scale_x))))
    y0 = max(0, min(height - 1, int(math.floor(box[1] * scale_y))))
    x1 = max(x0 + 1, min(width, int(math.ceil(box[2] * scale_x))))
    y1 = max(y0 + 1, min(height, int(math.ceil(box[3] * scale_y))))
    return x0, y0, x1, y1


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    n_pos, n_neg = positives.sum(), negatives.sum()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float(precision[y == 1].sum() / positives)


def bootstrap_ci(
    values: np.ndarray, statistic=np.median, samples: int = 1000, seed: int = 42
) -> list[float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    estimates = [
        statistic(values[rng.integers(0, len(values), len(values))])
        for _ in range(samples)
    ]
    return [float(x) for x in np.percentile(estimates, [2.5, 97.5])]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path.with_suffix(".parquet"), index=False)
    except (ImportError, ModuleNotFoundError):
        pass


class Diagnostic:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(__file__).resolve().parent
        self.mmdet_root = self.root / "mmdetection"
        sys.path.insert(0, str(self.mmdet_root))
        from mmengine.config import Config
        from mmdet.apis import init_detector
        from mmdet.datasets.transforms import PackDetInputs  # noqa: F401
        from mmdet.registry import TRANSFORMS
        from mmcv.transforms import Compose

        self.cfg = Config.fromfile(str(args.config))
        self.device = torch.device(args.device)
        self.model = init_detector(self.cfg, str(args.checkpoint), device=args.device)
        self.model.eval()
        self.model.requires_grad_(False)
        pipeline = list(self.cfg.test_dataloader.dataset.pipeline)
        self.pipeline = Compose([TRANSFORMS.build(step) for step in pipeline])
        self.annotations = {
            split: json.loads(Path(path).read_text(encoding="utf-8"))
            for split, path in (
                ("train", args.train_annotations),
                ("val", args.val_annotations),
            )
        }
        self.images = {
            split: {item["id"]: item for item in data["images"]}
            for split, data in self.annotations.items()
        }
        self.boxes = {}
        for split, data in self.annotations.items():
            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for annotation in data["annotations"]:
                grouped[annotation["image_id"]].append(annotation)
            self.boxes[split] = grouped
        self.image_root = Path(args.image_root)
        self.rng = np.random.default_rng(args.seed)

    def sample(self, split: str, image_id: int) -> tuple[torch.Tensor, Any, np.ndarray]:
        item = self.images[split][image_id]
        packed = self.pipeline(
            {
                "img_id": image_id,
                "img_path": str(self.image_root / item["file_name"]),
            }
        )
        data = self.model.data_preprocessor(
            {"inputs": [packed["inputs"]], "data_samples": [packed["data_samples"]]},
            training=False,
        )
        inputs = data["inputs"].to(self.device)
        sample = data["data_samples"][0]
        annotations = self.boxes[split].get(image_id, [])
        boxes = np.array(
            [
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
                for ann in annotations
            ],
            dtype=np.float64,
        ).reshape(-1, 4)
        scale = np.array(sample.scale_factor * 2, dtype=np.float64)
        return inputs, sample, boxes * scale

    @torch.inference_mode()
    def feature(self, inputs: torch.Tensor) -> torch.Tensor:
        feature = self.pyramid(inputs)[0].float()
        return feature

    @torch.inference_mode()
    def pyramid(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.model.extract_feat(inputs)
        feature = features[0]
        if feature.shape[1] != 256:
            raise RuntimeError(f"expected 256 P3 channels, got {feature.shape}")
        return features

    @torch.inference_mode()
    def predictions(
        self, inputs: torch.Tensor, sample: Any, features: tuple[torch.Tensor, ...] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        old_threshold = self.model.bbox_head.test_cfg.score_thr
        self.model.bbox_head.test_cfg.score_thr = 0.0
        try:
            if features is None:
                features = self.model.extract_feat(inputs)
            result = self.model.bbox_head.predict(
                features, [sample], rescale=False
            )[0]
        finally:
            self.model.bbox_head.test_cfg.score_thr = old_threshold
        return (
            result.bboxes.detach().cpu().numpy().astype(np.float64),
            result.scores.detach().cpu().numpy().astype(np.float64),
        )

    def fit(self) -> dict[str, Any]:
        ids = sorted(self.images["train"])
        if self.args.fit_images:
            ids = ids[: self.args.fit_images]
        per_image = max(1, math.ceil(self.args.pca_samples / len(ids)))
        vectors = []
        spatial_energy: dict[int, list[np.ndarray]] = defaultdict(list)
        with torch.inference_mode():
            for position, image_id in enumerate(ids, 1):
                inputs, _, _ = self.sample("train", image_id)
                feature = self.feature(inputs)
                flat = feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1])
                count = min(per_image, len(flat))
                indices = self.rng.choice(len(flat), count, replace=False)
                vectors.append(flat[indices].cpu())
                for kernel in self.args.kernels:
                    energy = (feature - ring_predict(feature, kernel)).abs().mean(1)
                    sample_count = min(self.args.energy_samples_per_image, energy.numel())
                    chosen = self.rng.choice(energy.numel(), sample_count, replace=False)
                    spatial_energy[kernel].append(
                        energy.flatten()[chosen].cpu().numpy()
                    )
                if position % 100 == 0:
                    print(f"fit features: {position}/{len(ids)}", flush=True)
        matrix = torch.cat(vectors)[: self.args.pca_samples].float()
        mean = matrix.mean(0)
        centered = matrix - mean
        q = max(self.args.ranks)
        _, singular, basis = torch.pca_lowrank(
            centered.to(self.device), q=q, center=False, niter=4
        )
        basis = basis.cpu()
        channel_energy: dict[str, list[np.ndarray]] = defaultdict(list)
        # Reuse sampled train vectors for channel robust scales.
        matrix4 = matrix.T.reshape(1, matrix.shape[1], 1, matrix.shape[0])
        for rank in self.args.ranks:
            reconstructed = pca_reconstruct(matrix4, mean, basis[:, :rank])
            values = (matrix4 - reconstructed).abs().mean(1).flatten().numpy()
            channel_energy[f"pca{rank}"].append(values)
        grouped = grouped_leave_one_out(matrix4, self.args.group_size)
        channel_energy["grouped"].append(
            (matrix4 - grouped).abs().mean(1).flatten().numpy()
        )
        artifact = {
            "mean": mean.tolist(),
            "basis": basis.tolist(),
            "singular_values": singular.cpu().tolist(),
            "spatial_stats": {
                str(k): robust_stats(np.concatenate(values))
                for k, values in spatial_energy.items()
            },
            "channel_stats": {
                name: robust_stats(np.concatenate(values))
                for name, values in channel_energy.items()
            },
            "fit_split": "train",
            "fit_images": len(ids),
            "pca_samples": len(matrix),
            "seed": self.args.seed,
        }
        self.args.work_dir.mkdir(parents=True, exist_ok=True)
        (self.args.work_dir / "fit_artifact.json").write_text(
            json.dumps(artifact), encoding="utf-8"
        )
        return artifact

    def load_artifact(self) -> dict[str, Any]:
        path = self.args.work_dir / "fit_artifact.json"
        if path.exists() and not self.args.refit:
            return json.loads(path.read_text(encoding="utf-8"))
        return self.fit()

    def configuration_maps(
        self, feature: torch.Tensor, artifact: dict[str, Any]
    ) -> Iterable[tuple[int, str, dict[str, torch.Tensor]]]:
        mean = torch.tensor(artifact["mean"], device=self.device)
        basis = torch.tensor(artifact["basis"], device=self.device)
        for kernel in self.args.kernels:
            spatial = ring_predict(feature, kernel)
            spatial_stats = tuple(artifact["spatial_stats"][str(kernel)])
            for operator in [*(f"pca{rank}" for rank in self.args.ranks), "grouped"]:
                if operator == "grouped":
                    channel = grouped_leave_one_out(feature, self.args.group_size)
                else:
                    rank = int(operator[3:])
                    channel = pca_reconstruct(feature, mean, basis[:, :rank])
                yield kernel, operator, residual_maps(
                    feature,
                    spatial,
                    channel,
                    spatial_stats,
                    tuple(artifact["channel_stats"][operator]),
                )

    def negative_points(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        scale_x: float,
        scale_y: float,
        width: int,
        height: int,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        hard = []
        for box, score in zip(pred_boxes, pred_scores):
            if score < self.args.score_threshold:
                continue
            if len(gt_boxes) and box_iou(box, gt_boxes).max(initial=0) >= 0.1:
                continue
            center = (box[:2] + box[2:]) / 2
            point = (
                min(width - 1, max(0, int(round(center[0] * scale_x)))),
                min(height - 1, max(0, int(round(center[1] * scale_y)))),
            )
            if point not in hard:
                hard.append(point)
            if len(hard) >= self.args.negatives_per_image:
                break
        gt_centers = (
            np.column_stack(
                (
                    (gt_boxes[:, 0] + gt_boxes[:, 2]) * scale_x / 2,
                    (gt_boxes[:, 1] + gt_boxes[:, 3]) * scale_y / 2,
                )
            )
            if len(gt_boxes)
            else np.empty((0, 2))
        )
        random_points = []
        attempts = 0
        while len(random_points) < self.args.negatives_per_image and attempts < 5000:
            point = (int(self.rng.integers(width)), int(self.rng.integers(height)))
            attempts += 1
            if len(gt_centers) and np.linalg.norm(gt_centers - point, axis=1).min() < 5:
                continue
            if point not in random_points:
                random_points.append(point)
        return hard, random_points

    def add_locations(
        self,
        rows: list[dict[str, Any]],
        maps: dict[str, torch.Tensor],
        base: dict[str, Any],
        points: Iterable[tuple[int, int]],
        label: int,
        kind: str,
    ) -> None:
        for x, y in points:
            for name, heatmap in maps.items():
                rows.append(
                    {
                        **base,
                        "map": name,
                        "label": label,
                        "location_type": kind,
                        "x": x,
                        "y": y,
                        "score": float(heatmap[0, y, x]),
                    }
                )

    def object_rows(
        self,
        maps: dict[str, torch.Tensor],
        gt_boxes: np.ndarray,
        annotations: list[dict[str, Any]],
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        base: dict[str, Any],
        scale_x: float,
        scale_y: float,
    ) -> list[dict[str, Any]]:
        rows = []
        _, height, width = maps["hardness"].shape
        for box, annotation in zip(gt_boxes, annotations):
            difficulty = gt_metrics(box, pred_boxes, pred_scores)
            center = np.array(
                [(box[0] + box[2]) * scale_x / 2, (box[1] + box[3]) * scale_y / 2]
            )
            fw = max(EPS, (box[2] - box[0]) * scale_x)
            fh = max(EPS, (box[3] - box[1]) * scale_y)
            norm = math.sqrt(fw * fh)
            x0, y0, x1, y1 = raster_bounds(box, scale_x, scale_y, width, height)
            search_radius = 3
            sx0, sx1 = max(0, int(round(center[0])) - search_radius), min(
                width, int(round(center[0])) + search_radius + 1
            )
            sy0, sy1 = max(0, int(round(center[1])) - search_radius), min(
                height, int(round(center[1])) + search_radius + 1
            )
            row: dict[str, Any] = {
                **base,
                "gt_id": annotation["id"],
                "size_bin": size_bin(box),
                "class": annotation["category_id"],
                **difficulty,
            }
            for name, heatmap3 in maps.items():
                heatmap = heatmap3[0]
                patch = heatmap[sy0:sy1, sx0:sx1]
                index = int(patch.argmax())
                py, px = np.unravel_index(index, patch.shape)
                peak = np.array([sx0 + px, sy0 + py])
                row[f"{name}_peak_distance"] = float(
                    np.linalg.norm(peak - center) / (norm + EPS)
                )
            hardness = maps["hardness"][0]
            search = hardness[sy0:sy1, sx0:sx1]
            object_mass = hardness[y0:y1, x0:x1].sum().item()
            row["joint_inside_mass"] = object_mass / (search.sum().item() + EPS)
            distribution = search / (search.sum() + EPS)
            row["joint_hardness_mean"] = float(hardness[y0:y1, x0:x1].mean())
            row["joint_hardness_max"] = float(hardness[y0:y1, x0:x1].max())
            row["joint_hardness_entropy"] = float(
                -(distribution * torch.log(distribution + EPS)).sum()
                / math.log(max(2, distribution.numel()))
            )
            row["spatial_energy_mean"] = float(maps["spatial"][0, y0:y1, x0:x1].mean())
            row["channel_energy_mean"] = float(maps["channel"][0, y0:y1, x0:x1].mean())
            row["agreement_mean"] = float(maps["agreement"][0, y0:y1, x0:x1].mean())
            if np.isfinite(difficulty["candidate_x"]):
                candidate = (
                    float(difficulty["candidate_x"]) * scale_x,
                    float(difficulty["candidate_y"]) * scale_y,
                )
                for window in self.args.windows:
                    for tau in self.args.temperatures:
                        peak, centroid = softmax_centroid(
                            hardness, candidate, window, tau
                        )
                        align = dict(row)
                        align.update(
                            window_size=window,
                            temperature=tau,
                            joint_peak_distance=float(np.linalg.norm(peak - center) / norm),
                            joint_centroid_distance=float(
                                np.linalg.norm(centroid - center) / norm
                            ),
                            has_candidate=True,
                        )
                        rows.append(align)
            else:
                row.update(
                    window_size="",
                    temperature="",
                    joint_peak_distance=float("nan"),
                    joint_centroid_distance=float("nan"),
                    has_candidate=False,
                )
                rows.append(row)
        return rows

    def plot_panel(
        self,
        image_id: int,
        image_path: Path,
        maps: dict[str, torch.Tensor],
        gt_boxes: np.ndarray,
        base: dict[str, Any],
    ) -> None:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        names = ("spatial", "channel", "agreement", "energy", "hardness", "minimum")
        figure, axes = plt.subplots(2, 3, figsize=(12, 8))
        for axis, name in zip(axes.flat, names):
            heat = maps[name][0].detach().cpu().numpy()
            axis.imshow(heat, cmap="magma")
            for box in gt_boxes:
                sx, sy = heat.shape[1] / image.shape[1], heat.shape[0] / image.shape[0]
                axis.add_patch(
                    plt.Rectangle(
                        (box[0] * sx, box[1] * sy),
                        (box[2] - box[0]) * sx,
                        (box[3] - box[1]) * sy,
                        fill=False,
                        edgecolor="cyan",
                        linewidth=0.8,
                    )
                )
            axis.set_title(name)
            axis.axis("off")
        figure.suptitle(f"image={image_id} k={base['kernel_size']} op={base['channel_operator']}")
        figure.tight_layout()
        plot_dir = self.args.work_dir / "heatmaps"
        plot_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            plot_dir / f"{image_id}_k{base['kernel_size']}_{base['channel_operator']}.png",
            dpi=130,
        )
        plt.close(figure)

    def evaluate(self, artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        object_rows: list[dict[str, Any]] = []
        location_rows: list[dict[str, Any]] = []
        ids = sorted(self.images["val"])
        if self.args.max_images:
            ids = ids[: self.args.max_images]
        plotted = 0
        for position, image_id in enumerate(ids, 1):
            inputs, sample, gt_boxes = self.sample("val", image_id)
            pyramid = self.pyramid(inputs)
            feature = pyramid[0].float()
            pred_boxes, pred_scores = self.predictions(inputs, sample, pyramid)
            _, _, fh, fw = feature.shape
            image_shape = sample.img_shape
            scale_x, scale_y = fw / image_shape[1], fh / image_shape[0]
            hard_points, random_points = self.negative_points(
                gt_boxes, pred_boxes, pred_scores, scale_x, scale_y, fw, fh
            )
            annotations = self.boxes["val"].get(image_id, [])
            positive_points = []
            for box in gt_boxes:
                cx = int(round((box[0] + box[2]) * scale_x / 2))
                cy = int(round((box[1] + box[3]) * scale_y / 2))
                positive_points.extend(
                    (x, y)
                    for y in range(max(0, cy - 1), min(fh, cy + 2))
                    for x in range(max(0, cx - 1), min(fw, cx + 2))
                )
            for kernel, operator, maps in self.configuration_maps(feature, artifact):
                base = {
                    "image_id": image_id,
                    "kernel_size": kernel,
                    "channel_operator": operator,
                    "pca_rank": int(operator[3:]) if operator.startswith("pca") else "",
                }
                self.add_locations(
                    location_rows, maps, base, positive_points, 1, "gt"
                )
                self.add_locations(
                    location_rows, maps, base, hard_points, 0, "hard_background"
                )
                self.add_locations(
                    location_rows, maps, base, random_points, 0, "random_background"
                )
                object_rows.extend(
                    self.object_rows(
                        maps,
                        gt_boxes,
                        annotations,
                        pred_boxes,
                        pred_scores,
                        base,
                        scale_x,
                        scale_y,
                    )
                )
                if plotted < self.args.plot_images and operator == f"pca{self.args.ranks[1]}":
                    self.plot_panel(
                        image_id,
                        self.image_root / self.images["val"][image_id]["file_name"],
                        maps,
                        gt_boxes,
                        base,
                    )
                    plotted += 1
            if position % 25 == 0:
                print(f"validation: {position}/{len(ids)}", flush=True)
        return object_rows, location_rows

    def one_configuration(
        self,
        feature: torch.Tensor,
        artifact: dict[str, Any],
        wanted_kernel: int,
        wanted_operator: str,
    ) -> dict[str, torch.Tensor]:
        for kernel, operator, maps in self.configuration_maps(feature, artifact):
            if kernel == wanted_kernel and operator == wanted_operator:
                return maps
        raise ValueError(f"missing configuration k={wanted_kernel}, op={wanted_operator}")

    def perturbation_consistency(
        self, artifact: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Compare peak/centroid stability under inference-time perturbations."""
        if self.args.perturb_images <= 0:
            return []
        kernel = 5 if 5 in self.args.kernels else self.args.kernels[0]
        preferred_rank = 64 if 64 in self.args.ranks else self.args.ranks[len(self.args.ranks) // 2]
        operator = f"pca{preferred_rank}"
        rows: list[dict[str, Any]] = []
        ids = sorted(self.images["val"])[: self.args.perturb_images]
        for image_id in ids:
            inputs, sample, gt_boxes = self.sample("val", image_id)
            original_pyramid = self.pyramid(inputs)
            original_feature = original_pyramid[0].float()
            original_map = self.one_configuration(
                original_feature, artifact, kernel, operator
            )["hardness"][0]
            pred_boxes, pred_scores = self.predictions(
                inputs, sample, original_pyramid
            )
            _, fh, fw = original_feature.shape[1:]
            scale_x = fw / sample.img_shape[1]
            scale_y = fh / sample.img_shape[0]
            candidates = []
            for box, annotation in zip(gt_boxes, self.boxes["val"].get(image_id, [])):
                difficulty = gt_metrics(box, pred_boxes, pred_scores)
                if not np.isfinite(difficulty["candidate_x"]):
                    continue
                candidates.append(
                    (
                        annotation["id"],
                        float(difficulty["candidate_x"]) * scale_x,
                        float(difficulty["candidate_y"]) * scale_y,
                    )
                )
            if not candidates:
                continue
            transformed = {
                "hflip": (torch.flip(inputs, (-1,)), lambda p: np.array([fw - 1 - p[0], p[1]])),
                "vflip": (torch.flip(inputs, (-2,)), lambda p: np.array([p[0], fh - 1 - p[1]])),
                "brightness": (inputs + self.args.brightness_delta, lambda p: np.asarray(p)),
            }
            for name, (changed_inputs, inverse) in transformed.items():
                changed_feature = self.feature(changed_inputs)
                changed_map = self.one_configuration(
                    changed_feature, artifact, kernel, operator
                )["hardness"][0]
                for gt_id, cx, cy in candidates:
                    original_peak, original_centroid = softmax_centroid(
                        original_map, (cx, cy), 5, 0.25
                    )
                    if name == "hflip":
                        changed_center = (fw - 1 - cx, cy)
                    elif name == "vflip":
                        changed_center = (cx, fh - 1 - cy)
                    else:
                        changed_center = (cx, cy)
                    changed_peak, changed_centroid = softmax_centroid(
                        changed_map, changed_center, 5, 0.25
                    )
                    changed_peak = inverse(changed_peak)
                    changed_centroid = inverse(changed_centroid)
                    rows.append(
                        {
                            "image_id": image_id,
                            "gt_id": gt_id,
                            "perturbation": name,
                            "kernel_size": kernel,
                            "channel_operator": operator,
                            "peak_consistency": float(
                                np.linalg.norm(original_peak - changed_peak)
                            ),
                            "centroid_consistency": float(
                                np.linalg.norm(original_centroid - changed_centroid)
                            ),
                        }
                    )
        return rows


def summarize(
    object_rows: list[dict[str, Any]],
    location_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    locator: dict[str, Any] = {}
    configs = sorted(
        {(row["kernel_size"], row["channel_operator"]) for row in location_rows}
    )
    for kernel, operator in configs:
        selected = [
            row
            for row in location_rows
            if row["kernel_size"] == kernel
            and row["channel_operator"] == operator
            and row["location_type"] in ("gt", "hard_background")
        ]
        by_map = defaultdict(list)
        for row in selected:
            by_map[row["map"]].append(row)
        key = f"k{kernel}_{operator}"
        locator[key] = {}
        for name, rows in by_map.items():
            labels = np.array([row["label"] for row in rows])
            scores = np.array([row["score"] for row in rows])
            locator[key][name] = {
                "auroc": auroc(labels, scores),
                "auprc": auprc(labels, scores),
                "count": len(rows),
            }
        h = locator[key].get("hardness", {}).get("auprc", float("nan"))
        comparisons = ("spatial", "channel", "energy")
        locator[key]["delta_auprc"] = {
            name: h - locator[key].get(name, {}).get("auprc", float("nan"))
            for name in comparisons
        }
        locator[key]["delta_bootstrap_95ci"] = {}
        for comparison in comparisons:
            image_deltas = []
            for image_id in sorted({row["image_id"] for row in selected}):
                image_rows = [row for row in selected if row["image_id"] == image_id]
                hard_rows = [row for row in image_rows if row["map"] == "hardness"]
                comparison_rows = [
                    row for row in image_rows if row["map"] == comparison
                ]
                labels = np.array([row["label"] for row in hard_rows])
                if np.unique(labels).size < 2:
                    continue
                image_deltas.append(
                    auprc(labels, np.array([row["score"] for row in hard_rows]))
                    - auprc(
                        np.array([row["label"] for row in comparison_rows]),
                        np.array([row["score"] for row in comparison_rows]),
                    )
                )
            locator[key]["delta_bootstrap_95ci"][comparison] = bootstrap_ci(
                np.asarray(image_deltas), np.mean, bootstrap_samples, seed
            )
        locator[key]["passes_effect"] = all(
            delta >= 0.03
            and locator[key]["delta_bootstrap_95ci"][name][0] > 0
            for name, delta in locator[key]["delta_auprc"].items()
        )

    canonical: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in object_rows:
        key = (
            row["image_id"],
            row["gt_id"],
            row["kernel_size"],
            row["channel_operator"],
            row["window_size"],
            row["temperature"],
        )
        canonical[key] = row
    rows = list(canonical.values())
    hardness_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        hardness_rows[
            (
                row["image_id"],
                row["gt_id"],
                row["kernel_size"],
                row["channel_operator"],
            )
        ] = row
    hardness: dict[str, Any] = {}
    alignment: dict[str, Any] = {}
    for size in ("tiny-2", "small"):
        sized = [row for row in rows if row["size_bin"] == size]
        hardness_sized = [
            row for row in hardness_rows.values() if row["size_bin"] == size
        ]
        hardness[size] = {}
        for field, target in (
            ("joint_hardness_mean", "baseline_score"),
            ("joint_hardness_entropy", "baseline_center_error"),
            ("joint_hardness_entropy", "is_missed"),
        ):
            pairs = [
                (float(row[field]), float(row[target]))
                for row in hardness_sized
                if np.isfinite(float(row[field])) and np.isfinite(float(row[target]))
            ]
            x = np.array([pair[0] for pair in pairs])
            y = np.array([pair[1] for pair in pairs])
            if target == "baseline_score":
                y = 1 - y
            result = spearmanr(x, y) if len(x) >= 3 else None
            hardness[size][f"{field}_vs_{target}"] = {
                "rho": float(result.statistic) if result else float("nan"),
                "pvalue": float(result.pvalue) if result else float("nan"),
                "count": len(x),
            }
        alignment[size] = {}
        groups = defaultdict(list)
        for row in sized:
            if row["has_candidate"] and np.isfinite(float(row["joint_peak_distance"])):
                groups[
                    (
                        row["kernel_size"],
                        row["channel_operator"],
                        row["window_size"],
                        row["temperature"],
                    )
                ].append(row)
        for config, group in groups.items():
            peak = np.array([float(row["joint_peak_distance"]) for row in group])
            centroid = np.array([float(row["joint_centroid_distance"]) for row in group])
            delta = peak - centroid
            try:
                test = wilcoxon(peak, centroid)
                pvalue = float(test.pvalue)
            except ValueError:
                pvalue = 1.0
            reduction = (np.median(peak) - np.median(centroid)) / (
                np.median(peak) + EPS
            )
            alignment[size]["_".join(map(str, config))] = {
                "count": len(group),
                "median_peak": float(np.median(peak)),
                "median_centroid": float(np.median(centroid)),
                "median_reduction": float(reduction),
                "fraction_improved": float(np.mean(delta > 0)),
                "delta_bootstrap_95ci": bootstrap_ci(
                    delta, np.median, bootstrap_samples, seed
                ),
                "wilcoxon_pvalue": pvalue,
                "passes": bool(
                    reduction >= 0.10
                    and np.mean(delta > 0) > 0.5
                    and pvalue < 0.05
                ),
            }
    locator_passes = sum(value["passes_effect"] for value in locator.values())
    alignment_passes = sum(
        result["passes"]
        for size in alignment.values()
        for result in size.values()
    )
    hardness_pass = any(
        abs(result["rho"]) >= 0.2 and result["pvalue"] < 0.05
        for size in hardness.values()
        for result in size.values()
    )
    perturbation = {}
    for name in sorted({row["perturbation"] for row in perturbation_rows}):
        selected = [row for row in perturbation_rows if row["perturbation"] == name]
        peak = np.array([row["peak_consistency"] for row in selected])
        centroid = np.array([row["centroid_consistency"] for row in selected])
        perturbation[name] = {
            "count": len(selected),
            "median_peak_consistency": float(np.median(peak)) if len(peak) else float("nan"),
            "median_centroid_consistency": (
                float(np.median(centroid)) if len(centroid) else float("nan")
            ),
            "fraction_centroid_more_stable": (
                float(np.mean(centroid < peak)) if len(peak) else float("nan")
            ),
        }
    return {
        "locator": locator,
        "hardness": hardness,
        "alignment": alignment,
        "perturbation_consistency": perturbation,
        "gates": {
            "locator_pass": locator_passes >= 2,
            "hardness_pass": hardness_pass,
            "alignment_pass": alignment_passes >= 2,
            "decision": (
                "develop_full_module"
                if locator_passes >= 2 and hardness_pass and alignment_passes >= 2
                else "stop_or_keep_only_passing_components"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "mmdetection/work_dirs/levir_baseline/fcos/patched_config.py",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "best_coco_bbox_mAP_epoch_12.pth"
    )
    parser.add_argument(
        "--train-annotations",
        type=Path,
        default=root / "mmdetection/data/levir_ship_coco/annotations/train.json",
    )
    parser.add_argument(
        "--val-annotations",
        type=Path,
        default=root / "mmdetection/data/levir_ship_coco/annotations/val.json",
    )
    parser.add_argument(
        "--image-root", type=Path, default=root / "LevirShipData/All Images"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=root / "mmdetection/work_dirs/fcos_p3_residual_diagnostic",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-samples", type=int, default=200_000)
    parser.add_argument("--fit-images", type=int, default=0)
    parser.add_argument("--energy-samples-per-image", type=int, default=256)
    parser.add_argument("--kernels", type=parse_ints, default=(3, 5, 7))
    parser.add_argument("--ranks", type=parse_ints, default=(32, 64, 128))
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--windows", type=parse_ints, default=(3, 5, 7))
    parser.add_argument("--temperatures", type=parse_floats, default=(0.1, 0.25, 0.5, 1.0))
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--negatives-per-image", type=int, default=5)
    parser.add_argument("--plot-images", type=int, default=6)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--perturb-images", type=int, default=24)
    parser.add_argument("--brightness-delta", type=float, default=5.0)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    diagnostic = Diagnostic(args)
    artifact = diagnostic.load_artifact()
    if args.fit_only:
        print(args.work_dir / "fit_artifact.json")
        return
    objects, locations = diagnostic.evaluate(artifact)
    perturbations = diagnostic.perturbation_consistency(artifact)
    write_rows(args.work_dir / "objects.csv", objects)
    write_rows(args.work_dir / "locations.csv", locations)
    write_rows(args.work_dir / "perturbations.csv", perturbations)
    summary = summarize(
        objects, locations, perturbations, args.bootstrap_samples, args.seed
    )
    (args.work_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary["gates"], indent=2))


if __name__ == "__main__":
    main()
