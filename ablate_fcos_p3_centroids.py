#!/usr/bin/env python3
"""MMDetection-exact FCOS/P3 source-map centroid ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from diagnose_fcos_p3_residuals import (
    Diagnostic,
    EPS,
    grouped_leave_one_out,
    pca_reconstruct,
    parse_floats,
    parse_ints,
    ring_predict,
    size_bin,
    write_rows,
)


PRIMARY = (5, 0.25)
SENSITIVITY = ((5, 0.5), (3, 0.25), (7, 0.25))
CALIBRATION_TEMPERATURES = (0.1, 0.25, 0.5, 1.0)


def pairwise_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = np.prod(np.maximum(rb - lt, 0), axis=2)
    area_a = np.prod(np.maximum(boxes_a[:, 2:] - boxes_a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(boxes_b[:, 2:] - boxes_b[:, :2], 0), axis=1)
    return intersection / (
        area_a[:, None] + area_b[None, :] - intersection + EPS
    )


def maximum_cardinality_match(
    candidate_boxes: np.ndarray,
    candidate_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality class-aware matching, then maximum total IoU."""
    ious = pairwise_iou(candidate_boxes, gt_boxes)
    valid = (ious >= threshold) & (
        candidate_classes[:, None] == gt_classes[None, :]
    )
    if not valid.any():
        return []
    # Every valid edge receives a reward > 1, so cardinality dominates IoU.
    reward = np.where(valid, 1.0 + ious, 0.0)
    candidate_ids, gt_ids = linear_sum_assignment(-reward)
    return [
        (int(candidate), int(gt), float(ious[candidate, gt]))
        for candidate, gt in zip(candidate_ids, gt_ids)
        if valid[candidate, gt]
    ]


def normalized_patch_distribution(
    source: torch.Tensor,
    anchor_xy: tuple[int, int],
    window: int,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, float, bool, float]:
    """Return offsets, probabilities, entropy, boundary flag and valid fraction."""
    if source.ndim != 2:
        raise ValueError("source must be HW")
    x, y = anchor_xy
    radius = window // 2
    height, width = source.shape
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = source[y0:y1, x0:x1].float()
    ys, xs = torch.meshgrid(
        torch.arange(y0, y1, device=source.device),
        torch.arange(x0, x1, device=source.device),
        indexing="ij",
    )
    offsets = torch.stack((xs - x, ys - y), dim=-1).reshape(-1, 2).float()
    flat = patch.reshape(-1)
    median = flat.median()
    mad = (flat - median).abs().median()
    if mad <= EPS:
        probabilities = torch.full_like(flat, 1.0 / flat.numel())
    else:
        probabilities = torch.softmax((flat - median) / (mad * tau), dim=0)
    count = probabilities.numel()
    entropy = (
        float(
            -(probabilities * torch.log(probabilities + EPS)).sum()
            / math.log(count)
        )
        if count > 1
        else 0.0
    )
    boundary = count != window * window
    return offsets, probabilities, entropy, boundary, count / (window * window)


def centroid_from_grid(
    source: torch.Tensor,
    anchor_xy: tuple[int, int],
    grid_xy: np.ndarray,
    stride: float,
    window: int,
    tau: float,
) -> tuple[np.ndarray, float, bool, float]:
    offsets, probabilities, entropy, boundary, valid_fraction = (
        normalized_patch_distribution(source, anchor_xy, window, tau)
    )
    displacement = (probabilities[:, None] * offsets).sum(0).cpu().numpy()
    return (
        grid_xy + stride * displacement,
        entropy,
        boundary,
        valid_fraction,
    )


def decode_candidates(
    head: Any,
    priors: torch.Tensor,
    bbox_pred: torch.Tensor,
    img_shape: tuple[int, int],
) -> torch.Tensor:
    """Decode head outputs as-is; forward_single already owns stride scaling."""
    return head.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)


def quality_bin(distance: float) -> str:
    if distance < 0.5:
        return "[0,.5)"
    if distance < 1:
        return "[.5,1)"
    if distance < 2:
        return "[1,2)"
    return "[2,inf)"


def cluster_bootstrap_ci(
    image_values: dict[int, list[float]], samples: int, seed: int
) -> list[float]:
    if not image_values:
        return [float("nan"), float("nan")]
    image_ids = np.array(sorted(image_values))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected = rng.choice(image_ids, len(image_ids), replace=True)
        values = [
            value
            for image_id in selected
            for value in image_values[int(image_id)]
        ]
        estimates.append(float(np.median(values)))
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]


class CentroidAblation(Diagnostic):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        from mmdet.models.utils import filter_scores_and_topk

        self.filter_scores_and_topk = filter_scores_and_topk
        self.stride = float(self.model.bbox_head.strides[0])
        self.norm_on_bbox = bool(self.model.bbox_head.norm_on_bbox)
        self.artifact = json.loads(
            Path(args.fit_artifact).read_text(encoding="utf-8")
        )
        self.mean = torch.tensor(self.artifact["mean"], device=self.device)
        self.basis = torch.tensor(self.artifact["basis"], device=self.device)
        self._candidate_equivalence_checked = False

    @torch.inference_mode()
    def forward_once(
        self, inputs: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[list[torch.Tensor], ...]]:
        pyramid = self.pyramid(inputs)
        return pyramid, self.model.bbox_head.forward(pyramid)

    def p3_candidates(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        centerness: torch.Tensor,
        img_shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """Faithfully reproduce the P3 branch of _predict_by_feat_single."""
        head = self.model.bbox_head
        height, width = cls_score.shape[-2:]
        priors = head.prior_generator.single_level_grid_priors(
            (height, width),
            level_idx=0,
            dtype=cls_score.dtype,
            device=cls_score.device,
        )
        flat_cls = cls_score[0].permute(1, 2, 0).reshape(
            -1, head.cls_out_channels
        )
        scores = (
            head.loss_cls.get_activation(flat_cls)
            if getattr(head.loss_cls, "custom_cls_channels", False)
            else flat_cls.sigmoid()
        )
        flat_bbox = bbox_pred[0].permute(1, 2, 0).reshape(
            -1, head.bbox_coder.encode_size
        )
        selection_scores, labels, keep, filtered = self.filter_scores_and_topk(
            scores,
            self.args.score_threshold,
            self.args.nms_pre,
            dict(bbox_pred=flat_bbox, priors=priors),
        )
        factors = centerness[0].permute(1, 2, 0).reshape(-1).sigmoid()[keep]
        decoded = decode_candidates(
            head, filtered["priors"], filtered["bbox_pred"], img_shape
        )
        keep_np = keep.detach().cpu().numpy().astype(np.int64)
        return {
            "candidate_id": keep_np,
            "class_id": labels.detach().cpu().numpy().astype(np.int64),
            "selection_score": selection_scores.detach().cpu().numpy(),
            "centerness": factors.detach().cpu().numpy(),
            "final_score": (selection_scores * factors).detach().cpu().numpy(),
            "grid": filtered["priors"].detach().cpu().numpy(),
            "boxes": decoded.detach().cpu().numpy(),
            "map_x": keep_np % width,
            "map_y": keep_np // width,
        }

    def assert_mmdet_candidate_equivalence(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        centerness: torch.Tensor,
        img_meta: dict[str, Any],
        candidates: dict[str, np.ndarray],
    ) -> None:
        """Hard integration check against the repository prediction helper."""
        from mmengine.config import ConfigDict

        cfg = ConfigDict(
            score_thr=self.args.score_threshold,
            nms_pre=self.args.nms_pre,
            min_bbox_size=-1,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=self.args.nms_pre,
        )
        head = self.model.bbox_head
        height, width = cls_score.shape[-2:]
        priors = head.prior_generator.single_level_grid_priors(
            (height, width),
            level_idx=0,
            dtype=cls_score.dtype,
            device=cls_score.device,
        )
        expected = head._predict_by_feat_single(
            cls_score_list=[cls_score[0]],
            bbox_pred_list=[bbox_pred[0]],
            score_factor_list=[centerness[0]],
            mlvl_priors=[priors],
            img_meta=img_meta,
            cfg=cfg,
            rescale=False,
            with_nms=False,
        )
        np.testing.assert_allclose(
            candidates["boxes"],
            expected.bboxes.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_equal(
            candidates["class_id"], expected.labels.detach().cpu().numpy()
        )
        np.testing.assert_allclose(
            candidates["final_score"],
            expected.scores.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )

    def source_maps(
        self,
        feature: torch.Tensor,
        cls_score: torch.Tensor,
        centerness: torch.Tensor,
    ) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}

        def add(
            name: str,
            family: str,
            tensor: torch.Tensor,
            reduction: str,
            kernel: int | str = "",
            rank: int | str = "",
        ) -> None:
            output[name] = {
                "map": tensor[0].float(),
                "family": family,
                "reduction": reduction,
                "kernel": kernel,
                "rank": rank,
            }

        add("uniform", "uniform", torch.ones_like(feature[:, :1]).squeeze(1), "none")
        add(
            "feature_norm",
            "feature_norm",
            torch.linalg.vector_norm(feature.float(), dim=1),
            "L2",
        )
        add(
            "local_contrast",
            "local_contrast",
            torch.linalg.vector_norm(
                feature.float() - F.avg_pool2d(feature.float(), 3, 1, 1),
                dim=1,
            ),
            "L2",
            kernel=3,
        )
        spatial: dict[int, torch.Tensor] = {}
        for kernel in self.args.kernels:
            energy = (feature - ring_predict(feature, kernel)).abs().mean(1)
            spatial[kernel] = energy
            add(f"spatial_k{kernel}", "spatial", energy, "L1-mean", kernel=kernel)
        channel: dict[str, torch.Tensor] = {}
        grouped = (feature - grouped_leave_one_out(feature, self.args.group_size)).abs().mean(1)
        channel["grouped"] = grouped
        add("channel_grouped", "channel", grouped, "L1-mean")
        for rank in self.args.ranks:
            energy = (
                feature - pca_reconstruct(feature, self.mean, self.basis[:, :rank])
            ).abs().mean(1)
            channel[f"pca{rank}"] = energy
            add(f"channel_pca{rank}", "channel", energy, "L1-mean", rank=rank)
        for kernel, spatial_energy in spatial.items():
            sm, sd = self.artifact["spatial_stats"][str(kernel)]
            spatial_norm = torch.sigmoid(
                (spatial_energy - sm) / (1.4826 * sd + EPS)
            )
            for operator, channel_energy in channel.items():
                cm, cd = self.artifact["channel_stats"][operator]
                channel_norm = torch.sigmoid(
                    (channel_energy - cm) / (1.4826 * cd + EPS)
                )
                add(
                    f"energy_k{kernel}_{operator}",
                    "energy",
                    torch.sqrt(spatial_norm * channel_norm),
                    "robust-sigmoid-geomean",
                    kernel=kernel,
                    rank=(operator[3:] if operator.startswith("pca") else ""),
                )
        cls = cls_score.sigmoid().amax(1)
        ctr = centerness.sigmoid().squeeze(1)
        add("classification", "classification", cls, "sigmoid")
        add("centerness", "centerness", ctr, "sigmoid")
        add("classification_x_centerness", "head_joint", cls * ctr, "product")
        return output

    def evaluate_split(
        self,
        split: str,
        image_ids: Iterable[int],
        temperatures: dict[str, float] | None = None,
        calibration: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        recall = {
            str(threshold): defaultdict(lambda: {"total": 0, "matched": 0})
            for threshold in self.args.match_thresholds
        }
        for position, image_id in enumerate(image_ids, 1):
            inputs, sample, gt_boxes = self.sample(split, image_id)
            pyramid, head_outputs = self.forward_once(inputs)
            cls_scores, bbox_preds, centernesses = head_outputs
            candidates = self.p3_candidates(
                cls_scores[0], bbox_preds[0], centernesses[0], sample.img_shape
            )
            if not self._candidate_equivalence_checked:
                self.assert_mmdet_candidate_equivalence(
                    cls_scores[0],
                    bbox_preds[0],
                    centernesses[0],
                    sample.metainfo,
                    candidates,
                )
                self._candidate_equivalence_checked = True
            annotations = self.boxes[split].get(image_id, [])
            gt_classes = np.array(
                [annotation["category_id"] - 1 for annotation in annotations],
                dtype=np.int64,
            )
            for threshold in self.args.match_thresholds:
                pairs = maximum_cardinality_match(
                    candidates["boxes"],
                    candidates["class_id"],
                    gt_boxes,
                    gt_classes,
                    threshold,
                )
                for gt_index, box in enumerate(gt_boxes):
                    bucket = size_bin(box)
                    recall[str(threshold)][bucket]["total"] += 1
                    recall[str(threshold)]["all"]["total"] += 1
                for _, gt_index, _ in pairs:
                    bucket = size_bin(gt_boxes[gt_index])
                    recall[str(threshold)][bucket]["matched"] += 1
                    recall[str(threshold)]["all"]["matched"] += 1
            primary_pairs = maximum_cardinality_match(
                candidates["boxes"],
                candidates["class_id"],
                gt_boxes,
                gt_classes,
                self.args.match_thresholds[0],
            )
            sources = self.source_maps(
                pyramid[0].float(), cls_scores[0], centernesses[0]
            )
            configurations = (
                [(5, tau, "calibration_fit") for tau in CALIBRATION_TEMPERATURES]
                if calibration
                else [
                    *((window, tau, "fixed") for window, tau in [PRIMARY, *SENSITIVITY]),
                ]
            )
            for candidate_index, gt_index, candidate_iou in primary_pairs:
                gt_box = gt_boxes[gt_index]
                gt_center = (gt_box[:2] + gt_box[2:]) / 2
                grid = candidates["grid"][candidate_index]
                decoded = candidates["boxes"][candidate_index]
                reg_center = (decoded[:2] + decoded[2:]) / 2
                d_grid = float(np.linalg.norm(grid - gt_center) / self.stride)
                d_reg = float(np.linalg.norm(reg_center - gt_center) / self.stride)
                anchor = (
                    int(candidates["map_x"][candidate_index]),
                    int(candidates["map_y"][candidate_index]),
                )
                for source_name, source in sources.items():
                    source_configurations = list(configurations)
                    if not calibration and temperatures is not None:
                        source_configurations.append(
                            (5, temperatures.get(source_name, PRIMARY[1]), "calibrated")
                        )
                    for window, tau, evaluation_mode in source_configurations:
                        centroid, entropy, boundary, valid_fraction = centroid_from_grid(
                            source["map"],
                            anchor,
                            grid,
                            self.stride,
                            window,
                            tau,
                        )
                        d_centroid = float(
                            np.linalg.norm(centroid - gt_center) / self.stride
                        )
                        rows.append(
                            {
                                "image_id": image_id,
                                "gt_id": annotations[gt_index]["id"],
                                "size_bin": size_bin(gt_box),
                                "candidate_id": int(
                                    candidates["candidate_id"][candidate_index]
                                ),
                                "class_id": int(candidates["class_id"][candidate_index]),
                                "selection_score": float(
                                    candidates["selection_score"][candidate_index]
                                ),
                                "centerness_score": float(
                                    candidates["centerness"][candidate_index]
                                ),
                                "final_score": float(
                                    candidates["final_score"][candidate_index]
                                ),
                                "candidate_iou": candidate_iou,
                                "grid_x": float(grid[0]),
                                "grid_y": float(grid[1]),
                                "reg_x": float(reg_center[0]),
                                "reg_y": float(reg_center[1]),
                                "gt_x": float(gt_center[0]),
                                "gt_y": float(gt_center[1]),
                                "source": source_name,
                                "source_family": source["family"],
                                "reduction": source["reduction"],
                                "kernel": source["kernel"],
                                "rank": source["rank"],
                                "window": window,
                                "tau": tau,
                                "evaluation_mode": evaluation_mode,
                                "centroid_x": float(centroid[0]),
                                "centroid_y": float(centroid[1]),
                                "d_grid": d_grid,
                                "d_reg": d_reg,
                                "d_centroid": d_centroid,
                                "gain_grid": d_grid - d_centroid,
                                "gain_reg": d_reg - d_centroid,
                                "relative_gain_grid": (d_grid - d_centroid)
                                / max(d_grid, 0.25),
                                "relative_gain_reg": (d_reg - d_centroid)
                                / max(d_reg, 0.25),
                                "grid_quality_bin": quality_bin(d_grid),
                                "reg_quality_bin": quality_bin(d_reg),
                                "entropy": entropy,
                                "boundary_flag": boundary,
                                "valid_patch_fraction": valid_fraction,
                            }
                        )
            if position % 50 == 0:
                print(f"{split}: {position}", flush=True)
        normalized_recall = {}
        for threshold, buckets in recall.items():
            normalized_recall[threshold] = {}
            for bucket, values in buckets.items():
                normalized_recall[threshold][bucket] = {
                    **values,
                    "recall": values["matched"] / max(1, values["total"]),
                }
        return rows, normalized_recall

    @staticmethod
    def transformed_input_and_boxes(
        inputs: torch.Tensor,
        boxes: np.ndarray,
        transform: str,
    ) -> tuple[torch.Tensor, np.ndarray]:
        height, width = inputs.shape[-2:]
        changed_boxes = boxes.copy()
        if transform == "hflip":
            changed = torch.flip(inputs, (-1,))
            changed_boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        elif transform == "vflip":
            changed = torch.flip(inputs, (-2,))
            changed_boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        elif transform.startswith("translate_"):
            _, dx, dy = transform.split("_")
            dx, dy = int(dx), int(dy)
            changed = torch.zeros_like(inputs)
            src_x0, src_x1 = max(0, -dx), min(width, width - dx)
            src_y0, src_y1 = max(0, -dy), min(height, height - dy)
            dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
            dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
            changed[..., dst_y0:dst_y1, dst_x0:dst_x1] = inputs[
                ..., src_y0:src_y1, src_x0:src_x1
            ]
            changed_boxes[:, [0, 2]] += dx
            changed_boxes[:, [1, 3]] += dy
        elif transform == "blur":
            kernel = inputs.new_tensor(
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            ) / 16
            kernel = kernel.expand(inputs.shape[1], 1, 3, 3)
            changed = F.conv2d(inputs, kernel, padding=1, groups=inputs.shape[1])
        else:
            raise ValueError(transform)
        return changed, changed_boxes

    @staticmethod
    def inverse_point(
        point: np.ndarray, transform: str, width: int, height: int
    ) -> np.ndarray:
        if transform == "hflip":
            return np.array([width - point[0], point[1]])
        if transform == "vflip":
            return np.array([point[0], height - point[1]])
        if transform.startswith("translate_"):
            _, dx, dy = transform.split("_")
            return point - np.array([int(dx), int(dy)])
        return point

    def perturbation_rows(
        self, split: str, image_ids: Iterable[int]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        transforms = (
            "hflip",
            "vflip",
            "translate_1_0",
            "translate_-1_0",
            "translate_2_0",
            "translate_-2_0",
            "translate_0_1",
            "translate_0_-1",
            "translate_0_2",
            "translate_0_-2",
            "blur",
        )
        for image_id in image_ids:
            inputs, sample, gt_boxes = self.sample(split, image_id)
            pyramid, outputs = self.forward_once(inputs)
            cls_scores, bbox_preds, centernesses = outputs
            base_candidates = self.p3_candidates(
                cls_scores[0], bbox_preds[0], centernesses[0], sample.img_shape
            )
            annotations = self.boxes[split].get(image_id, [])
            gt_classes = np.array(
                [annotation["category_id"] - 1 for annotation in annotations],
                dtype=np.int64,
            )
            base_pairs = maximum_cardinality_match(
                base_candidates["boxes"],
                base_candidates["class_id"],
                gt_boxes,
                gt_classes,
                self.args.match_thresholds[0],
            )
            base_sources = self.source_maps(
                pyramid[0].float(), cls_scores[0], centernesses[0]
            )
            base_by_gt: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
            for candidate_index, gt_index, _ in base_pairs:
                anchor = (
                    int(base_candidates["map_x"][candidate_index]),
                    int(base_candidates["map_y"][candidate_index]),
                )
                grid = base_candidates["grid"][candidate_index]
                for source_name, source in base_sources.items():
                    base_by_gt[gt_index][source_name] = centroid_from_grid(
                        source["map"], anchor, grid, self.stride, 5, 0.25
                    )[0]
            input_height, input_width = inputs.shape[-2:]
            for transform in transforms:
                changed_inputs, changed_gt = self.transformed_input_and_boxes(
                    inputs, gt_boxes, transform
                )
                changed_pyramid, changed_outputs = self.forward_once(changed_inputs)
                changed_cls, changed_bbox, changed_ctr = changed_outputs
                changed_candidates = self.p3_candidates(
                    changed_cls[0],
                    changed_bbox[0],
                    changed_ctr[0],
                    sample.img_shape,
                )
                changed_sources = self.source_maps(
                    changed_pyramid[0].float(), changed_cls[0], changed_ctr[0]
                )
                changed_pairs = maximum_cardinality_match(
                    changed_candidates["boxes"],
                    changed_candidates["class_id"],
                    changed_gt,
                    gt_classes,
                    self.args.match_thresholds[0],
                )
                pipeline_by_gt = {gt: candidate for candidate, gt, _ in changed_pairs}
                map_height, map_width = changed_pyramid[0].shape[-2:]
                for gt_index, source_centroids in base_by_gt.items():
                    base_candidate = next(
                        candidate
                        for candidate, target, _ in base_pairs
                        if target == gt_index
                    )
                    base_x = int(base_candidates["map_x"][base_candidate])
                    base_y = int(base_candidates["map_y"][base_candidate])
                    base_grid = base_candidates["grid"][base_candidate]
                    if transform == "hflip":
                        logical_anchor = (map_width - 1 - base_x, base_y)
                    elif transform == "vflip":
                        logical_anchor = (base_x, map_height - 1 - base_y)
                    elif transform.startswith("translate_"):
                        _, dx, dy = transform.split("_")
                        target_grid = base_grid + np.array([int(dx), int(dy)])
                        logical_anchor = (
                            int(round(target_grid[0] / self.stride - 0.5)),
                            int(round(target_grid[1] / self.stride - 0.5)),
                        )
                        logical_anchor = (
                            min(map_width - 1, max(0, logical_anchor[0])),
                            min(map_height - 1, max(0, logical_anchor[1])),
                        )
                    else:
                        logical_anchor = (base_x, base_y)
                    logical_grid = np.array(
                        [
                            (logical_anchor[0] + 0.5) * self.stride,
                            (logical_anchor[1] + 0.5) * self.stride,
                        ]
                    )
                    for source_name, base_centroid in source_centroids.items():
                        map_centroid = centroid_from_grid(
                            changed_sources[source_name]["map"],
                            logical_anchor,
                            logical_grid,
                            self.stride,
                            5,
                            0.25,
                        )[0]
                        map_centroid = self.inverse_point(
                            map_centroid, transform, input_width, input_height
                        )
                        row = {
                            "image_id": image_id,
                            "gt_id": annotations[gt_index]["id"],
                            "source": source_name,
                            "transform": transform,
                            "map_equivariance_error": float(
                                np.linalg.norm(map_centroid - base_centroid)
                                / self.stride
                            ),
                            "pipeline_equivariance_error": float("nan"),
                            "pipeline_rematched": gt_index in pipeline_by_gt,
                        }
                        if gt_index in pipeline_by_gt:
                            changed_candidate = pipeline_by_gt[gt_index]
                            anchor = (
                                int(changed_candidates["map_x"][changed_candidate]),
                                int(changed_candidates["map_y"][changed_candidate]),
                            )
                            pipeline_centroid = centroid_from_grid(
                                changed_sources[source_name]["map"],
                                anchor,
                                changed_candidates["grid"][changed_candidate],
                                self.stride,
                                5,
                                0.25,
                            )[0]
                            pipeline_centroid = self.inverse_point(
                                pipeline_centroid,
                                transform,
                                input_width,
                                input_height,
                            )
                            row["pipeline_equivariance_error"] = float(
                                np.linalg.norm(pipeline_centroid - base_centroid)
                                / self.stride
                            )
                        rows.append(row)
        return rows


def calibrate_temperatures(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["source"]][float(row["tau"])].append(float(row["d_centroid"]))
    return {
        source: min(
            temperatures,
            key=lambda tau: (np.median(temperatures[tau]), tau),
        )
        for source, temperatures in grouped.items()
    }


def summarize_rows(
    rows: list[dict[str, Any]], bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sources": {},
        "quality_breakdowns": {},
        "decision": "kill_alignment",
    }
    grouped: dict[tuple[str, int, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["source"],
                int(row["window"]),
                float(row["tau"]),
                row["size_bin"],
                row["evaluation_mode"],
            )
        ].append(row)
    for (source, window, tau, bucket, mode), selected in grouped.items():
        key = f"{source}|w{window}|t{tau}|{bucket}|{mode}"
        result: dict[str, Any] = {"count": len(selected)}
        for distance in ("d_grid", "d_reg", "d_centroid"):
            values = np.array([float(row[distance]) for row in selected])
            result[distance] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
                "median_pixels": float(np.median(values) * 8),
            }
        for baseline in ("grid", "reg"):
            gains = np.array([float(row[f"gain_{baseline}"]) for row in selected])
            by_image: dict[int, list[float]] = defaultdict(list)
            for row in selected:
                by_image[int(row["image_id"])].append(float(row[f"gain_{baseline}"]))
            result[f"gain_{baseline}"] = {
                "median": float(np.median(gains)),
                "win_rate": float(np.mean(gains > 0)),
                "cluster_bootstrap_95ci": cluster_bootstrap_ci(
                    by_image, bootstrap_samples, seed
                ),
            }
        entropy = np.array([float(row["entropy"]) for row in selected])
        centroid_error = np.array([float(row["d_centroid"]) for row in selected])
        reg_gain = np.array([float(row["gain_reg"]) for row in selected])
        result["entropy_correlations"] = {
            "centroid_error": float(spearmanr(entropy, centroid_error).statistic),
            "reg_gain": float(spearmanr(entropy, reg_gain).statistic),
        }
        result["primary_pass"] = bool(
            bucket == "tiny-2"
            and mode == "fixed"
            and window == PRIMARY[0]
            and tau == PRIMARY[1]
            and all(
                result[f"gain_{baseline}"]["median"] > 0
                and result[f"gain_{baseline}"]["win_rate"] > 0.5
                and result[f"gain_{baseline}"]["cluster_bootstrap_95ci"][0] > 0
                for baseline in ("grid", "reg")
            )
        )
        summary["sources"][key] = result
    primary_sources = {
        key.split("|")[0]
        for key, value in summary["sources"].items()
        if value["primary_pass"]
    }
    robust_sources = []
    for source in primary_sources:
        neighboring_pass = False
        for key, result in summary["sources"].items():
            if not key.startswith(source + "|") or "|tiny-2|fixed" not in key:
                continue
            if "|w5|t0.25|" in key:
                continue
            if all(
                result[f"gain_{baseline}"]["median"] > 0
                and result[f"gain_{baseline}"]["win_rate"] > 0.5
                and result[f"gain_{baseline}"]["cluster_bootstrap_95ci"][0] > 0
                for baseline in ("grid", "reg")
            ):
                neighboring_pass = True
                break
        if neighboring_pass:
            robust_sources.append(source)
    summary["primary_pass_sources"] = sorted(primary_sources)
    summary["robust_pass_sources"] = sorted(robust_sources)
    primary_rows = [
        row
        for row in rows
        if row["evaluation_mode"] == "fixed"
        and int(row["window"]) == PRIMARY[0]
        and float(row["tau"]) == PRIMARY[1]
    ]
    breakdown_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in primary_rows:
        for dimension in ("grid_quality_bin", "reg_quality_bin"):
            breakdown_groups[
                (
                    row["source"],
                    row["size_bin"],
                    dimension,
                    row[dimension],
                )
            ].append(row)
    for (source, size, dimension, bucket), selected in breakdown_groups.items():
        key = f"{source}|{size}|{dimension}|{bucket}"
        summary["quality_breakdowns"][key] = {
            "count": len(selected),
            "median_d_grid": float(np.median([row["d_grid"] for row in selected])),
            "median_d_reg": float(np.median([row["d_reg"] for row in selected])),
            "median_d_centroid": float(
                np.median([row["d_centroid"] for row in selected])
            ),
            "median_gain_grid": float(
                np.median([row["gain_grid"] for row in selected])
            ),
            "median_gain_reg": float(
                np.median([row["gain_reg"] for row in selected])
            ),
            "win_rate_grid": float(
                np.mean([row["gain_grid"] > 0 for row in selected])
            ),
            "win_rate_reg": float(
                np.mean([row["gain_reg"] > 0 for row in selected])
            ),
        }
    if robust_sources:
        summary["decision"] = "keep_alignment_for_minimal_module"
    return summary


def summarize_perturbations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["transform"])].append(row)
    output = {}
    for (source, transform), selected in grouped.items():
        map_errors = np.array(
            [float(row["map_equivariance_error"]) for row in selected]
        )
        pipeline_errors = np.array(
            [
                float(row["pipeline_equivariance_error"])
                for row in selected
                if np.isfinite(float(row["pipeline_equivariance_error"]))
            ]
        )
        output[f"{source}|{transform}"] = {
            "count": len(selected),
            "pipeline_rematch_rate": len(pipeline_errors) / max(1, len(selected)),
            "map_error_median": float(np.median(map_errors)),
            "map_error_p90": float(np.percentile(map_errors, 90)),
            "pipeline_error_median": (
                float(np.median(pipeline_errors))
                if len(pipeline_errors)
                else float("nan")
            ),
            "pipeline_error_p90": (
                float(np.percentile(pipeline_errors, 90))
                if len(pipeline_errors)
                else float("nan")
            ),
        }
    return output


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "--fit-artifact",
        type=Path,
        default=root
        / "mmdetection/work_dirs/fcos_p3_residual_diagnostic/fit_artifact.json",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=root / "mmdetection/work_dirs/fcos_p3_centroid_ablation",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-pre", type=int, default=1000)
    parser.add_argument("--match-thresholds", type=parse_floats, default=(0.1, 0.3, 0.5))
    parser.add_argument("--kernels", type=parse_ints, default=(3, 5, 7))
    parser.add_argument("--ranks", type=parse_ints, default=(32, 64, 128))
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--calibration-images", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--perturb-images", type=int, default=24)
    parser.add_argument("--skip-calibration", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    ablation = CentroidAblation(args)
    calibration_temperatures = None
    if not args.skip_calibration:
        train_ids = sorted(ablation.images["train"])
        if args.calibration_images:
            train_ids = train_ids[: args.calibration_images]
        calibration_rows, _ = ablation.evaluate_split(
            "train", train_ids, calibration=True
        )
        calibration_temperatures = calibrate_temperatures(calibration_rows)
        (args.work_dir / "calibration.json").write_text(
            json.dumps(calibration_temperatures, indent=2), encoding="utf-8"
        )
    val_ids = sorted(ablation.images["val"])
    if args.max_images:
        val_ids = val_ids[: args.max_images]
    rows, recall = ablation.evaluate_split(
        "val", val_ids, temperatures=calibration_temperatures
    )
    perturbation_ids = val_ids[: args.perturb_images]
    perturbations = ablation.perturbation_rows("val", perturbation_ids)
    write_rows(args.work_dir / "objects.csv", rows)
    write_rows(args.work_dir / "perturbations.csv", perturbations)
    summary = summarize_rows(rows, args.bootstrap_samples, args.seed)
    summary["perturbation_equivariance"] = summarize_perturbations(perturbations)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "config_sha256": config_hash(args.config),
        "train_annotations": str(args.train_annotations.resolve()),
        "val_annotations": str(args.val_annotations.resolve()),
        "seed": args.seed,
        "stride": ablation.stride,
        "norm_on_bbox": ablation.norm_on_bbox,
        "score_semantics": (
            "filter/topk by activated classification; final score is "
            "classification*sigmoid(centerness)"
        ),
        "coordinate_convention": "centroid = grid_prior + stride*offset",
        "primary": {"window": PRIMARY[0], "tau": PRIMARY[1]},
        "calibrated_temperatures": calibration_temperatures,
    }
    (args.work_dir / "candidate_recall.json").write_text(
        json.dumps(recall, indent=2), encoding="utf-8"
    )
    (args.work_dir / "source_summary.json").write_text(
        json.dumps({"metadata": metadata, **summary}, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps({"decision": summary["decision"], "robust": summary["robust_pass_sources"]}, indent=2))


if __name__ == "__main__":
    main()
