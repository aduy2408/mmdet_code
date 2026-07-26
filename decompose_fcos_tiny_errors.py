#!/usr/bin/env python3
"""Causal error decomposition for frozen FCOS on LEVIR tiny ships."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import matplotlib
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ablate_fcos_p3_centroids import (
    cluster_bootstrap_ci,
    maximum_cardinality_match,
    pairwise_iou,
)
from diagnose_fcos_p3_residuals import Diagnostic, EPS, parse_floats, size_bin, write_rows


ORACLE_GAIN = 0.05


def oracle_boxes(predicted: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    pred_center = (predicted[:2] + predicted[2:]) / 2
    gt_center = (target[:2] + target[2:]) / 2
    pred_size = predicted[2:] - predicted[:2]
    gt_size = target[2:] - target[:2]

    def box(center: np.ndarray, size: np.ndarray) -> np.ndarray:
        return np.concatenate((center - size / 2, center + size / 2))

    return {
        "center": box(gt_center, pred_size),
        "extent": box(pred_center, gt_size),
        "width": box(pred_center, np.array([gt_size[0], pred_size[1]])),
        "height": box(pred_center, np.array([pred_size[0], gt_size[1]])),
        "full": target.copy(),
    }


def one_iou(box: np.ndarray, target: np.ndarray) -> float:
    return float(pairwise_iou(box[None], target[None])[0, 0])


def geometry_category(
    baseline_iou: float, center_gain: float, extent_gain: float
) -> str:
    if baseline_iou >= 0.5:
        return "success"
    center = center_gain >= ORACLE_GAIN
    extent = extent_gain >= ORACLE_GAIN
    if center and extent:
        return "center_and_extent"
    if center:
        return "center_only"
    if extent:
        return "extent_only"
    return "geometry_other"


def instrumented_assignment(
    head: Any,
    points: list[torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """FCOS assignment plus the GT index selected at every FPN point."""
    num_points_per_level = [len(level) for level in points]
    ranges = [
        level.new_tensor(head.regress_ranges[index])[None].expand(len(level), 2)
        for index, level in enumerate(points)
    ]
    all_points = torch.cat(points)
    all_ranges = torch.cat(ranges)
    if not len(gt_boxes):
        labels = gt_labels.new_full((len(all_points),), head.num_classes)
        targets = gt_boxes.new_zeros((len(all_points), 4))
        indices = gt_labels.new_full((len(all_points),), -1)
    else:
        xs, ys = all_points[:, 0, None], all_points[:, 1, None]
        left = xs - gt_boxes[None, :, 0]
        right = gt_boxes[None, :, 2] - xs
        top = ys - gt_boxes[None, :, 1]
        bottom = gt_boxes[None, :, 3] - ys
        all_targets = torch.stack((left, top, right, bottom), -1)
        inside = all_targets.min(-1).values > 0
        maximum = all_targets.max(-1).values
        in_range = (
            (maximum >= all_ranges[:, None, 0])
            & (maximum <= all_ranges[:, None, 1])
        )
        areas = (
            (gt_boxes[:, 2] - gt_boxes[:, 0])
            * (gt_boxes[:, 3] - gt_boxes[:, 1])
        )[None].repeat(len(all_points), 1)
        areas[~inside | ~in_range] = float("inf")
        minimum_area, indices = areas.min(1)
        labels = gt_labels[indices]
        labels[torch.isinf(minimum_area)] = head.num_classes
        all_targets = all_targets[torch.arange(len(all_points)), indices]
        indices = indices.clone()
        indices[torch.isinf(minimum_area)] = -1
        targets = all_targets
    split_labels = list(labels.split(num_points_per_level))
    split_targets = list(targets.split(num_points_per_level))
    split_indices = list(indices.split(num_points_per_level))
    if head.norm_on_bbox:
        split_targets = [
            target / head.strides[index]
            for index, target in enumerate(split_targets)
        ]
    return split_labels, split_targets, split_indices


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if not positives:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


class ErrorDecomposition(Diagnostic):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        from mmdet.models.utils import filter_scores_and_topk

        self.filter_scores_and_topk = filter_scores_and_topk
        self.head = self.model.bbox_head
        self.assignment_checked = False
        self.inference_checked = False

    @torch.inference_mode()
    def forward_once(
        self, inputs: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[list[torch.Tensor], ...]]:
        pyramid = self.pyramid(inputs)
        return pyramid, self.head.forward(pyramid)

    def all_candidates(
        self,
        outputs: tuple[list[torch.Tensor], ...],
        img_shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        cls_scores, bbox_preds, centernesses = outputs
        records: dict[str, list[np.ndarray]] = defaultdict(list)
        for level, (cls, bbox, ctr) in enumerate(
            zip(cls_scores, bbox_preds, centernesses)
        ):
            height, width = cls.shape[-2:]
            priors = self.head.prior_generator.single_level_grid_priors(
                (height, width),
                level_idx=level,
                dtype=cls.dtype,
                device=cls.device,
            )
            logits = cls[0].permute(1, 2, 0).reshape(
                -1, self.head.cls_out_channels
            )
            probabilities = logits.sigmoid()
            distances = bbox[0].permute(1, 2, 0).reshape(-1, 4)
            factors = ctr[0].permute(1, 2, 0).reshape(-1).sigmoid()
            cells, classes = torch.where(torch.ones_like(probabilities, dtype=torch.bool))
            selected_boxes = self.head.bbox_coder.decode(
                priors[cells], distances[cells], max_shape=img_shape
            )
            cls_probability = probabilities[cells, classes]
            centerness = factors[cells]
            candidate_id = (
                level * 10_000_000
                + cells * self.head.cls_out_channels
                + classes
            )
            tensors = {
                "candidate_id": candidate_id,
                "level": torch.full_like(cells, level),
                "cell_index": cells,
                "class_id": classes,
                "cls_score": cls_probability,
                "centerness": centerness,
                "final_score": cls_probability * centerness,
                "grid_x": priors[cells, 0],
                "grid_y": priors[cells, 1],
                "boxes": selected_boxes,
            }
            for key, value in tensors.items():
                records[key].append(value.detach().cpu().numpy())
        return {
            key: np.concatenate(values, axis=0)
            for key, values in records.items()
        }

    def filtered_ids_by_level(
        self, outputs: tuple[list[torch.Tensor], ...]
    ) -> set[int]:
        cls_scores = outputs[0]
        kept: set[int] = set()
        for level, cls in enumerate(cls_scores):
            probabilities = cls[0].permute(1, 2, 0).reshape(
                -1, self.head.cls_out_channels
            ).sigmoid()
            _, labels, cells, _ = self.filter_scores_and_topk(
                probabilities,
                self.args.score_threshold,
                self.args.nms_pre,
                None,
            )
            ids = level * 10_000_000 + cells * self.head.cls_out_channels + labels
            kept.update(map(int, ids.detach().cpu()))
        return kept

    def stage_populations(
        self,
        candidates: dict[str, np.ndarray],
        filtered_ids: set[int],
    ) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
        filtered_mask = np.array(
            [int(candidate) in filtered_ids for candidate in candidates["candidate_id"]]
        )
        filtered_indices = np.flatnonzero(filtered_mask)
        boxes = torch.tensor(candidates["boxes"][filtered_indices], device=self.device)
        scores = torch.tensor(
            candidates["final_score"][filtered_indices], device=self.device
        )
        labels = torch.tensor(
            candidates["class_id"][filtered_indices], device=self.device
        )
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths > 0) & (heights > 0)
        filtered_indices = filtered_indices[valid.cpu().numpy()]
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        if len(boxes):
            _, keep = batched_nms(
                boxes, scores, labels, self.head.test_cfg.nms
            )
            nms_indices = filtered_indices[keep.cpu().numpy()]
        else:
            nms_indices = np.empty(0, dtype=np.int64)
        official_indices = nms_indices[: self.head.test_cfg.max_per_img]
        stage = {}
        filtered_set = set(map(int, candidates["candidate_id"][filtered_indices]))
        nms_set = set(map(int, candidates["candidate_id"][nms_indices]))
        official_set = set(map(int, candidates["candidate_id"][official_indices]))
        for index, candidate_id in enumerate(candidates["candidate_id"]):
            candidate_id = int(candidate_id)
            if candidates["cls_score"][index] <= self.args.score_threshold:
                stage[candidate_id] = "classification_threshold"
            elif candidate_id not in filtered_ids:
                stage[candidate_id] = "per_level_topk"
            elif candidate_id not in filtered_set:
                stage[candidate_id] = "min_bbox_size"
            elif candidate_id not in nms_set:
                stage[candidate_id] = "nms"
            elif candidate_id not in official_set:
                stage[candidate_id] = "max_per_img"
            else:
                stage[candidate_id] = "official"
        return filtered_indices, official_indices, stage

    def assert_official_equivalence(
        self,
        outputs: tuple[list[torch.Tensor], ...],
        sample: Any,
        candidates: dict[str, np.ndarray],
        official_indices: np.ndarray,
    ) -> None:
        expected = self.head.predict_by_feat(
            *outputs,
            batch_img_metas=[sample.metainfo],
            cfg=self.head.test_cfg,
            rescale=False,
            with_nms=True,
        )[0]
        np.testing.assert_allclose(
            candidates["boxes"][official_indices],
            expected.bboxes.detach().cpu().numpy(),
            atol=1e-4,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            candidates["final_score"][official_indices],
            expected.scores.detach().cpu().numpy(),
            atol=1e-6,
            rtol=1e-5,
        )
        np.testing.assert_array_equal(
            candidates["class_id"][official_indices],
            expected.labels.detach().cpu().numpy(),
        )

    def assignments(
        self,
        outputs: tuple[list[torch.Tensor], ...],
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        featmap_sizes = [score.shape[-2:] for score in outputs[0]]
        points = self.head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=outputs[0][0].dtype,
            device=outputs[0][0].device,
        )
        gt_instances = InstanceData(
            bboxes=torch.tensor(gt_boxes, device=self.device, dtype=torch.float32),
            labels=torch.tensor(gt_classes, device=self.device, dtype=torch.long),
        )
        labels, targets, indices = instrumented_assignment(
            self.head, points, gt_instances.bboxes, gt_instances.labels
        )
        if not self.assignment_checked:
            expected_labels, expected_targets = self.head.get_targets(
                points, [gt_instances]
            )
            for actual, expected in zip(labels, expected_labels):
                torch.testing.assert_close(actual, expected)
            for actual, expected in zip(targets, expected_targets):
                torch.testing.assert_close(actual, expected)
            self.assignment_checked = True
        return (
            [index.detach().cpu().numpy() for index in indices],
            [label.detach().cpu().numpy() for label in labels],
        )

    def score_oracle_indices(
        self,
        candidates: dict[str, np.ndarray],
        filtered_indices: np.ndarray,
        gt_boxes: np.ndarray,
        score_name: str,
    ) -> np.ndarray:
        pool = filtered_indices
        if score_name == "iou_oracle":
            scores_np = pairwise_iou(candidates["boxes"][pool], gt_boxes).max(
                axis=1, initial=0
            )
        else:
            scores_np = candidates[score_name][pool]
        if len(pool) > self.args.oracle_nms_pre:
            order = np.argsort(-scores_np)[: self.args.oracle_nms_pre]
            pool, scores_np = pool[order], scores_np[order]
        if not len(pool):
            return np.empty(0, dtype=np.int64)
        boxes = torch.tensor(candidates["boxes"][pool], device=self.device)
        scores = torch.tensor(scores_np, device=self.device)
        labels = torch.tensor(candidates["class_id"][pool], device=self.device)
        _, keep = batched_nms(boxes, scores, labels, self.head.test_cfg.nms)
        return pool[keep[: self.head.test_cfg.max_per_img].cpu().numpy()]

    def run_split(
        self, split: str, image_ids: Iterable[int]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        gt_rows: list[dict[str, Any]] = []
        stage_rows: list[dict[str, Any]] = []
        recall_counts = {
            population: {
                str(threshold): defaultdict(lambda: {"matched": 0, "total": 0})
                for threshold in self.args.iou_thresholds
            }
            for population in ("official", "filtered_pre_nms", "exhaustive")
        }
        score_oracle = {
            name: {"matched": 0, "total": 0, "labels": [], "scores": []}
            for name in ("cls_score", "centerness", "final_score", "iou_oracle")
        }
        for position, image_id in enumerate(image_ids, 1):
            inputs, sample, gt_boxes = self.sample(split, image_id)
            _, outputs = self.forward_once(inputs)
            annotations = self.boxes[split].get(image_id, [])
            gt_classes = np.array(
                [annotation["category_id"] - 1 for annotation in annotations],
                dtype=np.int64,
            )
            candidates = self.all_candidates(outputs, sample.img_shape)
            filtered_ids = self.filtered_ids_by_level(outputs)
            filtered_indices, official_indices, stages = self.stage_populations(
                candidates, filtered_ids
            )
            if not self.inference_checked:
                self.assert_official_equivalence(
                    outputs, sample, candidates, official_indices
                )
                self.inference_checked = True
            assignment_indices, _ = self.assignments(
                outputs, gt_boxes, gt_classes
            )
            assignment_lookup: dict[int, list[int]] = defaultdict(list)
            for level, indices in enumerate(assignment_indices):
                for cell, gt_index in enumerate(indices):
                    if gt_index >= 0:
                        candidate_id = (
                            level * 10_000_000
                            + cell * self.head.cls_out_channels
                            + int(gt_classes[gt_index])
                        )
                        assignment_lookup[int(gt_index)].append(candidate_id)
            populations = {
                "official": official_indices,
                "filtered_pre_nms": filtered_indices,
                "exhaustive": np.arange(len(candidates["boxes"])),
            }
            pair_cache = {}
            for population, indices in populations.items():
                for threshold in self.args.iou_thresholds:
                    pairs = maximum_cardinality_match(
                        candidates["boxes"][indices],
                        candidates["class_id"][indices],
                        gt_boxes,
                        gt_classes,
                        threshold,
                    )
                    pair_cache[(population, threshold)] = [
                        (int(indices[candidate]), gt, iou)
                        for candidate, gt, iou in pairs
                    ]
                    matched_gt = {gt for _, gt, _ in pairs}
                    for gt_index, box in enumerate(gt_boxes):
                        bucket = size_bin(box)
                        for key in (bucket, "all"):
                            values = recall_counts[population][str(threshold)][key]
                            values["total"] += 1
                            values["matched"] += int(gt_index in matched_gt)
            official_success = {
                gt: (candidate, iou)
                for candidate, gt, iou in pair_cache[("official", 0.5)]
            }
            candidate_index_by_id = {
                int(candidate_id): index
                for index, candidate_id in enumerate(candidates["candidate_id"])
            }
            for gt_index, (gt_box, annotation) in enumerate(
                zip(gt_boxes, annotations)
            ):
                compatible = np.flatnonzero(candidates["class_id"] == gt_classes[gt_index])
                ious = pairwise_iou(
                    candidates["boxes"][compatible], gt_box[None]
                )[:, 0]
                best_position = int(ious.argmax()) if len(ious) else -1
                best_index = int(compatible[best_position]) if len(ious) else -1
                best_iou = float(ious[best_position]) if len(ious) else 0.0
                assigned_ids = assignment_lookup.get(gt_index, [])
                assigned_candidate_indices = [
                    candidate_index_by_id[candidate_id]
                    for candidate_id in assigned_ids
                    if candidate_id in candidate_index_by_id
                ]
                score_index = (
                    max(
                        assigned_candidate_indices,
                        key=lambda index: candidates["final_score"][index],
                    )
                    if assigned_candidate_indices
                    else -1
                )
                if score_index >= 0:
                    predicted = candidates["boxes"][score_index]
                    baseline_iou = one_iou(predicted, gt_box)
                    oracle = oracle_boxes(predicted, gt_box)
                    oracle_ious = {
                        name: one_iou(box, gt_box) for name, box in oracle.items()
                    }
                else:
                    baseline_iou = 0.0
                    oracle_ious = {
                        name: 0.0
                        for name in ("center", "extent", "width", "height", "full")
                    }
                center_gain = oracle_ious["center"] - baseline_iou
                extent_gain = oracle_ious["extent"] - baseline_iou
                if gt_index in official_success:
                    category = "success"
                elif not assigned_ids:
                    category = "assignment_failure"
                elif best_iou < 0.1:
                    category = "no_geometric_candidate"
                elif best_iou >= 0.5:
                    category = "score_ranking_failure"
                else:
                    category = geometry_category(
                        baseline_iou, center_gain, extent_gain
                    )
                best_stage = stages.get(
                    int(candidates["candidate_id"][best_index]), "none"
                ) if best_index >= 0 else "none"
                row = {
                    "image_id": image_id,
                    "gt_id": annotation["id"],
                    "class_id": int(gt_classes[gt_index]),
                    "size_bin": size_bin(gt_box),
                    "category": category,
                    "assigned_positive_count": len(assigned_ids),
                    "assigned_levels": ",".join(
                        map(
                            str,
                            sorted(
                                {
                                    candidate_id // 10_000_000
                                    for candidate_id in assigned_ids
                                }
                            ),
                        )
                    ),
                    "official_iou": (
                        official_success[gt_index][1]
                        if gt_index in official_success
                        else 0.0
                    ),
                    "best_geometry_iou": best_iou,
                    "best_geometry_stage": best_stage,
                    "score_candidate_id": (
                        int(candidates["candidate_id"][score_index])
                        if score_index >= 0
                        else -1
                    ),
                    "score_candidate_level": (
                        int(candidates["level"][score_index])
                        if score_index >= 0
                        else -1
                    ),
                    "score_candidate_cls": (
                        float(candidates["cls_score"][score_index])
                        if score_index >= 0
                        else 0.0
                    ),
                    "score_candidate_centerness": (
                        float(candidates["centerness"][score_index])
                        if score_index >= 0
                        else 0.0
                    ),
                    "score_candidate_final": (
                        float(candidates["final_score"][score_index])
                        if score_index >= 0
                        else 0.0
                    ),
                    "baseline_iou": baseline_iou,
                    "center_oracle_iou": oracle_ious["center"],
                    "extent_oracle_iou": oracle_ious["extent"],
                    "width_oracle_iou": oracle_ious["width"],
                    "height_oracle_iou": oracle_ious["height"],
                    "full_oracle_iou": oracle_ious["full"],
                    "center_gain": center_gain,
                    "extent_gain": extent_gain,
                }
                if score_index >= 0:
                    predicted = candidates["boxes"][score_index]
                    row.update(
                        pred_x1=float(predicted[0]),
                        pred_y1=float(predicted[1]),
                        pred_x2=float(predicted[2]),
                        pred_y2=float(predicted[3]),
                    )
                row.update(
                    gt_x1=float(gt_box[0]),
                    gt_y1=float(gt_box[1]),
                    gt_x2=float(gt_box[2]),
                    gt_y2=float(gt_box[3]),
                )
                gt_rows.append(row)
                if best_index >= 0:
                    stage_rows.append(
                        {
                            "image_id": image_id,
                            "gt_id": annotation["id"],
                            "candidate_id": int(candidates["candidate_id"][best_index]),
                            "class_id": int(candidates["class_id"][best_index]),
                            "level": int(candidates["level"][best_index]),
                            "cls_score": float(candidates["cls_score"][best_index]),
                            "centerness": float(candidates["centerness"][best_index]),
                            "final_score": float(candidates["final_score"][best_index]),
                            "iou": best_iou,
                            "terminal_stage": best_stage,
                        }
                    )
            for score_name, values in score_oracle.items():
                selected = self.score_oracle_indices(
                    candidates, filtered_indices, gt_boxes, score_name
                )
                pairs = maximum_cardinality_match(
                    candidates["boxes"][selected],
                    candidates["class_id"][selected],
                    gt_boxes,
                    gt_classes,
                    0.5,
                )
                matched = {gt for _, gt, _ in pairs}
                values["matched"] += len(matched)
                values["total"] += len(gt_boxes)
                # Location-level AP proxy: candidate geometry quality.
                selected_ious = pairwise_iou(
                    candidates["boxes"][selected], gt_boxes
                ).max(axis=1, initial=0)
                values["labels"].extend((selected_ious >= 0.5).astype(int).tolist())
                values["scores"].extend(
                    (
                        selected_ious
                        if score_name == "iou_oracle"
                        else candidates[score_name][selected]
                    ).tolist()
                )
            if position % 50 == 0:
                print(f"{split}: {position}", flush=True)
        recall = {}
        for population, thresholds in recall_counts.items():
            recall[population] = {}
            for threshold, buckets in thresholds.items():
                recall[population][threshold] = {
                    bucket: {
                        **values,
                        "recall": values["matched"] / max(1, values["total"]),
                    }
                    for bucket, values in buckets.items()
                }
        score_summary = {}
        for name, values in score_oracle.items():
            labels = np.array(values.pop("labels"))
            scores = np.array(values.pop("scores"))
            score_summary[name] = {
                **values,
                "recall": values["matched"] / max(1, values["total"]),
                "candidate_ap50": average_precision(labels, scores),
            }
        return gt_rows, stage_rows, {"recall": recall, "score_oracles": score_summary}


def bootstrap_prevalence_difference(
    rows: list[dict[str, Any]],
    first: str,
    second: str,
    samples: int,
    seed: int,
) -> list[float]:
    by_image: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        by_image[int(row["image_id"])].append(row["category"])
    image_ids = np.array(sorted(by_image))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        selected = rng.choice(image_ids, len(image_ids), replace=True)
        categories = [
            category
            for image_id in selected
            for category in by_image[int(image_id)]
        ]
        count = Counter(categories)
        total = max(1, len(categories))
        differences.append((count[first] - count[second]) / total)
    return [float(value) for value in np.percentile(differences, [2.5, 97.5])]


def summarize(
    rows: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    tiny = [
        row
        for row in rows
        if row["size_bin"] == "tiny-2" and row["category"] != "success"
    ]
    counts = Counter(row["category"] for row in tiny)
    ordered = counts.most_common()
    total = max(1, len(tiny))
    prevalence = {
        category: {"count": count, "fraction": count / total}
        for category, count in ordered
    }
    dominant = None
    dominance_ci = [float("nan"), float("nan")]
    if len(ordered) == 1:
        dominant = ordered[0][0]
        dominance_ci = [1.0, 1.0]
    elif len(ordered) >= 2:
        dominance_ci = bootstrap_prevalence_difference(
            tiny, ordered[0][0], ordered[1][0], samples, seed
        )
        if dominance_ci[0] > 0:
            dominant = ordered[0][0]
    geometry_rows = [
        row
        for row in tiny
        if row["score_candidate_id"] >= 0
        and row["category"]
        in ("center_only", "extent_only", "center_and_extent", "geometry_other")
    ]
    oracle_summary = {}
    for name in ("baseline_iou", "center_oracle_iou", "extent_oracle_iou", "full_oracle_iou"):
        values = np.array([row[name] for row in geometry_rows], dtype=float)
        oracle_summary[name] = {
            "count": len(values),
            "mean": float(values.mean()) if len(values) else float("nan"),
            "median": float(np.median(values)) if len(values) else float("nan"),
            "p75": float(np.percentile(values, 75)) if len(values) else float("nan"),
            "p90": float(np.percentile(values, 90)) if len(values) else float("nan"),
        }
    comparison_by_image: dict[int, list[float]] = defaultdict(list)
    for row in geometry_rows:
        comparison_by_image[int(row["image_id"])].append(
            row["extent_gain"] - row["center_gain"]
        )
    extent_minus_center_ci = cluster_bootstrap_ci(
        comparison_by_image, samples, seed
    )
    decision = "mixed_failure_no_module"
    if dominant == "assignment_failure":
        decision = "assignment_project"
    elif dominant == "score_ranking_failure":
        decision = "score_path_project"
    elif dominant == "extent_only" and extent_minus_center_ci[0] > 0:
        decision = "extent_regression_project"
    elif dominant == "center_only" and extent_minus_center_ci[1] < 0:
        decision = "center_regression_project"
    elif dominant == "center_and_extent":
        decision = "general_regression_project"
    elif dominant in ("no_geometric_candidate", "geometry_other"):
        decision = "conditional_feature_probe"
    return {
        "tiny2_failure_count": len(tiny),
        "prevalence": prevalence,
        "dominant_category": dominant,
        "top_vs_second_cluster_bootstrap_95ci": dominance_ci,
        "oracle_iou": oracle_summary,
        "extent_minus_center_gain_cluster_bootstrap_95ci": extent_minus_center_ci,
        "decision": decision,
    }


def write_error_panels(
    rows: list[dict[str, Any]],
    diagnostic: ErrorDecomposition,
    work_dir: Path,
    per_category: int = 2,
) -> None:
    panel_dir = work_dir / "error_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["size_bin"] != "tiny-2" or row["category"] == "success":
            continue
        if len(selected[row["category"]]) < per_category:
            selected[row["category"]].append(row)
    for category, category_rows in selected.items():
        for index, row in enumerate(category_rows):
            image_info = diagnostic.images["val"][int(row["image_id"])]
            image = np.asarray(
                Image.open(diagnostic.image_root / image_info["file_name"]).convert("RGB")
            )
            figure, axis = plt.subplots(figsize=(6, 6))
            axis.imshow(image)
            gt = [row[f"gt_{name}"] for name in ("x1", "y1", "x2", "y2")]
            axis.add_patch(
                plt.Rectangle(
                    (gt[0], gt[1]),
                    gt[2] - gt[0],
                    gt[3] - gt[1],
                    fill=False,
                    color="lime",
                    linewidth=2,
                    label="GT",
                )
            )
            if row.get("pred_x1") is not None:
                pred = [
                    row.get(f"pred_{name}", float("nan"))
                    for name in ("x1", "y1", "x2", "y2")
                ]
                if np.isfinite(pred).all():
                    axis.add_patch(
                        plt.Rectangle(
                            (pred[0], pred[1]),
                            pred[2] - pred[0],
                            pred[3] - pred[1],
                            fill=False,
                            color="yellow",
                            linewidth=1.5,
                            label="score candidate",
                        )
                    )
            axis.set_title(
                f"{category}\nIoU={row['baseline_iou']:.3f} "
                f"center={row['center_oracle_iou']:.3f} "
                f"extent={row['extent_oracle_iou']:.3f}"
            )
            axis.legend(loc="upper right")
            axis.axis("off")
            figure.tight_layout()
            figure.savefig(
                panel_dir
                / f"{category}_{index}_img{row['image_id']}_gt{row['gt_id']}.png",
                dpi=140,
            )
            plt.close(figure)


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
        default=root / "mmdetection/work_dirs/fcos_tiny_error_decomposition",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-pre", type=int, default=1000)
    parser.add_argument("--oracle-nms-pre", type=int, default=1000)
    parser.add_argument("--iou-thresholds", type=parse_floats, default=(0.1, 0.3, 0.5))
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = ErrorDecomposition(args)
    ids = sorted(diagnostic.images["val"])
    if args.max_images:
        ids = ids[: args.max_images]
    rows, stages, metrics = diagnostic.run_split("val", ids)
    write_rows(args.work_dir / "gt_decomposition.csv", rows)
    write_rows(args.work_dir / "candidate_stages.csv", stages)
    write_error_panels(rows, diagnostic, args.work_dir)
    summary = summarize(rows, args.bootstrap_samples, args.seed)
    failed_assigned = [
        row
        for row in rows
        if row["size_bin"] == "tiny-2"
        and row["category"] != "success"
        and row["assigned_positive_count"] > 0
    ]
    tiny_failures = [
        row
        for row in rows
        if row["size_bin"] == "tiny-2" and row["category"] != "success"
    ]
    should_probe = (
        len(failed_assigned) >= 50
        and len(failed_assigned) / max(1, len(tiny_failures)) >= 0.10
        and summary["decision"] == "conditional_feature_probe"
    )
    summary["conditional_probe"] = {
        "eligible": should_probe,
        "assigned_failed_count": len(failed_assigned),
        "fraction_of_tiny2_failures": len(failed_assigned)
        / max(1, len(tiny_failures)),
        "status": "not_needed" if not should_probe else "required",
    }
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "val_annotations": str(args.val_annotations.resolve()),
        "seed": args.seed,
        "strides": list(diagnostic.head.strides),
        "regress_ranges": list(diagnostic.head.regress_ranges),
        "center_sampling": diagnostic.head.center_sampling,
        "score_threshold": args.score_threshold,
        "nms_pre": args.nms_pre,
        "max_per_img": diagnostic.head.test_cfg.max_per_img,
        "oracle_gain_threshold": ORACLE_GAIN,
        "taxonomy_order": [
            "success",
            "assignment_failure",
            "no_geometric_candidate",
            "score_ranking_failure",
            "center_only",
            "extent_only",
            "center_and_extent",
            "geometry_other",
        ],
    }
    (args.work_dir / "recall_score_oracles.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    (args.work_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, **summary}, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps({"decision": summary["decision"], "dominant": summary["dominant_category"], "probe": should_probe}, indent=2))


if __name__ == "__main__":
    main()
