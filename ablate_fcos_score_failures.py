#!/usr/bin/env python3
"""Trace and oracle-test FCOS tiny-2 score/ranking failures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch
from mmcv.ops import batched_nms
from PIL import Image
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ablate_fcos_p3_centroids import maximum_cardinality_match, pairwise_iou
from decompose_fcos_tiny_errors import ErrorDecomposition
from diagnose_fcos_p3_residuals import parse_floats, size_bin, write_rows


def select_with_nms(
    candidates: dict[str, np.ndarray],
    indices: np.ndarray,
    scores: np.ndarray,
    nms_cfg: dict[str, Any],
    max_per_img: int,
) -> np.ndarray:
    if not len(indices):
        return np.empty(0, dtype=np.int64)
    boxes = torch.as_tensor(candidates["boxes"][indices], dtype=torch.float32)
    score_tensor = torch.as_tensor(scores, dtype=torch.float32)
    labels = torch.as_tensor(candidates["class_id"][indices], dtype=torch.long)
    _, keep = batched_nms(boxes, score_tensor, labels, nms_cfg)
    return indices[keep[:max_per_img].cpu().numpy()]


def direct_suppressor(
    candidate_index: int,
    nms_survivors: np.ndarray,
    candidates: dict[str, np.ndarray],
    nms_iou: float,
) -> int:
    """Find the highest-scored surviving box that directly suppresses a box."""
    if not len(nms_survivors):
        return -1
    same_class = nms_survivors[
        candidates["class_id"][nms_survivors]
        == candidates["class_id"][candidate_index]
    ]
    higher = same_class[
        candidates["final_score"][same_class]
        >= candidates["final_score"][candidate_index]
    ]
    if not len(higher):
        return -1
    overlaps = pairwise_iou(
        candidates["boxes"][candidate_index][None],
        candidates["boxes"][higher],
    )[0]
    valid = higher[overlaps > nms_iou]
    if not len(valid):
        return -1
    return int(
        valid[np.argmax(candidates["final_score"][valid])]
    )


def cluster_fraction_ci(
    rows: list[dict[str, Any]],
    predicate,
    samples: int,
    seed: int,
) -> list[float]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[int(row["image_id"])].append(row)
    if not by_image:
        return [float("nan"), float("nan")]
    image_ids = np.array(sorted(by_image))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        chosen = rng.choice(image_ids, len(image_ids), replace=True)
        sampled = [row for image_id in chosen for row in by_image[int(image_id)]]
        estimates.append(np.mean([predicate(row) for row in sampled]))
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]


def cluster_median_difference_ci(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    samples: int,
    seed: int,
) -> list[float]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[int(row["image_id"])].append(row)
    image_ids = np.array(sorted(by_image))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        chosen = rng.choice(image_ids, len(image_ids), replace=True)
        differences = [
            float(row[left]) - float(row[right])
            for image_id in chosen
            for row in by_image[int(image_id)]
            if np.isfinite(float(row[left])) and np.isfinite(float(row[right]))
        ]
        estimates.append(float(np.median(differences)) if differences else float("nan"))
    estimates = np.asarray(estimates)
    estimates = estimates[np.isfinite(estimates)]
    return (
        [float(value) for value in np.percentile(estimates, [2.5, 97.5])]
        if len(estimates)
        else [float("nan"), float("nan")]
    )


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


class ScoreFailureAblation(ErrorDecomposition):
    def nms_survivors(
        self,
        candidates: dict[str, np.ndarray],
        filtered_indices: np.ndarray,
    ) -> np.ndarray:
        if not len(filtered_indices):
            return np.empty(0, dtype=np.int64)
        boxes = torch.tensor(candidates["boxes"][filtered_indices], device=self.device)
        scores = torch.tensor(
            candidates["final_score"][filtered_indices], device=self.device
        )
        labels = torch.tensor(
            candidates["class_id"][filtered_indices], device=self.device
        )
        _, keep = batched_nms(boxes, scores, labels, self.head.test_cfg.nms)
        return filtered_indices[keep.cpu().numpy()]

    def oracle_populations(
        self,
        candidates: dict[str, np.ndarray],
        filtered_indices: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ) -> dict[str, np.ndarray]:
        all_indices = np.arange(len(candidates["boxes"]))
        all_ious = pairwise_iou(candidates["boxes"], gt_boxes)
        class_compatible = (
            candidates["class_id"][:, None] == gt_classes[None, :]
            if len(gt_classes)
            else np.zeros((len(all_indices), 0), dtype=bool)
        )
        compatible_ious = np.where(class_compatible, all_ious, 0)
        oracle_scores = compatible_ious.max(axis=1, initial=0)
        # Zero-IoU candidates cannot improve recall and only make oracle NMS noisy.
        oracle_pool = all_indices[oracle_scores > 0]
        true_iou = select_with_nms(
            candidates,
            oracle_pool,
            oracle_scores[oracle_pool],
            self.head.test_cfg.nms,
            self.head.test_cfg.max_per_img,
        )
        filtered_ious = oracle_scores[filtered_indices]
        cls_times_iou = select_with_nms(
            candidates,
            filtered_indices,
            candidates["cls_score"][filtered_indices] * filtered_ious,
            self.head.test_cfg.nms,
            self.head.test_cfg.max_per_img,
        )
        no_nms = filtered_indices[
            np.argsort(-candidates["final_score"][filtered_indices])[
                : self.head.test_cfg.max_per_img
            ]
        ]
        # One best class-compatible exhaustive candidate per GT, then deduplicate.
        gt_aware = []
        for gt_index, gt_class in enumerate(gt_classes):
            valid = np.flatnonzero(candidates["class_id"] == gt_class)
            if len(valid):
                gt_aware.append(
                    int(valid[np.argmax(all_ious[valid, gt_index])])
                )
        return {
            "true_iou_ranking": true_iou,
            "cls_times_iou": cls_times_iou,
            "no_nms": no_nms,
            "gt_aware_selection": np.array(sorted(set(gt_aware)), dtype=np.int64),
        }

    def run_ablation(
        self, image_ids: Iterable[int]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        cohort_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        all_tiny_counts = {
            name: {"matched": 0, "total": 0}
            for name in (
                "official",
                "true_iou_ranking",
                "cls_times_iou",
                "no_nms",
                "gt_aware_selection",
            )
        }
        for position, image_id in enumerate(image_ids, 1):
            inputs, sample, gt_boxes = self.sample("val", image_id)
            _, outputs = self.forward_once(inputs)
            annotations = self.boxes["val"].get(image_id, [])
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
            assigned_gt = set(
                int(gt)
                for level in assignment_indices
                for gt in level
                if gt >= 0
            )
            official_pairs = maximum_cardinality_match(
                candidates["boxes"][official_indices],
                candidates["class_id"][official_indices],
                gt_boxes,
                gt_classes,
                0.5,
            )
            official_matched = {gt for _, gt, _ in official_pairs}
            oracle_indices = self.oracle_populations(
                candidates, filtered_indices, gt_boxes, gt_classes
            )
            oracle_matched: dict[str, set[int]] = {}
            for name, indices in oracle_indices.items():
                pairs = maximum_cardinality_match(
                    candidates["boxes"][indices],
                    candidates["class_id"][indices],
                    gt_boxes,
                    gt_classes,
                    0.5,
                )
                oracle_matched[name] = {gt for _, gt, _ in pairs}
            tiny_gt = {
                gt_index
                for gt_index, box in enumerate(gt_boxes)
                if size_bin(box) == "tiny-2"
            }
            all_tiny_counts["official"]["matched"] += len(
                tiny_gt & official_matched
            )
            all_tiny_counts["official"]["total"] += len(tiny_gt)
            for name in oracle_indices:
                all_tiny_counts[name]["matched"] += len(
                    tiny_gt & oracle_matched[name]
                )
                all_tiny_counts[name]["total"] += len(tiny_gt)

            nms_survivors = self.nms_survivors(candidates, filtered_indices)
            survivor_ids = set(map(int, candidates["candidate_id"][nms_survivors]))
            candidate_by_id = {
                int(candidate_id): index
                for index, candidate_id in enumerate(candidates["candidate_id"])
            }
            for gt_index in sorted(tiny_gt):
                if gt_index in official_matched or gt_index not in assigned_gt:
                    continue
                valid = np.flatnonzero(
                    candidates["class_id"] == gt_classes[gt_index]
                )
                ious = pairwise_iou(
                    candidates["boxes"][valid], gt_boxes[gt_index][None]
                )[:, 0]
                best_pos = int(ious.argmax())
                best_index = int(valid[best_pos])
                best_iou = float(ious[best_pos])
                if best_iou < 0.5:
                    continue
                best_id = int(candidates["candidate_id"][best_index])
                terminal_stage = stages[best_id]
                local = valid[ious >= 0.1]
                local_ious = pairwise_iou(
                    candidates["boxes"][local], gt_boxes[gt_index][None]
                )[:, 0]
                competitor = int(
                    local[np.argmax(candidates["final_score"][local])]
                )
                suppressor = (
                    direct_suppressor(
                        best_index,
                        nms_survivors,
                        candidates,
                        float(self.head.test_cfg.nms.iou_threshold),
                    )
                    if terminal_stage == "nms"
                    else -1
                )
                row = {
                    "image_id": image_id,
                    "gt_id": annotations[gt_index]["id"],
                    "gt_index": gt_index,
                    "best_candidate_id": best_id,
                    "best_level": int(candidates["level"][best_index]),
                    "best_iou": best_iou,
                    "best_cls": float(candidates["cls_score"][best_index]),
                    "best_centerness": float(candidates["centerness"][best_index]),
                    "best_product": float(candidates["final_score"][best_index]),
                    "terminal_stage": terminal_stage,
                    "competitor_id": int(candidates["candidate_id"][competitor]),
                    "competitor_iou": float(
                        pairwise_iou(
                            candidates["boxes"][competitor][None],
                            gt_boxes[gt_index][None],
                        )[0, 0]
                    ),
                    "competitor_product": float(
                        candidates["final_score"][competitor]
                    ),
                    "suppressor_id": (
                        int(candidates["candidate_id"][suppressor])
                        if suppressor >= 0
                        else -1
                    ),
                    "suppressor_iou_to_gt": (
                        float(
                            pairwise_iou(
                                candidates["boxes"][suppressor][None],
                                gt_boxes[gt_index][None],
                            )[0, 0]
                        )
                        if suppressor >= 0
                        else float("nan")
                    ),
                    "suppressor_product": (
                        float(candidates["final_score"][suppressor])
                        if suppressor >= 0
                        else float("nan")
                    ),
                    "mutual_iou_with_suppressor": (
                        float(
                            pairwise_iou(
                                candidates["boxes"][best_index][None],
                                candidates["boxes"][suppressor][None],
                            )[0, 0]
                        )
                        if suppressor >= 0
                        else float("nan")
                    ),
                    "rescued_true_iou": gt_index
                    in oracle_matched["true_iou_ranking"],
                    "rescued_cls_times_iou": gt_index
                    in oracle_matched["cls_times_iou"],
                    "rescued_no_nms": gt_index in oracle_matched["no_nms"],
                    "rescued_gt_aware": gt_index
                    in oracle_matched["gt_aware_selection"],
                    "rho_cls_iou": safe_spearman(
                        candidates["cls_score"][local], local_ious
                    ),
                    "rho_centerness_iou": safe_spearman(
                        candidates["centerness"][local], local_ious
                    ),
                    "rho_product_iou": safe_spearman(
                        candidates["final_score"][local], local_ious
                    ),
                    "gt_x1": float(gt_boxes[gt_index, 0]),
                    "gt_y1": float(gt_boxes[gt_index, 1]),
                    "gt_x2": float(gt_boxes[gt_index, 2]),
                    "gt_y2": float(gt_boxes[gt_index, 3]),
                }
                for prefix, index in (
                    ("best", best_index),
                    ("competitor", competitor),
                    ("suppressor", suppressor),
                ):
                    if index >= 0:
                        box = candidates["boxes"][index]
                        row.update(
                            {
                                f"{prefix}_x1": float(box[0]),
                                f"{prefix}_y1": float(box[1]),
                                f"{prefix}_x2": float(box[2]),
                                f"{prefix}_y2": float(box[3]),
                            }
                        )
                cohort_rows.append(row)
                roles = {best_index: "best_iou", competitor: "highest_product"}
                if suppressor >= 0:
                    roles[suppressor] = (
                        roles.get(suppressor, "") + "|direct_suppressor"
                    ).strip("|")
                for index in local:
                    candidate_id = int(candidates["candidate_id"][index])
                    candidate_rows.append(
                        {
                            "image_id": image_id,
                            "gt_id": annotations[gt_index]["id"],
                            "candidate_id": candidate_id,
                            "level": int(candidates["level"][index]),
                            "cls_score": float(candidates["cls_score"][index]),
                            "centerness": float(candidates["centerness"][index]),
                            "product_score": float(
                                candidates["final_score"][index]
                            ),
                            "gt_iou": float(
                                pairwise_iou(
                                    candidates["boxes"][index][None],
                                    gt_boxes[gt_index][None],
                                )[0, 0]
                            ),
                            "terminal_stage": stages[candidate_id],
                            "role": roles.get(int(index), "local"),
                        }
                    )
            if position % 50 == 0:
                print(f"val: {position}", flush=True)
        for values in all_tiny_counts.values():
            values["recall"] = values["matched"] / max(1, values["total"])
        return cohort_rows, candidate_rows, {"all_tiny2": all_tiny_counts}


def summarize(
    rows: list[dict[str, Any]],
    all_tiny: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    stage_counts = Counter(row["terminal_stage"] for row in rows)
    cohort_size = max(1, len(rows))
    stage_prevalence = {
        stage: {"count": count, "fraction": count / cohort_size}
        for stage, count in stage_counts.items()
    }
    threshold_ci = cluster_fraction_ci(
        rows,
        lambda row: row["terminal_stage"] == "classification_threshold",
        bootstrap_samples,
        seed,
    )
    rescue = {}
    for oracle in (
        "rescued_true_iou",
        "rescued_cls_times_iou",
        "rescued_no_nms",
        "rescued_gt_aware",
    ):
        fraction = (
            float(np.mean([bool(row[oracle]) for row in rows]))
            if rows
            else 0.0
        )
        rescue[oracle] = {
            "count": sum(bool(row[oracle]) for row in rows),
            "fraction": fraction,
            "cluster_bootstrap_95ci": cluster_fraction_ci(
                rows, lambda row, key=oracle: bool(row[key]), bootstrap_samples, seed
            ),
        }
    non_threshold = [
        row
        for row in rows
        if row["terminal_stage"] != "classification_threshold"
    ]
    quality_fraction = (
        np.mean(
            [
                row["terminal_stage"]
                in ("per_level_topk", "nms", "max_per_img")
                for row in non_threshold
            ]
        )
        if non_threshold
        else 0.0
    )
    nms_rows = [row for row in rows if row["terminal_stage"] == "nms"]
    nms_rescue_fraction = (
        np.mean([row["rescued_no_nms"] for row in nms_rows])
        if nms_rows
        else 0.0
    )
    actual_recall = all_tiny["official"]["recall"]
    quality_gain_all = (
        all_tiny["cls_times_iou"]["recall"] - actual_recall
    )
    decision = "abandon_fcos_diagnostic"
    if stage_counts["classification_threshold"] / cohort_size > 0.5 and threshold_ci[0] > 0.5:
        decision = "classification_branch"
    elif (
        rescue["rescued_cls_times_iou"]["fraction"] >= 0.20
        and rescue["rescued_cls_times_iou"]["cluster_bootstrap_95ci"][0] > 0
        and quality_fraction > 0.5
    ):
        decision = "quality_aware_scoring"
    elif (
        (
            rescue["rescued_no_nms"]["fraction"] > 0.5
            and rescue["rescued_no_nms"]["cluster_bootstrap_95ci"][0] > 0.5
        )
        or (
            nms_rescue_fraction >= 0.8
            and rescue["rescued_no_nms"]["fraction"]
            > rescue["rescued_cls_times_iou"]["fraction"]
        )
    ):
        decision = "nms_policy"
    correlations = {
        key: {
            "median": float(
                np.nanmedian([row[key] for row in rows])
            ) if rows else float("nan"),
            "count": int(
                np.isfinite([row[key] for row in rows]).sum()
            ),
        }
        for key in ("rho_cls_iou", "rho_centerness_iou", "rho_product_iou")
    }
    correlation_differences = {
        "product_minus_cls": cluster_median_difference_ci(
            rows,
            "rho_product_iou",
            "rho_cls_iou",
            bootstrap_samples,
            seed,
        ),
        "product_minus_centerness": cluster_median_difference_ci(
            rows,
            "rho_product_iou",
            "rho_centerness_iou",
            bootstrap_samples,
            seed,
        ),
    }
    return {
        "cohort_size": len(rows),
        "stage_prevalence": stage_prevalence,
        "classification_threshold_fraction_cluster_bootstrap_95ci": threshold_ci,
        "oracle_rescue": rescue,
        "all_tiny2_recall": all_tiny,
        "cls_times_iou_all_tiny2_recall_gain": quality_gain_all,
        "nms_stage_rescue_fraction": float(nms_rescue_fraction),
        "rank_correlations": correlations,
        "rank_correlation_difference_cluster_bootstrap_95ci": correlation_differences,
        "decision": decision,
    }


def write_panels(
    rows: list[dict[str, Any]],
    diagnostic: ScoreFailureAblation,
    output_dir: Path,
) -> None:
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        *[row for row in rows if row["terminal_stage"] == "nms"],
        *[
            row
            for row in rows
            if row["terminal_stage"] == "classification_threshold"
        ][:6],
    ]
    for row in selected:
        image_info = diagnostic.images["val"][int(row["image_id"])]
        image = np.asarray(
            Image.open(diagnostic.image_root / image_info["file_name"]).convert("RGB")
        )
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(image)
        for prefix, color, label in (
            ("gt", "lime", "GT"),
            ("best", "cyan", "good candidate"),
            ("suppressor", "red", "suppressor"),
        ):
            values = [row.get(f"{prefix}_{key}") for key in ("x1", "y1", "x2", "y2")]
            if all(value is not None and np.isfinite(float(value)) for value in values):
                x1, y1, x2, y2 = map(float, values)
                axis.add_patch(
                    plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        color=color,
                        linewidth=2,
                        label=label,
                    )
                )
        axis.set_title(
            f"{row['terminal_stage']} | IoU={row['best_iou']:.3f} "
            f"cls={row['best_cls']:.3f} ctr={row['best_centerness']:.3f}"
        )
        axis.legend(loc="upper right")
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(
            panel_dir
            / f"{row['terminal_stage']}_img{row['image_id']}_gt{row['gt_id']}.png",
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
        default=root / "mmdetection/work_dirs/fcos_score_failure_ablation",
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
    diagnostic = ScoreFailureAblation(args)
    ids = sorted(diagnostic.images["val"])
    if args.max_images:
        ids = ids[: args.max_images]
    rows, candidates, metrics = diagnostic.run_ablation(ids)
    summary = summarize(
        rows, metrics["all_tiny2"], args.bootstrap_samples, args.seed
    )
    write_rows(args.work_dir / "cohort.csv", rows)
    write_rows(args.work_dir / "candidates.csv", candidates)
    write_panels(rows, diagnostic, args.work_dir)
    metadata = {
        "score_threshold": args.score_threshold,
        "nms_pre": args.nms_pre,
        "nms_iou": float(diagnostic.head.test_cfg.nms.iou_threshold),
        "max_per_img": diagnostic.head.test_cfg.max_per_img,
        "success_iou": 0.5,
        "seed": args.seed,
        "oracle_universes": {
            "true_iou_ranking": "exhaustive candidates with IoU>0",
            "cls_times_iou": "exact thresholded/per-level-topk candidates",
            "no_nms": "exact filtered candidates, product score, no NMS",
            "gt_aware_selection": "best exhaustive class-compatible candidate per GT",
        },
    }
    (args.work_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, **summary}, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "cohort": summary["cohort_size"],
                "stages": summary["stage_prevalence"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
