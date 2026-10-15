#!/usr/bin/env python3
"""Run merged TinyPerson evaluation for the archived DETR checkpoints."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import re
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from mmengine.config import Config

from evaluate_tinyperson_hf_matrix import score_premerged

MODELS = ("deformable_detr_r50", "rtdetr_r18")
SEEDS = (42, 43)
SOURCE = {
    ("deformable_detr_r50", 42): {
        "repo": "duyle2408/deformable-detr-tinyperson-seed42",
        "prefix": "manual_snapshot/seed42/retry2_seed42",
        "checkpoint": "best_coco_bbox_mAP_epoch_60.pth",
        "config": "patched_config.py",
        "source_commit": "17449f3b0b471510071ff683f2f59868cb25e432",
        "runtime": "/marimo/mmdet_code/mmdetection",
    },
    ("deformable_detr_r50", 43): {
        "repo": "duyle2408/deformable-detr-tinyperson-seed43",
        "prefix": "manual_snapshot/seed43/retry2_seed43",
        "checkpoint": "best_coco_bbox_mAP_epoch_58.pth",
        "config": "patched_config.py",
        "source_commit": "17449f3b0b471510071ff683f2f59868cb25e432",
        "runtime": "/marimo/mmdet_code/mmdetection",
    },
    ("rtdetr_r18", 42): {
        "repo": "duyle2408/detr_r18_rtdetr_r18_matrix_runs",
        "prefix": "detr_matrix/tinyperson/rtdetr_r18/seed42",
        "checkpoint": "best_coco_bbox_mAP_epoch_74.pth",
        "config": "patched_config.py",
        "source_commit": "66365c1553ffd121ca4ee2be9d091735faf3c182",
        "runtime": "/marimo/rtdetr-mmdet",
    },
    ("rtdetr_r18", 43): {
        "repo": "duyle2408/detr_r18_rtdetr_r18_matrix_runs",
        "prefix": "detr_matrix/tinyperson/rtdetr_r18/seed43",
        "checkpoint": "best_coco_bbox_mAP_epoch_63.pth",
        "config": "patched_config.py",
        "source_commit": "66365c1553ffd121ca4ee2be9d091735faf3c182",
        "runtime": "/marimo/rtdetr-mmdet",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/marimo/TinyPerson"))
    p.add_argument("--code-root", type=Path, default=Path("/marimo/mmdet_code"))
    p.add_argument("--python", default="/marimo/mmdet-venv/bin/python")
    p.add_argument("--output-repo-id", required=True)
    p.add_argument("--hf-repo-id", default="")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    return p.parse_args()


def prepare_val_split(data_root: Path, output: Path) -> dict[str, Path]:
    from train_all_tinyperson_baseline import prepare_validation_split

    return prepare_validation_split(data_root, output, seed=42, val_ratio=0.2)


def download_sources(args: argparse.Namespace, token: str) -> dict[tuple[str, int], dict[str, Any]]:
    downloaded = {}
    source_root = args.project / "source"
    for key, spec in SOURCE.items():
        out = source_root / key[0] / f"seed{key[1]}"
        out.mkdir(parents=True, exist_ok=True)
        ckpt = hf_hub_download(
            repo_id=spec["repo"],
            filename=f'{spec["prefix"]}/{spec["checkpoint"]}',
            repo_type="dataset",
            token=token,
            local_dir=str(out),
        )
        config = hf_hub_download(
            repo_id=spec["repo"],
            filename=f'{spec["prefix"]}/{spec["config"]}',
            repo_type="dataset",
            token=token,
            local_dir=str(out),
        )
        downloaded[key] = {**spec, "model": key[0], "seed": key[1], "checkpoint": Path(ckpt), "config": Path(config)}
    return downloaded


def patch_config(
    source_config: Path,
    output: Path,
    split: str,
    split_paths: dict[str, Path],
    data_root: Path,
    result_dir: Path,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(str(source_config))
    cfg.load_from = None
    corner_ann = (
        split_paths["val"] if split == "validation" else data_root / "annotations/corner/task/tiny_set_test_sw640_sh512_all.json"
    )
    merged_ann = (
        split_paths["val_merged"] if split == "validation" else data_root / "annotations/task/tiny_set_test_all.json"
    )
    loader_key = "val_dataloader" if split == "validation" else "test_dataloader"
    evaluator_key = "val_evaluator" if split == "validation" else "test_evaluator"
    if split == "validation":
        cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
        cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
        loader_key = "test_dataloader"
        evaluator_key = "test_evaluator"
    cfg[loader_key].dataset.ann_file = str(corner_ann)
    cfg[loader_key].dataset.data_root = ""
    cfg[loader_key].dataset.data_prefix.img = (
        str(data_root / "erase_with_uncertain_dataset/train") + "/"
        if split == "validation"
        else str(data_root / "test") + "/"
    )
    cfg[evaluator_key].ann_file = str(corner_ann)
    cfg[evaluator_key].outfile_prefix = str(result_dir / "predictions")
    cfg.work_dir = str(result_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    (result_dir / "merged_ground_truth.json").write_text(str(merged_ann), encoding="utf-8")


def predictions_to_coco(pickle_path: Path, output: Path) -> Path:
    predictions = pickle.loads(pickle_path.read_bytes())
    output_rows = []
    for sample in predictions:
        instances = sample["pred_instances"]
        boxes = instances["bboxes"]
        scores = instances["scores"]
        labels = instances["labels"]
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().tolist()
            scores = scores.detach().cpu().tolist()
            labels = labels.detach().cpu().tolist()
        width = float(sample["ori_shape"][1])
        height = float(sample["ori_shape"][0])
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            x1 = max(0.0, min(float(x1), width))
            y1 = max(0.0, min(float(y1), height))
            x2 = max(x1, min(float(x2), width))
            y2 = max(y1, min(float(y2), height))
            output_rows.append({
                "image_id": int(sample["img_id"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
                "category_id": int(label) + 1,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_rows), encoding="utf-8")
    return output


def find_prediction_json(result_dir: Path) -> Path:
    candidates = sorted(result_dir.glob("predictions*.bbox.json"))
    if not candidates:
        candidates = sorted(result_dir.glob("*.bbox.json"))
    if candidates:
        return candidates[0]
    pickle_path = result_dir / "predictions.pkl"
    if pickle_path.is_file():
        return predictions_to_coco(pickle_path, result_dir / "predictions_bbox.json")
    raise FileNotFoundError(f"No predictions.pkl or COCO bbox JSON under {result_dir}")


def run_one(args: argparse.Namespace, spec: dict[str, Any], split: str, config: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    log = out / "evaluation.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(args.code_root),
        str(args.code_root / "mmdetection"),
        "/marimo/rtdetr-mmdet",
    ])
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    cmd = [
        args.python,
        str(Path(spec["runtime"]) / "tools/test.py"),
        str(config),
        str(spec["checkpoint"]),
        "--work-dir",
        str(out),
        "--out",
        str(out / "predictions.pkl"),
    ]
    if not (out / "predictions.pkl").is_file():
        with log.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, cwd=spec["runtime"], env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        if proc.returncode:
            raise RuntimeError(f"inference failed for {spec['model']}/seed{spec['seed']}/{split}; see {log}")
    return {"command": cmd, "log": str(log), "prediction_json": str(find_prediction_json(out))}


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if args.hf_repo_id:
        args.output_repo_id = args.hf_repo_id
    args.project.mkdir(parents=True, exist_ok=True)
    split_paths = prepare_val_split(args.data_root, args.project / "data" / "tinyperson" / "seed42")
    selected = {key: spec for key, spec in SOURCE.items() if key[0] in args.models}
    original_source = SOURCE.copy()
    SOURCE.clear()
    SOURCE.update(selected)
    sources = download_sources(args, token)
    SOURCE.clear()
    SOURCE.update(original_source)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.output_repo_id, repo_type="dataset", exist_ok=True, private=False)
    summary: dict[str, Any] = {}
    prior_summary = args.project / "summary.json"
    if prior_summary.is_file():
        summary.update(json.loads(prior_summary.read_text(encoding="utf-8")))
    for (model, seed), spec in sources.items():
        run_dir = args.project / "runs" / model / f"seed{seed}"
        split_metrics: dict[str, Any] = {}
        split_records: dict[str, Any] = {}
        for split, prefix in (("validation", "val"), ("test", "test")):
            split_dir = run_dir / split
            cfg_path = run_dir / f"{split}_config.py"
            patch_config(spec["config"], cfg_path, split, split_paths, args.data_root, split_dir)
            record = run_one(args, spec, split, cfg_path, split_dir)
            corner_ann = split_paths["val"] if split == "validation" else args.data_root / "annotations/corner/task/tiny_set_test_sw640_sh512_all.json"
            merged_ann = split_paths["val_merged"] if split == "validation" else args.data_root / "annotations/task/tiny_set_test_all.json"
            merged_result = split_dir / "merged_detections.json"
            prepared_gt = split_dir / "prepared_ground_truth.json"
            from evaluate_tinyperson_metrics import merge_detections, prepare_ground_truth
            merge_detections(Path(record["prediction_json"]), corner_ann, merged_result)
            prepare_ground_truth(merged_ann, prepared_gt)
            merged = score_premerged(merged_result, prepared_gt, split_dir / "merged_metrics.json")
            split_metrics.update({f"{prefix}_merged/{k}": float(v) for k, v in merged.items()})
            split_records[split] = {**record, "merged_metrics": merged}
        key = f"tinyperson/{model}/seed{seed}"
        manifest = {
            "dataset": "TinyPerson",
            "model": model,
            "training_seed": seed,
            "split_seed": args.split_seed,
            "variant": "post_hoc_inference_and_merged_re_evaluation",
            "source_repo": spec["repo"],
            "source_prefix": spec["prefix"],
            "source_commit": spec["source_commit"],
            "checkpoint": spec["checkpoint"].name,
            "config": spec["config"].name,
            "runner": "MMDetection tools/test.py + evaluate_tinyperson_metrics.py",
            "python_executable": args.python,
            "dataset_root": str(args.data_root),
            "nms_iou": args.nms_iou,
            "validation_corner_gt": str(split_paths["val"]),
            "validation_merged_gt": str(split_paths["val_merged"]),
            "test_corner_gt": str(args.data_root / "annotations/corner/task/tiny_set_test_sw640_sh512_all.json"),
            "test_merged_gt": str(args.data_root / "annotations/task/tiny_set_test_all.json"),
            "upload_required": True,
        }
        out = args.project / "artifacts" / model / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "evaluation_metrics.json").write_text(json.dumps(split_metrics, indent=2, sort_keys=True) + "\n")
        (out / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (out / "evaluation_records.json").write_text(json.dumps(split_records, indent=2, sort_keys=True) + "\n")
        remote = f"tinyperson_detr_merged_eval/{model}/seed{seed}"
        api.upload_folder(folder_path=str(out), path_in_repo=remote, repo_id=args.output_repo_id, repo_type="dataset")
        files = set(api.list_repo_files(repo_id=args.output_repo_id, repo_type="dataset"))
        required = {f"{remote}/{n}" for n in ("evaluation_metrics.json", "evaluation_manifest.json", "evaluation_records.json")}
        missing = sorted(required - files)
        if missing:
            raise RuntimeError(f"Remote verification failed: {missing}")
        summary[key] = split_metrics
        print(json.dumps({"run": key, "metrics": split_metrics}, sort_keys=True), flush=True)
    summary_path = args.project / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    api.upload_file(path_or_fileobj=str(summary_path), path_in_repo="tinyperson_detr_merged_eval/summary.json", repo_id=args.output_repo_id, repo_type="dataset")
    print(json.dumps({"complete": len(summary), "output_repo": args.output_repo_id}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
