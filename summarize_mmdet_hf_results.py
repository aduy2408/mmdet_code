#!/usr/bin/env python3
"""Download lightweight MMDet HF artifacts and summarize test metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

MODELS = ("cascade_rcnn", "faster_rcnn", "fcos")
VARIANTS = ("base",)
METRICS = (
    "coco/bbox_mAP",
    "coco/bbox_mAP_50",
    "coco/bbox_mAP_75",
    "coco/bbox_mAP_s",
    "coco/bbox_mAP_m",
    "coco/bbox_mAP_l",
)
COMPLEXITY = {
    "cascade_rcnn": {"gflops": 87.898, "params_m": 69.152, "input_shape": "640x384"},
    "faster_rcnn": {"gflops": 60.099, "params_m": 41.348, "input_shape": "640x384"},
    "fcos": {"gflops": 47.148, "params_m": 32.113, "input_shape": "640x384"},
}
REPOS = (
    ("duyle2408/varroa_mmdet_runs", 42),
    ("duyle2408/varroa_mmdet_runs_seed43", None),
)
ARTIFACT_HINTS = ("job_summary.json", "test_results", ".json", ".log")


def is_metric_artifact(path: str) -> bool:
    if path.endswith(".pth") or path.endswith(".pkl"):
        return False
    return any(model in path for model in MODELS) and any(hint in path for hint in ARTIFACT_HINTS)


def infer_seed(repo_id: str, default_seed: int | None, rel_path: str) -> int:
    normalized = rel_path.lower().replace("-", "_")
    for seed in (42, 43, 44):
        if re.search(rf"seed_?{seed}|(?:^|/){seed}(?:/|$)", normalized):
            return seed
    if default_seed is not None:
        return default_seed
    # This repo was used with seed 44 at the top level and seed 43 in a nested folder.
    if repo_id.endswith("_seed43"):
        return 44
    raise ValueError(f"Cannot infer seed for {repo_id}:{rel_path}")


def download_artifacts(out_dir: Path, token: str | None) -> None:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub or run inside the ml2 conda env.") from exc

    api = HfApi(token=token)
    for repo_id, _ in REPOS:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        wanted = sorted(path for path in files if is_metric_artifact(path))
        if not wanted:
            print(f"WARN no matching artifacts found in {repo_id}")
            continue
        local_dir = out_dir / "downloads" / repo_id.replace("/", "__")
        print(f"DOWNLOAD {repo_id}: {len(wanted)} files -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=wanted,
        )


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def flatten_numbers(obj: Any, out: dict[str, float]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in METRICS and isinstance(value, (int, float)):
                out[key] = float(value)
            flatten_numbers(value, out)
    elif isinstance(obj, list):
        for item in obj:
            flatten_numbers(item, out)


def parse_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if path.suffix == ".json":
        data = load_json(path)
        if data is not None:
            flatten_numbers(data, metrics)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except UnicodeDecodeError:
        return metrics
    for metric in METRICS:
        pattern = rf"{re.escape(metric)}:\s*(-?\d+(?:\.\d+)?)"
        matches = re.findall(pattern, text)
        if matches:
            metrics[metric] = float(matches[-1])
    copy_paste = re.findall(r"bbox_mAP_copypaste:\s+([0-9.\s-]+)", text)
    if copy_paste:
        values = [float(value) for value in copy_paste[-1].split()[: len(METRICS)]]
        metrics.update(dict(zip(METRICS, values)))
    return metrics


def candidate_root(rel: Path, model: str, variant: str) -> Path:
    parts = rel.parts
    model_idx = parts.index(model)
    try:
        variant_idx = parts.index(variant, model_idx + 1)
    except ValueError:
        variant_idx = model_idx
    return Path(*parts[: variant_idx + 1])


def timestamp_for(root: Path, files: list[Path]) -> str:
    summary = root / "job_summary.json"
    data = load_json(summary) if summary.exists() else None
    if isinstance(data, dict):
        for key in ("finished_at", "started_at"):
            if data.get(key):
                return str(data[key])
    newest = max((path.stat().st_mtime for path in files), default=root.stat().st_mtime if root.exists() else 0)
    return datetime.fromtimestamp(newest).isoformat()


def timestamp_key(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def collect_candidates(out_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    groups: dict[tuple[str, int, str, str, Path], list[Path]] = defaultdict(list)
    for repo_id, default_seed in REPOS:
        repo_dir = out_dir / "downloads" / repo_id.replace("/", "__")
        if not repo_dir.exists():
            continue
        for path in repo_dir.rglob("*"):
            if not path.is_file() or path.suffix == ".pth":
                continue
            rel = path.relative_to(repo_dir)
            rel_s = rel.as_posix()
            for model in MODELS:
                if model not in rel.parts:
                    continue
                for variant in VARIANTS:
                    if variant not in rel.parts:
                        continue
                    seed = infer_seed(repo_id, default_seed, rel_s)
                    groups[(repo_id, seed, model, variant, candidate_root(rel, model, variant))].append(path)

    candidates = []
    for (repo_id, seed, model, variant, root_rel), files in groups.items():
        root = out_dir / "downloads" / repo_id.replace("/", "__") / root_rel
        metrics: dict[str, float] = {}
        for path in files:
            metrics.update(parse_metrics(path))
        candidates.append(
            {
                "repo": repo_id,
                "seed": seed,
                "model": model,
                "variant": variant,
                "root": root,
                "has_job_summary": (root / "job_summary.json").exists(),
                "has_test_results": any("test_results" in path.parts for path in files),
                "metrics": metrics,
                "timestamp": timestamp_for(root, files),
            }
        )

    selected: list[dict[str, Any]] = []
    notes: list[str] = []
    by_run: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_run[(candidate["seed"], candidate["model"], candidate["variant"])].append(candidate)

    for key, items in sorted(by_run.items()):
        valid = [
            item
            for item in items
            if item["has_job_summary"] and item["has_test_results"] and any(metric in item["metrics"] for metric in METRICS)
        ]
        pool = valid or [item for item in items if any(metric in item["metrics"] for metric in METRICS)]
        if not pool:
            notes.append(f"missing metrics for seed={key[0]} model={key[1]} variant={key[2]}")
            continue
        chosen = max(pool, key=lambda item: timestamp_key(item["timestamp"]))
        selected.append(chosen)
        skipped = [item for item in pool if item is not chosen]
        if skipped:
            skipped_roots = "; ".join(str(item["root"]) for item in skipped)
            notes.append(f"duplicate seed={key[0]} model={key[1]} variant={key[2]} chose {chosen['root']} skipped {skipped_roots}")
    return selected, notes


def write_per_seed(out_dir: Path, selected: list[dict[str, Any]]) -> Path:
    path = out_dir / "mmdet_results_per_seed.csv"
    rows = []
    by_key = {(row["seed"], row["model"], row["variant"]): row for row in selected}
    for seed in (42, 43, 44):
        for model in MODELS:
            for variant in VARIANTS:
                row = by_key.get((seed, model, variant))
                base = {"seed": seed, "model": model, "variant": variant, "status": "ok" if row else "missing"}
                if row:
                    base.update({metric: row["metrics"].get(metric, "") for metric in METRICS})
                    base.update({"repo": row["repo"], "source": str(row["root"]), "timestamp": row["timestamp"]})
                rows.append(base)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "model", "variant", "status", *METRICS, "repo", "source", "timestamp"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary(out_dir: Path, selected: list[dict[str, Any]], notes: list[str]) -> tuple[Path, Path]:
    csv_path = out_dir / "mmdet_results_summary.csv"
    md_path = out_dir / "mmdet_results_summary.md"
    rows = []
    for model in MODELS:
        for variant in VARIANTS:
            matching = [row for row in selected if row["model"] == model and row["variant"] == variant]
            for metric in METRICS:
                values = [row["metrics"][metric] for row in matching if metric in row["metrics"]]
                rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "metric": metric,
                        "n": len(values),
                        "mean": statistics.mean(values) if values else "",
                        "std": statistics.stdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else ""),
                        "seeds": " ".join(str(row["seed"]) for row in matching if metric in row["metrics"]),
                        "status": "ok" if len(values) == 3 else "missing",
                        "gflops": COMPLEXITY[model]["gflops"],
                        "params_m": COMPLEXITY[model]["params_m"],
                        "complexity_input": COMPLEXITY[model]["input_shape"],
                    }
                )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "variant",
                "metric",
                "n",
                "mean",
                "std",
                "seeds",
                "status",
                "gflops",
                "params_m",
                "complexity_input",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# MMDetection HF Results", "", "## mAP / AP50", ""]
    lines.append("| model | mAP mean | mAP std | AP50 mean | AP50 std | GFLOPs | Params(M) | FLOPs input | seeds |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for model in MODELS:
        matching = [row for row in selected if row["model"] == model and row["variant"] == "base"]
        map_values = [row["metrics"]["coco/bbox_mAP"] for row in matching if "coco/bbox_mAP" in row["metrics"]]
        ap50_values = [row["metrics"]["coco/bbox_mAP_50"] for row in matching if "coco/bbox_mAP_50" in row["metrics"]]
        seeds = " ".join(str(row["seed"]) for row in matching if "coco/bbox_mAP" in row["metrics"])
        map_mean = statistics.mean(map_values) if map_values else None
        map_std = statistics.stdev(map_values) if len(map_values) > 1 else (0.0 if map_values else None)
        ap50_mean = statistics.mean(ap50_values) if ap50_values else None
        ap50_std = statistics.stdev(ap50_values) if len(ap50_values) > 1 else (0.0 if ap50_values else None)
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    f"{map_mean:.4f}" if map_mean is not None else "",
                    f"{map_std:.4f}" if map_std is not None else "",
                    f"{ap50_mean:.4f}" if ap50_mean is not None else "",
                    f"{ap50_std:.4f}" if ap50_std is not None else "",
                    f"{COMPLEXITY[model]['gflops']:.3f}",
                    f"{COMPLEXITY[model]['params_m']:.3f}",
                    COMPLEXITY[model]["input_shape"],
                    seeds,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Full Metrics", ""])
    lines.append("| model | variant | metric | n | mean | std | seeds | status |")
    lines.append("|---|---|---|---:|---:|---:|---|---|")
    for row in rows:
        mean = f"{row['mean']:.4f}" if isinstance(row["mean"], float) else ""
        std = f"{row['std']:.4f}" if isinstance(row["std"], float) else ""
        lines.append(
            f"| {row['model']} | {row['variant']} | {row['metric']} | {row['n']} | {mean} | {std} | {row['seeds']} | {row['status']} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append(
        "- GFLOPs/Params were computed with `mmdetection/tools/analysis_tools/get_flops.py`, "
        "`--num-images 1`, local `.venv-mmdet`, and `val_dataloader.dataset.pipeline.1.scale=(640,640)` "
        "to match the current `train_all_mmdet.py` default; unsupported ops warnings from MMEngine still apply."
    )
    lines.extend(f"- {note}" for note in notes)
    if not notes:
        lines.append("- no duplicates or missing parsed runs detected")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="hf_results")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_artifacts(out_dir, args.hf_token)
    selected, notes = collect_candidates(out_dir)
    per_seed = write_per_seed(out_dir, selected)
    summary_csv, summary_md = write_summary(out_dir, selected, notes)
    print(f"WROTE {per_seed}")
    print(f"WROTE {summary_csv}")
    print(f"WROTE {summary_md}")


if __name__ == "__main__":
    main()
