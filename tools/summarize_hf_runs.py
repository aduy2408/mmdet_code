from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path("hf_runs")
OUT = Path("reports")


def seed_from_text(text: str) -> str:
    match = re.search(r"seed(\d+)", text)
    return match.group(1) if match else ""


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def infer_model_variant(summary_path: Path, summary: dict) -> tuple[str, str]:
    parts = summary_path.parts
    model = summary.get("model", "")
    variant = summary.get("variant", "")
    if parts[1] == "varroa_srtod_runs":
        return model.replace("srtod_", ""), "srtod"
    if model.startswith("srtod_"):
        return model.replace("srtod_", ""), "srtod"
    if not variant:
        variant = parts[-2] if len(parts) >= 2 else ""
    if not model and len(parts) >= 3:
        model = parts[-3]
    return model, variant


def row_for_summary(summary_path: Path) -> dict:
    summary = load_json(summary_path)
    model, variant = infer_model_variant(summary_path, summary)
    run_dir = summary_path.parent
    tests = sorted(run_dir.glob("test_results/*/*.json"))
    chosen = tests[-1] if tests else None
    metrics = load_json(chosen) if chosen else {}
    notes = []
    if len(tests) > 1:
        notes.append(f"multiple test jsons: {len(tests)}; used latest {chosen.parent.name}")
    if model in {"vfnet", "reppoints"}:
        notes.append("excluded from main table: user said this model failed")
    if not tests:
        notes.append("missing test_results json")
    return {
        "repo": summary_path.parts[1],
        "seed": seed_from_text(summary_path.as_posix()) or seed_from_text(summary.get("config_path", "")),
        "model": model,
        "variant": variant,
        "bbox_mAP": metrics.get("coco/bbox_mAP", ""),
        "bbox_mAP_50": metrics.get("coco/bbox_mAP_50", ""),
        "bbox_mAP_75": metrics.get("coco/bbox_mAP_75", ""),
        "bbox_mAP_s": metrics.get("coco/bbox_mAP_s", ""),
        "bbox_mAP_m": metrics.get("coco/bbox_mAP_m", ""),
        "finished_at": summary.get("finished_at", ""),
        "summary_path": summary_path.as_posix(),
        "test_json": chosen.as_posix() if chosen else "",
        "notes": "; ".join(notes),
    }


def mean(xs: list[float]) -> str:
    return f"{sum(xs) / len(xs):.3f}" if xs else ""


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = [row_for_summary(p) for p in sorted(ROOT.glob("*/**/job_summary.json"))]
    rows.sort(key=lambda r: (r["model"], r["variant"], r["seed"], r["repo"]))

    fields = list(rows[0]) if rows else []
    with (OUT / "varroa_results_all_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    candidates = [r for r in rows if r["model"] not in {"vfnet", "reppoints"} and r["bbox_mAP"] != ""]
    best_by_key = {}
    for row in candidates:
        key = (row["model"], row["variant"], row["seed"])
        old = best_by_key.get(key)
        if old is None or row["finished_at"] > old["finished_at"]:
            best_by_key[key] = row
    main_rows = sorted(best_by_key.values(), key=lambda r: (r["model"], r["variant"], r["seed"]))

    with (OUT / "varroa_results_selected_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(main_rows)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in main_rows:
        groups[(row["model"], row["variant"])].append(row)

    summary_rows = []
    for (model, variant), group in sorted(groups.items()):
        vals = [float(r["bbox_mAP"]) for r in group]
        vals50 = [float(r["bbox_mAP_50"]) for r in group]
        vals75 = [float(r["bbox_mAP_75"]) for r in group]
        notes = sorted({r["notes"] for r in group if r["notes"]})
        summary_rows.append({
            "model": model,
            "variant": variant,
            "n": len(group),
            "seeds": ",".join(sorted({r["seed"] for r in group})),
            "mAP_mean": mean(vals),
            "mAP50_mean": mean(vals50),
            "mAP75_mean": mean(vals75),
            "mAP_values": ",".join(f"{float(r['bbox_mAP']):.3f}" for r in group),
            "notes": " | ".join(notes),
        })

    with (OUT / "varroa_results_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = ["# Varroa MMDetection Results Summary", ""]
    lines.append("Downloaded Hugging Face snapshots into `hf_runs/` with `*.pt` and `*.pth` ignored.")
    lines.append("")
    lines.append("## Mean By Model/Variant")
    lines.append("")
    lines.append("| model | variant | n | seeds | mAP | mAP50 | mAP75 | per-run mAP | notes |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---|---|")
    for r in summary_rows:
        lines.append(
            f"| {r['model']} | {r['variant']} | {r['n']} | {r['seeds']} | "
            f"{r['mAP_mean']} | {r['mAP50_mean']} | {r['mAP75_mean']} | {r['mAP_values']} | {r['notes']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `vfnet` and `reppoints` are not included in the mean table because the user noted they failed.")
    lines.append("- The mean table deduplicates by `(model, variant, seed)` and keeps the latest `finished_at`; all raw rows are still in the full CSV.")
    lines.append("- If a run had multiple `test_results/*/*.json`, the latest timestamped JSON was used and noted.")
    lines.append("- Selected per-run table: `reports/varroa_results_selected_runs.csv`.")
    lines.append("- Full raw per-run table: `reports/varroa_results_all_runs.csv`.")
    (OUT / "varroa_results_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
