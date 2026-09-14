#!/usr/bin/env python3
"""Sequential Marimo runner for SET FCOS_set on the three Varroa datasets."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SET_ROOT = (ROOT.parent / "SET").resolve()
SET_PYTHON = "/marimo/mmdet2-venv/bin/python"
CONFIGS = {
    "varroa": SET_ROOT / "configs/aitod/fcos_r50_varroa_set.py",
    "tinyperson": SET_ROOT / "configs/aitod/fcos_r50_tinyperson_set.py",
    "levir_ship": SET_ROOT / "configs/aitod/fcos_r50_levir_ship_set.py",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-repo-id", required=True)
    p.add_argument("--hf-prefix", required=True)
    p.add_argument("--work-root", default=str(ROOT / "work_dirs/set_fcos"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--datasets", default="varroa,tinyperson,levir_ship")
    return p.parse_args()


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("COMMAND", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def write_split_config(source: Path, output: Path, split: str) -> None:
    body = (
        "from mmcv import Config\n"
        f"cfg = Config.fromfile({str(source)!r})\n"
        f"cfg.data.test = cfg.data.{split}\n"
        f"cfg.dump({str(output)!r})\n"
    )
    subprocess.run([SET_PYTHON, "-c", body], check=True)


def result_json(config: Path, checkpoint: Path, result_dir: Path, env: dict[str, str]) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix = result_dir / "detections"
    run(
        [
            SET_PYTHON,
            str(SET_ROOT / "tools/test.py"),
            str(config),
            str(checkpoint),
            "--work-dir",
            str(result_dir),
            "--out",
            str(result_dir / "predictions.pkl"),
            "--eval",
            "bbox",
            "--eval-options",
            f"jsonfile_prefix={prefix}",
        ],
        cwd=SET_ROOT,
        env=env,
    )
    output = prefix.with_suffix(".bbox.json")
    if not output.is_file():
        raise FileNotFoundError(output)
    return output


def make_tiny_merged_val(prepared: Path, source: Path, output: Path) -> None:
    corner = json.loads((prepared / "annotations/val_corner.json").read_text())
    merged = json.loads((source / "erase_with_uncertain_dataset/annotations/task/tiny_set_train_all.json").read_text())
    names = {item["file_name"] for item in corner["old_images"]}
    images = [item for item in merged["images"] if item["file_name"] in names]
    ids = {item["id"] for item in images}
    payload = {k: v for k, v in merged.items() if k not in {"images", "annotations"}}
    payload["images"] = images
    payload["annotations"] = [item for item in merged["annotations"] if item["image_id"] in ids]
    output.write_text(json.dumps(payload))


def checkpoint(work: Path) -> Path:
    candidates = sorted(work.glob("epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no epoch checkpoint in {work}")
    return candidates[-1]


def upload(work: Path, repo_id: str, prefix: str, token: str) -> list[str]:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    remote = f"{prefix}/{work.name}"
    api.upload_folder(folder_path=str(work), path_in_repo=remote, repo_id=repo_id, repo_type="dataset")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    expected = [f"{remote}/manifest.json", f"{remote}/final_results.json"]
    missing = [name for name in expected if name not in files]
    if missing:
        raise FileNotFoundError(f"HF upload missing {missing}")
    return [name for name in files if name.startswith(remote + "/")]


def main() -> None:
    a = args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be supplied by the live Marimo kernel")
    env = os.environ.copy()
    env["SET_ROOT"] = str(SET_ROOT)
    env["PYTHONPATH"] = os.pathsep.join([str(SET_ROOT), str(ROOT), env.get("PYTHONPATH", "")])
    work_root = Path(a.work_root).resolve()
    datasets = [item.strip() for item in a.datasets.split(",") if item.strip()]
    for dataset in datasets:
        if dataset not in CONFIGS:
            raise ValueError(f"unknown dataset: {dataset}")
        if dataset == "tinyperson":
            run([SET_PYTHON, str(SET_ROOT / "tools/prepare_tinyperson.py"), "--seed", str(a.split_seed)], cwd=SET_ROOT, env=env)
        elif dataset == "levir_ship":
            run([SET_PYTHON, str(SET_ROOT / "tools/prepare_levir_ship.py")], cwd=SET_ROOT, env=env)
        work = work_root / dataset / f"seed{a.seed}"
        work.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "experiment_id": f"set-fcos-{dataset}-seed{a.seed}",
            "control_config_or_baseline_config": str(SET_ROOT / "configs/aitod/fcos_r50_baseline.py"),
            "variant_config_or_explicit_change": str(CONFIGS[dataset]),
            "source_commit": subprocess.check_output(["git", "-C", str(SET_ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "runner": str(ROOT / "run_set_fcos.py"),
            "python_executable": SET_PYTHON,
            "dataset_root": str((SET_ROOT / "data" / ("levir_ship" if dataset == "levir_ship" else "tinyperson_seed42" if dataset == "tinyperson" else "../mmdetection/mmdetection/data/varroa_coco")).resolve()),
            "split_seed": a.split_seed,
            "training_seed": a.seed,
            "model/backbone/pretrained source": "FCOS_set / ResNet-50 / torchvision://resnet50",
            "image_size, batch_size, epochs, patience, AMP": "1333x800 keep_ratio, 2, 12, none, false",
            "NMS IoU": 0.5,
            "HF repo and remote prefix": f"{a.hf_repo_id}:{a.hf_prefix}/{dataset}/seed{a.seed}",
            "required artifacts": ["manifest.json", "final_results.json", "checkpoint", "validation metrics", "test metrics"],
            "upload_required": True,
        }
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        config = CONFIGS[dataset]
        run([SET_PYTHON, str(SET_ROOT / "tools/train.py"), str(config), "--work-dir", str(work), "--gpus", str(a.gpus), "--seed", str(a.seed), "--deterministic"], cwd=SET_ROOT, env=env)
        ckpt = checkpoint(work)
        results: dict[str, Any] = {}
        for split in ("val", "test"):
            split_config = work / f"config_{split}.py"
            write_split_config(config, split_config, split)
            detection = result_json(split_config, ckpt, work / f"{split}_results", env)
            if dataset == "tinyperson":
                prepared = SET_ROOT / "data/tinyperson_seed42"
                source = SET_ROOT.parent / "TinyPerson/tiny_set"
                merged_gt = work / f"{split}_merged_gt.json"
                if split == "val":
                    make_tiny_merged_val(prepared, source, merged_gt)
                    corner_gt = prepared / "annotations/val_corner.json"
                else:
                    corner_gt = source / "annotations/corner/task/tiny_set_test_sw640_sh512_all.json"
                    merged_gt = source / "annotations/task/tiny_set_test_all.json"
                metrics = work / f"{split}_results/metrics.json"
                run([SET_PYTHON, str(ROOT / "evaluate_tinyperson_metrics.py"), "--res", str(detection), "--corner-gt", str(corner_gt), "--merged-gt", str(merged_gt), "--out", str(metrics)], cwd=ROOT, env=env)
            else:
                root = DATA_ROOTS[dataset]
                metrics = work / f"{split}_results/metrics.json"
                run([SET_PYTHON, str(ROOT / "evaluate_coco_metrics.py"), "--gt", str(root / "annotations" / f"{split}.json"), "--res", str(detection), "--out", str(metrics)], cwd=ROOT, env=env)
            results[split] = json.loads(metrics.read_text())
        final = {"dataset": dataset, "model": "FCOS_set", "seed": a.seed, "split_seed": a.split_seed, **results}
        (work / "final_results.json").write_text(json.dumps(final, indent=2) + "\n")
        remote_files = upload(work, a.hf_repo_id, f"{a.hf_prefix}/{dataset}", token)
        (work / "upload_verified.json").write_text(json.dumps({"repo_id": a.hf_repo_id, "remote_prefix": f"{a.hf_prefix}/{dataset}/{work.name}", "files": remote_files}, indent=2) + "\n")
        print("UPLOAD VERIFIED", dataset, len(remote_files), flush=True)


DATA_ROOTS = {
    "varroa": Path(os.environ.get("SET_VARROA_ROOT", SET_ROOT.parent / "mmdetection/mmdetection/data/varroa_coco")).resolve(),
    "levir_ship": (SET_ROOT / "data/levir_ship").resolve(),
}

if __name__ == "__main__":
    main()
