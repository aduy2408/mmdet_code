#!/usr/bin/env python3
"""Train the FCOS DBSS ablations on LEVIR-Ship."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import train_all_levir_baseline as levir


VARIANTS = {
    'baseline': 'configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py',
    'ridge': 'projects/dbss/configs/fcos_dbss_ridge.py',
    'ridge_g03': 'projects/dbss/configs/fcos_dbss_ridge_gamma03.py',
    'ridge_g06': 'projects/dbss/configs/fcos_dbss_ridge_gamma06.py',
    'ridge_g10': 'projects/dbss/configs/fcos_dbss_ridge_gamma10.py',
    'softmax': 'projects/dbss/configs/fcos_dbss_softmax.py',
    'ridge_haar': 'projects/dbss/configs/fcos_dbss_ridge_haar.py',
}
SELECTORS = ('legacy_forced_k', 'variable_k')
RESIDUALS = (
    'ridge', 'learned_control', 'random_bases', 'shuffled_bases',
    'topk_only', 'softmax')
FALSIFICATION_VARIANTS = {
    f'{selector}_{residual}': (selector, residual)
    for selector in SELECTORS for residual in RESIDUALS
}
VARIANTS.update({
    variant: 'projects/dbss/configs/fcos_dbss_ridge_gamma06.py'
    for variant in FALSIFICATION_VARIANTS
})
DEFAULT_VARIANTS = ','.join(FALSIFICATION_VARIANTS)


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def set_resize_scale(obj: Any, image_size: int) -> None:
    if isinstance(obj, dict):
        if obj.get('type') == 'Resize':
            obj['scale'] = (image_size, image_size)
        for value in obj.values():
            set_resize_scale(value, image_size)
    elif isinstance(obj, list):
        for value in obj:
            set_resize_scale(value, image_size)


def exclude_black_images(
        dataset_out: Path, data_root: Path, inventory: str) -> None:
    inventory_path = Path(inventory)
    if not inventory_path.is_absolute():
        repo_inventory = Path(__file__).resolve().parent / inventory_path
        inventory_path = (
            repo_inventory if repo_inventory.is_file()
            else data_root / inventory_path)
    with inventory_path.open(encoding='utf-8-sig', newline='') as stream:
        excluded = {row['image'] for row in csv.DictReader(stream)}
    for split in ('train', 'val', 'test'):
        annotation_path = dataset_out / 'annotations' / f'{split}.json'
        payload = json.loads(annotation_path.read_text(encoding='utf-8'))
        removed_ids = {
            image['id'] for image in payload['images']
            if Path(image['file_name']).name in excluded
        }
        payload['images'] = [
            image for image in payload['images']
            if image['id'] not in removed_ids
        ]
        payload['annotations'] = [
            annotation for annotation in payload['annotations']
            if annotation['image_id'] not in removed_ids
        ]
        annotation_path.write_text(
            json.dumps(payload), encoding='utf-8')
        print(f'{split}: excluded {len(removed_ids)} black images')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='LevirShipData')
    parser.add_argument(
        '--dataset-out', default='mmdetection/data/levir_ship_coco')
    parser.add_argument(
        '--work-dir', default='mmdetection/work_dirs/levir_dbss')
    parser.add_argument(
        '--variants', default=DEFAULT_VARIANTS,
        help=f"Comma-separated: {', '.join(VARIANTS)}")
    parser.add_argument('--image-size', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--pilot-calibrate', action='store_true')
    parser.add_argument('--pilot-iters', type=int, default=300)
    parser.add_argument('--target-displacement', type=float, default=0.014)
    parser.add_argument(
        '--black-inventory', default='black_images_inventory.csv')
    parser.add_argument('--keep-black-images', action='store_true')
    parser.add_argument('--hf-repo-id', default='duyle2408/fcos_dbss_falsification')
    parser.add_argument('--hf-repo-type', default='dataset')
    parser.add_argument('--hf-token', default='')
    parser.add_argument('--no-hf-upload', action='store_true')
    parser.add_argument('--num-machines', type=int, default=1)
    parser.add_argument('--machine-index', type=int, default=0)
    return parser.parse_args()


def write_config(
        variant: str,
        args: argparse.Namespace,
        dataset_out: Path,
        image_dir: Path,
        gamma_max: float | None = None,
        pilot_round: int | None = None) -> Path:
    root = str(levir.mmdet_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from mmengine.config import Config
    from mmdet.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(str(levir.mmdet_root() / VARIANTS[variant]))
    if variant in FALSIFICATION_VARIANTS:
        selector_mode, residual_mode = FALSIFICATION_VARIANTS[variant]
        cfg.model.neck.selector_mode = selector_mode
        cfg.model.neck.residual_mode = residual_mode
        cfg.model.neck.projection_mode = (
            'softmax' if residual_mode == 'softmax' else 'ridge')
    if gamma_max is not None:
        cfg.model.neck.gamma_max = gamma_max
    cfg = levir.patch_config(cfg, variant, args, dataset_out, image_dir)
    cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt = False
    for dataloader in (
            cfg.train_dataloader, cfg.val_dataloader, cfg.test_dataloader):
        set_resize_scale(dataloader.dataset.pipeline, args.image_size)
    for scheduler in cfg.param_scheduler:
        if scheduler.type == 'MultiStepLR':
            scheduler.end = args.epochs
            scheduler.milestones = [
                round(args.epochs * 2 / 3),
                round(args.epochs * 11 / 12)]
    cfg.dbss_experiment = dict(
        variant=variant,
        selector_mode=cfg.model.neck.get('selector_mode'),
        residual_mode=cfg.model.neck.get('residual_mode'),
        gamma_max=cfg.model.neck.gamma_max,
        image_size=args.image_size,
        feature_stride=8,
        seed=args.seed)
    if pilot_round is not None:
        warmup = [
            scheduler for scheduler in cfg.param_scheduler
            if scheduler.type == 'LinearLR'
        ]
        for scheduler in warmup:
            scheduler.end = min(50, args.pilot_iters)
        cfg.work_dir = str(
            Path(args.work_dir) / '_pilot' / variant / f'round_{pilot_round}')
        cfg.train_cfg = dict(
            type='IterBasedTrainLoop',
            max_iters=args.pilot_iters,
            val_interval=args.pilot_iters + 1)
        cfg.param_scheduler = warmup
        cfg.default_hooks.checkpoint.update(
            by_epoch=False, interval=args.pilot_iters, save_best=None)
        cfg.default_hooks.logger.interval = 1
    output = Path(cfg.work_dir) / 'patched_config.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output))
    return output


def latest_metrics(work_dir: Path) -> dict[str, float]:
    scalar_files = sorted(work_dir.glob('**/vis_data/scalars.json'))
    if not scalar_files:
        return {}
    metrics: dict[str, float] = {}
    for line in scalar_files[-1].read_text(encoding='utf-8').splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key, value in record.items():
            if isinstance(value, (int, float)) and (
                    key.startswith('coco/') or key.startswith('dbss_')
                    or key == 'loss_dbss_sep'):
                metrics[key] = float(value)
    return metrics


def recent_displacement(work_dir: Path, count: int = 50) -> float:
    values = []
    for scalar_file in sorted(work_dir.glob('**/vis_data/scalars.json')):
        for line in scalar_file.read_text(encoding='utf-8').splitlines():
            try:
                value = json.loads(line).get('dbss_displacement_ratio')
            except json.JSONDecodeError:
                continue
            if isinstance(value, (int, float)):
                values.append(float(value))
    if not values:
        raise RuntimeError(f'No displacement diagnostics in {work_dir}')
    return sum(values[-count:]) / min(count, len(values))


def calibrate_gamma(
        variant: str, args: argparse.Namespace, dataset_out: Path,
        image_dir: Path) -> float:
    gamma = 0.6
    records = []
    for pilot_round in (1, 2):
        config = write_config(
            variant, args, dataset_out, image_dir,
            gamma_max=gamma, pilot_round=pilot_round)
        command = [
            sys.executable, str(levir.mmdet_root() / 'tools/train.py'),
            str(config), '--work-dir', str(config.parent), '--auto-scale-lr']
        if args.amp:
            command.append('--amp')
        levir.run(command)
        observed = recent_displacement(config.parent)
        records.append(dict(
            round=pilot_round, gamma_max=gamma,
            displacement_ratio=observed))
        if 0.011 <= observed <= 0.017:
            break
        if pilot_round == 1:
            gamma = min(2.0, max(
                0.05,
                gamma * args.target_displacement / max(observed, 1e-12)))
    calibration_file = Path(args.work_dir) / variant / 'calibration.json'
    calibration_file.parent.mkdir(parents=True, exist_ok=True)
    calibration_file.write_text(json.dumps(
        dict(variant=variant, selected_gamma=gamma, pilots=records), indent=2))
    return gamma


def upload_variant(variant: str, args: argparse.Namespace) -> None:
    if args.no_hf_upload:
        return
    token = args.hf_token or os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError('HF_TOKEN is required unless --no-hf-upload is set')
    from huggingface_hub import HfApi
    work_dir = Path(args.work_dir) / variant
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.hf_repo_id, repo_type=args.hf_repo_type,
        exist_ok=True)
    api.upload_folder(
        folder_path=str(work_dir),
        path_in_repo=variant,
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type)


def _distribution(values: list[float]) -> dict[str, float]:
    import torch
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        'mean': float(tensor.mean()),
        'median': float(tensor.median()),
        'p10': float(torch.quantile(tensor, 0.1)),
        'p90': float(torch.quantile(tensor, 0.9)),
    }


def dbss_diagnostics(config: Path, checkpoint: Path) -> dict[str, Any]:
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(str(config))
    model = init_detector(cfg, str(checkpoint), device='cuda:0')
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if not hasattr(model.neck, 'forward_with_aux'):
        return dict(parameters=parameters)
    loader = Runner.build_dataloader(cfg.val_dataloader)
    values = {
        key: [] for key in (
            'displacement_ratio', 'basis_max_cosine',
            'basis_effective_rank', 'basis_count', 'gamma_mean', 'gamma_std',
            'residual_rms', 'gap_pre', 'gap_post', 'gap_gain',
            'active_ratio', 'direction_weight_ratio', 'ridge_retry',
            'ridge_lstsq_fallback')
    }
    basis_histogram: dict[str, int] = {}
    latencies = []
    p3_shape = None
    with torch.inference_mode():
        for raw_batch in loader:
            batch = model.data_preprocessor(raw_batch, training=False)
            samples = batch['data_samples']
            torch.cuda.synchronize()
            started = time.perf_counter()
            features, aux = model._extract_feat_with_dbss_aux(
                batch['inputs'], samples)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            objective = model.separation_objective(aux, samples)
            p3_shape = list(features[0].shape)
            values['displacement_ratio'].extend(
                aux['displacement_ratio_per_image'].float().cpu().tolist())
            for key in (
                    'basis_max_cosine', 'basis_effective_rank', 'basis_count',
                    'gamma_mean', 'gamma_std', 'residual_rms'):
                values[key].extend(aux[key].float().cpu().tolist())
            for key in ('gap_pre', 'gap_post', 'gap_gain', 'active_ratio'):
                values[key].append(float(objective[f'dbss_{key}']))
            values['direction_weight_ratio'].append(
                float(aux['direction_weight_ratio']))
            values['ridge_retry'].append(float(aux['ridge_retry']))
            values['ridge_lstsq_fallback'].append(
                float(aux['ridge_lstsq_fallback']))
            for count in aux['basis_count'].tolist():
                label = str(int(count))
                basis_histogram[label] = basis_histogram.get(label, 0) + 1
    total_images = sum(basis_histogram.values())
    return dict(
        parameters=parameters,
        batch_latency_seconds=_distribution(latencies),
        distributions={
            key: _distribution(metric_values)
            for key, metric_values in values.items()
        },
        basis_count_histogram=basis_histogram,
        single_basis_ratio=(
            basis_histogram.get('1', 0) / max(total_images, 1)),
        p3_shape=p3_shape)


def run_variant(
        variant: str,
        args: argparse.Namespace,
        dataset_out: Path,
        image_dir: Path,
        gamma_max: float | None = None) -> None:
    config = write_config(
        variant, args, dataset_out, image_dir, gamma_max=gamma_max)
    print(f'CONFIG {variant}: {config}')
    if args.dry_run:
        return
    work_dir = config.parent
    if not args.test_only:
        command = [
            sys.executable, str(levir.mmdet_root() / 'tools/train.py'),
            str(config), '--work-dir', str(work_dir), '--auto-scale-lr']
        if args.amp:
            command.append('--amp')
        levir.run(command)
    checkpoint = levir.find_checkpoint(work_dir)
    predictions = work_dir / 'test_results/predictions.pkl'
    predictions.parent.mkdir(parents=True, exist_ok=True)
    levir.run([
        sys.executable, str(levir.mmdet_root() / 'tools/test.py'),
        str(config), str(checkpoint), '--work-dir', str(predictions.parent),
        '--out', str(predictions)])
    summary = dict(
        variant=variant,
        checkpoint=str(checkpoint),
        metrics=latest_metrics(work_dir))
    if variant != 'baseline':
        summary['dbss'] = dbss_diagnostics(config, checkpoint)
    (work_dir / 'experiment_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    upload_variant(variant, args)


def main() -> None:
    args = parse_args()
    variants = comma_list(args.variants)
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    if args.num_machines < 1:
        raise ValueError('--num-machines must be positive')
    if not 0 <= args.machine_index < args.num_machines:
        raise ValueError('--machine-index must be in [0, num_machines)')
    dataset_out, image_dir = levir.prepare_coco_dataset(args)
    if not args.keep_black_images:
        exclude_black_images(
            dataset_out, levir.resolve_path(args.data_root),
            args.black_inventory)
    assigned = [
        variant for index, variant in enumerate(variants)
        if index % args.num_machines == args.machine_index]
    print(f'Assigned variants: {assigned}')
    for variant in assigned:
        gamma = None
        if args.pilot_calibrate and variant in FALSIFICATION_VARIANTS:
            gamma = calibrate_gamma(
                variant, args, dataset_out, image_dir)
        run_variant(
            variant, args, dataset_out, image_dir, gamma_max=gamma)


if __name__ == '__main__':
    main()
