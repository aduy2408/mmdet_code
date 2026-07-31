from argparse import Namespace
from pathlib import Path

from mmengine.config import Config

import train_all_highres_repair as sweep


def test_latest_validation_metrics_ignores_non_object_json(tmp_path: Path):
    scalar_file = tmp_path / 'scalars.json'
    scalar_file.write_text(
        '\n'.join([
            '1',
            'null',
            '[]',
            '{"coco/bbox_mAP": 0.28, "coco/bbox_mAP_75": 0.11}',
            '{"coco/bbox_mAP": 0.30, "coco/bbox_mAP_75": 0.12}',
        ]),
        encoding='utf-8',
    )

    assert sweep.latest_validation_metrics(tmp_path) == {
        'coco/bbox_mAP': 0.30,
        'coco/bbox_mAP_75': 0.12,
    }


def test_write_config_matrix(tmp_path: Path):
    dataset_out = tmp_path / 'dataset'
    annotation_dir = dataset_out / 'annotations'
    annotation_dir.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        (annotation_dir / f'{split}.json').write_text(
            '{"images":[],"annotations":[],"categories":[]}',
            encoding='utf-8')
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    args = Namespace(
        batch_size=8,
        num_workers=0,
        seed=42,
        work_dir=str(tmp_path / 'runs'),
    )

    configs = []
    for resolution in sweep.DEFAULT_RESOLUTIONS:
        for variant in sweep.DEFAULT_VARIANTS:
            path = sweep.write_config(
                variant,
                resolution,
                args,
                dataset_out,
                image_dir,
                'a' * 40,
            )
            cfg = Config.fromfile(path)
            configs.append(path)
            assert cfg.resolution_sweep.variant == variant
            assert cfg.resolution_sweep.resolution == resolution
            assert cfg.resolution_sweep.effective_batch == 8
            assert cfg.resolution_sweep.learning_rate == 0.005
            assert cfg.resolution_sweep.epochs == sweep.EPOCHS[variant]
            assert tuple(cfg.resolution_sweep.milestones) == (
                sweep.MILESTONES[variant])
            assert cfg.resolution_sweep.warmup_optimizer_updates == 500
            assert cfg.resolution_sweep.warmup_dataloader_iterations == (
                500 * sweep.ACCUMULATION[resolution])
            assert cfg.optim_wrapper.optimizer.lr == 0.005
            assert cfg.train_dataloader.batch_size == (
                sweep.MICRO_BATCH[resolution])
            assert cfg.optim_wrapper.accumulative_counts == (
                sweep.ACCUMULATION[resolution])
            assert cfg.train_cfg.max_epochs == sweep.EPOCHS[variant]
            assert cfg.randomness.seed == 42
            assert cfg.auto_scale_lr.enable is False
            assert all(
                tuple(step.scale) == (resolution, resolution)
                for loader in (
                    cfg.train_dataloader,
                    cfg.val_dataloader,
                    cfg.test_dataloader,
                )
                for step in loader.dataset.pipeline
                if step.type == 'Resize'
            )
            schedulers = {item.type: item for item in cfg.param_scheduler}
            assert schedulers['ConstantLR'].end == (
                500 * sweep.ACCUMULATION[resolution])
            assert tuple(schedulers['MultiStepLR'].milestones) == (
                sweep.MILESTONES[variant])
            assert schedulers['MultiStepLR'].end == sweep.EPOCHS[variant]
            manifest = path.parent / 'protocol_manifest.json'
            assert manifest.is_file()

    assert len(configs) == 8
    assert len({str(path.parent) for path in configs}) == 8


def test_dbss_uses_legacy_artifact_initialization(tmp_path: Path):
    dataset_out = tmp_path / 'dataset'
    annotation_dir = dataset_out / 'annotations'
    annotation_dir.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        (annotation_dir / f'{split}.json').write_text(
            '{"images":[],"annotations":[],"categories":[]}',
            encoding='utf-8')
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    args = Namespace(
        batch_size=8,
        num_workers=0,
        seed=42,
        work_dir=str(tmp_path / 'runs'),
    )
    path = sweep.write_config(
        'dbss_gamma06',
        1024,
        args,
        dataset_out,
        image_dir,
        'b' * 40,
    )
    cfg = Config.fromfile(path)
    assert cfg.model.neck.legacy_artifact_mode is True
    assert cfg.model.neck.gamma_max == 0.6
    assert cfg.model.neck.selector_mode == 'legacy_forced_k'
    assert cfg.model.neck.residual_mode == 'ridge'
