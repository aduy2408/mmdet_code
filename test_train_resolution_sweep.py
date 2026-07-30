from argparse import Namespace
from pathlib import Path

from mmengine.config import Config

import train_resolution_sweep as sweep


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
        epochs=30,
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
            assert cfg.train_dataloader.batch_size == (
                sweep.MICRO_BATCH[resolution])
            assert cfg.optim_wrapper.accumulative_counts == (
                sweep.ACCUMULATION[resolution])
            assert cfg.train_cfg.max_epochs == 30
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

    assert len(configs) == 8
    assert len({str(path.parent) for path in configs}) == 8
