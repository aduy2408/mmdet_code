from argparse import Namespace
import json
from pathlib import Path

from mmengine.config import Config

import train_levir_dbss_detectors as sweep


def test_write_paired_detector_matrix(tmp_path: Path):
    dataset = tmp_path / 'coco'
    annotations = dataset / 'annotations'
    annotations.mkdir(parents=True)
    payload = dict(images=[], annotations=[], categories=[
        dict(id=1, name='ship', supercategory='ship')])
    for split in ('train', 'val', 'test'):
        (annotations / f'{split}.json').write_text(json.dumps(payload))

    args = Namespace(
        image_size=768,
        epochs=20,
        num_workers=0,
        seed=42,
        work_dir=str(tmp_path / 'runs'),
    )
    configs = {}
    for model in sweep.DEFAULT_MODELS:
        for variant in sweep.DEFAULT_VARIANTS:
            path = sweep.write_config(
                model, variant, args, dataset, tmp_path / 'images', 'abc123')
            configs[model, variant] = Config.fromfile(path)

    assert len(configs) == 8
    for model in sweep.DEFAULT_MODELS:
        baseline = configs[model, 'baseline']
        dbss = configs[model, 'dbss_gamma06']
        expected_stride = 4 if model.endswith('rcnn') else 8
        assert baseline.model.neck.get(
            'start_level', 0) == dbss.model.neck.get('start_level', 0)
        assert dbss.model.neck.target_level == 'lowest'
        assert dbss.model.neck.gamma_max == 0.6
        assert dbss.model.neck.legacy_artifact_mode is True
        assert dbss.optim_wrapper.clip_grad.max_norm == (
            sweep.DBSS_GRAD_MAX_NORM)
        assert 'clip_grad' not in baseline.optim_wrapper
        assert dbss.model.target_stride == expected_stride
        assert dbss.dbss_detector_sweep.target_pyramid_level == (
            'P2' if expected_stride == 4 else 'P3')
        assert dbss.train_dataloader.batch_size == sweep.MICRO_BATCH[model]
        assert dbss.optim_wrapper.accumulative_counts == (
            sweep.EFFECTIVE_BATCH // sweep.MICRO_BATCH[model])
        assert dbss.optim_wrapper.optimizer.lr == (
            0.005 if model in ('atss', 'retinanet') else 0.01)
        linear = next(
            scheduler for scheduler in dbss.param_scheduler
            if scheduler.type == 'LinearLR')
        assert linear.end == (
            sweep.WARMUP_OPTIMIZER_UPDATES
            * dbss.optim_wrapper.accumulative_counts)
        for dataloader in (
                dbss.train_dataloader, dbss.val_dataloader,
                dbss.test_dataloader):
            resize = next(
                step for step in dataloader.dataset.pipeline
                if step.type == 'Resize')
            assert tuple(resize.scale) == (768, 768)
