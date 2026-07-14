"""Benchmark DGFE two-stage variants on a deterministic synthetic batch."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'mmdetection'), str(ROOT)]

from mmdet.registry import MODELS  # noqa: E402
from mmdet.structures import DetDataSample  # noqa: E402
from mmdet.utils import register_all_modules  # noqa: E402


def sample(size: int) -> DetDataSample:
    item = DetDataSample()
    item.set_metainfo(dict(
        img_shape=(size, size),
        ori_shape=(size, size),
        pad_shape=(size, size),
        scale_factor=(1.0, 1.0),
    ))
    item.gt_instances = InstanceData(
        bboxes=torch.tensor([[size * .25, size * .25,
                              size * .75, size * .75]]),
        labels=torch.zeros(1, dtype=torch.long))
    return item


def benchmark(config_path: str, size: int, iterations: int, warmup: int,
              device: str) -> dict:
    torch.manual_seed(42)
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model).to(device).train()
    inputs = torch.rand(1, 3, size, size, device=device)
    samples = [sample(size).to(device)]
    counts = dict(backbone=0, neck=0, rpn=0, roi=0, bbox=0)
    modules = dict(backbone=model.backbone, neck=model.neck)
    hooks = []
    for name, module in modules.items():
        hooks.append(module.register_forward_hook(
            lambda _m, _a, _o, name=name: counts.__setitem__(
                name, counts[name] + 1)))
    if hasattr(model, 'rpn_head'):
        rpn_loss = model.rpn_head.loss_and_predict
        roi_loss = model.roi_head.loss

        def counted_rpn(*args, **kwargs):
            counts['rpn'] += 1
            return rpn_loss(*args, **kwargs)

        def counted_roi(*args, **kwargs):
            counts['roi'] += 1
            return roi_loss(*args, **kwargs)

        model.rpn_head.loss_and_predict = counted_rpn
        model.roi_head.loss = counted_roi
        adapter = model.roi_head
    else:
        bbox_loss = model.bbox_head.loss

        def counted_bbox(*args, **kwargs):
            counts['bbox'] += 1
            return bbox_loss(*args, **kwargs)

        model.bbox_head.loss = counted_bbox
        adapter = model.bbox_head

    def run_once() -> float:
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        losses = model.loss(inputs, samples)
        model.parse_losses(losses)[0].backward()
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
        return time.perf_counter() - start

    for _ in range(warmup):
        run_once()
    counts.update(backbone=0, neck=0, rpn=0, roi=0, bbox=0)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    timings = [run_once() for _ in range(iterations)]
    for hook in hooks:
        hook.remove()
    stage_records = (adapter.dgfe_stage_records()
                     if hasattr(adapter, 'dgfe_stage_records') else {})
    return dict(
        config=str(config_path),
        detector=type(model).__name__,
        head=type(adapter).__name__,
        mean_seconds=sum(timings) / len(timings),
        peak_cuda_bytes=(torch.cuda.max_memory_allocated(device)
                         if device.startswith('cuda') else None),
        call_counts=counts,
        assignment_records=len(adapter.dgfe_assignment_records()),
        stage_record_counts={str(k): len(v) for k, v in stage_records.items()},
        replay_supported=(hasattr(model, 'dgfe_replay_losses')
                          or not hasattr(model, 'rpn_head')),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    args = parser.parse_args()
    register_all_modules(init_default_scope=True)
    import dgfe_project  # noqa: F401
    results = [benchmark(path, args.size, args.iterations, args.warmup,
                         args.device)
               for path in args.configs]
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
