from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


def _tensor_stats(tensor):
    tensor = tensor.detach().float()
    return dict(
        min=float(tensor.min()),
        max=float(tensor.max()),
        rms=float(tensor.square().mean().sqrt()),
        finite=bool(torch.isfinite(tensor).all()))


@HOOKS.register_module()
class NonFiniteDiagnosticHook(Hook):
    """Stop at the first non-finite training value and save its context."""

    priority = 'VERY_LOW'

    def __init__(self, report_name='nonfinite_report.json'):
        self.report_name = report_name
        self.current = {}
        self.previous_state = None
        self.handles = []

    def before_run(self, runner):
        model = runner.model.module if hasattr(
            runner.model, 'module') else runner.model

        def capture(name):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(tensor):
                    self.current[name] = _tensor_stats(tensor)
            return hook

        self.handles.append(
            model.bbox_head.conv_cls.register_forward_hook(
                capture('classification_logits')))
        if hasattr(model.neck, 'embedding'):
            self.handles.append(
                model.neck.embedding.register_forward_hook(
                    capture('embedding')))
        if hasattr(model.neck, 'direction'):
            self.handles.append(
                model.neck.direction[-1].register_forward_hook(
                    capture('direction')))

        def capture_targets(_module, args, kwargs):
            prediction, target = args[:2]
            self.current['classification_logits'] = _tensor_stats(prediction)
            self.current['positive_locations'] = int(
                ((target >= 0) & (
                    target < model.bbox_head.num_classes)).sum())

        self.handles.append(
            model.bbox_head.loss_cls.register_forward_pre_hook(
                capture_targets, with_kwargs=True))

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        model = runner.model.module if hasattr(
            runner.model, 'module') else runner.model
        samples = data_batch['data_samples']
        gt_counts = [len(sample.gt_instances) for sample in samples]
        self.current = dict(
            iteration=int(runner.iter + 1),
            filenames=[
                sample.metainfo.get('img_path', '') for sample in samples
            ],
            gt_per_image=gt_counts,
            images_with_gt=sum(count > 0 for count in gt_counts),
            total_gt=sum(gt_counts))
        self.previous_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    @staticmethod
    def _module_norms(model, attribute):
        groups = {
            'backbone': model.backbone,
            'neck': model.neck,
            'head': model.bbox_head,
        }
        if hasattr(model.neck, 'direction'):
            groups['dbss_direction'] = model.neck.direction
            groups['dbss_magnitude'] = model.neck.magnitude
            groups['dbss_embedding'] = model.neck.embedding
        output = {}
        for name, module in groups.items():
            tensors = [
                getattr(parameter, attribute)
                for parameter in module.parameters()
                if getattr(parameter, attribute) is not None
            ]
            squared = sum(
                tensor.detach().float().square().sum() for tensor in tensors)
            output[name] = (
                math.sqrt(float(squared)) if tensors else 0.0)
        return output

    def after_train_iter(
            self, runner, batch_idx, data_batch=None, outputs=None):
        model = runner.model.module if hasattr(
            runner.model, 'module') else runner.model
        losses = {
            key: float(value)
            for key, value in outputs.items()
            if key.startswith('loss')
        }
        self.current['raw_losses'] = losses
        self.current['gradient_norms'] = self._module_norms(model, 'grad')
        self.current['parameter_norms'] = self._module_norms(model, 'data')
        nonfinite = (
            any(not math.isfinite(value) for value in losses.values())
            or any(not math.isfinite(value)
                   for value in self.current['gradient_norms'].values())
            or any(not math.isfinite(value)
                   for value in self.current['parameter_norms'].values())
            or any(
                isinstance(value, dict) and not value.get('finite', True)
                for value in self.current.values()))
        if not nonfinite:
            return
        work_dir = Path(runner.work_dir)
        report = work_dir / self.report_name
        report.write_text(
            json.dumps(self.current, indent=2), encoding='utf-8')
        torch.save(
            dict(state_dict=self.previous_state, meta=self.current),
            work_dir / 'last_finite_before_failure.pth')
        raise FloatingPointError(
            f'Non-finite training value; diagnostic saved to {report}')

    def after_run(self, runner):
        for handle in self.handles:
            handle.remove()
