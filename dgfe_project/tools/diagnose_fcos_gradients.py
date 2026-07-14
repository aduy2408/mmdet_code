#!/usr/bin/env python3
"""Probe DGFE target and gradient interactions on one FCOS training batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

import dgfe_project  # noqa: F401
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


def loss_sum(losses: dict, names: set[str]) -> torch.Tensor:
    values = []
    for name in names:
        value = losses.get(name)
        if isinstance(value, torch.Tensor):
            values.append(value.mean())
        elif isinstance(value, (list, tuple)):
            values.extend(item.mean() for item in value)
    return sum(values)


def flat_grad(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(param) if grad is None else grad).reshape(-1)
        for param, grad in zip(params, grads)
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output', default='')
    args = parser.parse_args()

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).cuda().train()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    loader = Runner.build_dataloader(cfg.train_dataloader)
    data = model.data_preprocessor(next(iter(loader)), training=True)
    losses = model.loss(data['inputs'].cuda(), data['data_samples'])

    dgfe_params = [
        param for name, param in model.named_parameters()
        if 'dgfe_modules' in name and param.requires_grad
    ]
    groups = {
        'detection': {'loss_cls', 'loss_bbox', 'loss_centerness'},
        'reconstruction': {'loss_dgfe_rec'},
        'spatial': {'loss_dgfe_spatial'},
    }
    grads = {}
    for name, loss_names in groups.items():
        present = loss_names & set(losses)
        if dgfe_params and present:
            grads[name] = flat_grad(loss_sum(losses, present), dgfe_params)

    result: dict[str, object] = {
        'losses': {
            name: float(value.detach().mean())
            for name, value in losses.items() if isinstance(value, torch.Tensor)
        },
        'gradient_norms': {
            name: float(grad.norm()) for name, grad in grads.items()
        },
        'gradient_cosines': {},
    }
    for left in grads:
        for right in grads:
            if left >= right:
                continue
            denom = grads[left].norm() * grads[right].norm()
            cosine = 0.0 if denom == 0 else float(
                torch.dot(grads[left], grads[right]) / denom)
            result['gradient_cosines'][f'{left}:{right}'] = cosine

    aux_list = model.dgfe_aux_list()
    if aux_list:
        aux = aux_list[0]
        target = model.build_dgfe_spatial_target(
            aux['spatial_logits'], data['inputs'], data['data_samples'])
        result['spatial_target'] = {
            'positive_ratio': float((target > 0).float().mean()),
            'mean': float(target.mean()),
            'max': float(target.max()),
        }
        result['reconstruction_mae'] = float(
            (aux['recon'] - aux['image_target']).abs().mean())
        module = model.neck.dgfe_modules[str(model.neck.levels[0])]
        result['alpha'] = float(module.alpha.detach())
        result['threshold'] = float(module.threshold.detach())

    records = model.bbox_head.dgfe_assignment_records()
    if records:
        qualities = torch.tensor([record['quality'] for record in records])
        result['assignments'] = {
            'count': len(records),
            'iou_mean': float(qualities.mean()),
            'iou_max': float(qualities.max()),
        }
    api_modules = model.api_modules()
    if api_modules and api_modules[0].last_perturbation_norm is not None:
        result['api_perturbation_norm'] = float(
            api_modules[0].last_perturbation_norm.mean())

    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
