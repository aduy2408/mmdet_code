"""MMCV-lite compatibility for the SET FCOS port on Blackwell GPUs.

The published mmcv-lite wheel omits ``mmcv._ext``.  MMDetection imports
optional ops eagerly, so provide the small subset needed by FCOS: NMS and
sigmoid focal loss.  Other optional ops fail clearly if selected.
"""
from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

import torch


def _nms(boxes, scores, iou_threshold=0.5, offset=0, **kwargs):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        index = order[0].item()
        keep.append(index)
        if order.numel() == 1:
            break
        box = boxes[index]
        rest = boxes[order[1:]]
        left = torch.maximum(box[0], rest[:, 0])
        top = torch.maximum(box[1], rest[:, 1])
        right = torch.minimum(box[2], rest[:, 2])
        bottom = torch.minimum(box[3], rest[:, 3])
        intersection = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
        area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        rest_area = ((rest[:, 2] - rest[:, 0]).clamp(min=0) *
                     (rest[:, 3] - rest[:, 1]).clamp(min=0))
        overlap = intersection / (area + rest_area - intersection + 1e-6)
        order = order[1:][overlap <= float(iou_threshold)]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def _sigmoid_focal_forward(input, target, weight, output, gamma=2.0, alpha=0.25,
                           **kwargs):
    with torch.no_grad():
        one_hot = torch.zeros_like(input)
        valid = (target >= 0) & (target < input.shape[1])
        rows = torch.arange(input.shape[0], device=input.device)[valid]
        one_hot[rows, target[valid]] = 1
        probability = input.sigmoid()
        probability_target = (probability * one_hot +
                              (1 - probability) * (1 - one_hot))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input, one_hot, reduction='none') * (1 - probability_target).pow(gamma)
        if alpha >= 0:
            loss = loss * (alpha * one_hot + (1 - alpha) * (1 - one_hot))
        if weight.numel():
            loss = loss * weight
        output.copy_(loss)


def _sigmoid_focal_backward(input, target, weight, grad_input, gamma=2.0,
                            alpha=0.25, **kwargs):
    with torch.enable_grad():
        value = input.detach().requires_grad_(True)
        one_hot = torch.zeros_like(value)
        valid = (target >= 0) & (target < value.shape[1])
        rows = torch.arange(value.shape[0], device=value.device)[valid]
        one_hot[rows, target[valid]] = 1
        probability = value.sigmoid()
        probability_target = probability * one_hot + (1 - probability) * (1 - one_hot)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            value, one_hot, reduction='none') * (1 - probability_target).pow(gamma)
        if alpha >= 0:
            loss = loss * (alpha * one_hot + (1 - alpha) * (1 - one_hot))
        if weight.numel():
            loss = loss * weight
        gradient = torch.autograd.grad(loss.sum(), value)[0]
    grad_input.copy_(gradient)


class _Extension(types.ModuleType):
    def __getattr__(self, name):
        if name == 'nms':
            return _nms
        if name == 'sigmoid_focal_loss_forward':
            return _sigmoid_focal_forward
        if name == 'sigmoid_focal_loss_backward':
            return _sigmoid_focal_backward
        if name in (
            'softmax_focal_loss_forward', 'softmax_focal_loss_backward',
            'active_rotated_filter_forward', 'active_rotated_filter_backward',
            'roi_align_forward', 'roi_align_backward',
        ):
            return lambda *args, **kwargs: None
        if name == 'nms_match':
            return lambda *args, **kwargs: []
        if name in ('nms_rotated', 'nms_quadri'):
            return lambda *args, **kwargs: torch.arange(
                args[0].shape[0], device=args[0].device)
        return lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError(f'Optional MMCV op unavailable: {name}'))


extension = _Extension('mmcv._ext')
extension.__file__ = 'mmcv/_ext.so'
extension.__spec__ = ModuleSpec('mmcv._ext', loader=None)
sys.modules.setdefault('mmcv._ext', extension)
