"""Standalone MuSGD port for the Hugging Face DETR runner.

The update equations mirror ``mmdetection/projects/set/optim/musgd.py``.
Parameter groups are split using the same project rule: 2D and 4D tensors use
the hybrid Muon plus SGD branch, while 1D and other tensors use SGD only.
"""

from __future__ import annotations

import torch
from torch import optim


def zeropower_via_newtonschulz5(matrix: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    shape = matrix.shape
    x = matrix.reshape(-1, shape[-2], shape[-1]).bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    transposed = shape[-2] > shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        gram = x @ x.transpose(-2, -1)
        cubic = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, cubic, x, beta=a)
    if transposed:
        x = x.transpose(-2, -1)
    return x.reshape(shape)


def muon_update(gradients, momentums, beta: float, nesterov: bool):
    torch._foreach_mul_(momentums, beta)
    torch._foreach_add_(momentums, gradients, alpha=1 - beta)
    if nesterov:
        updates = list(torch._foreach_mul(momentums, beta))
        torch._foreach_add_(updates, gradients, alpha=1 - beta)
    else:
        updates = list(momentums)

    buckets = {}
    for index, update in enumerate(updates):
        matrix = update.view(len(update), -1) if update.ndim > 2 else update
        transposed = matrix.size(0) > matrix.size(1)
        if transposed:
            matrix = matrix.transpose(0, 1)
        scale = max(1, gradients[index].size(-2) / gradients[index].size(-1)) ** 0.5
        key = (matrix.size(0), scale, matrix.device, matrix.dtype)
        buckets.setdefault(key, []).append((index, matrix, transposed))

    for items in buckets.values():
        width = max(matrix.size(1) for _, matrix, _ in items)
        batch = torch.stack([
            torch.nn.functional.pad(matrix, (0, width - matrix.size(1)))
            for _, matrix, _ in items
        ])
        batch = zeropower_via_newtonschulz5(batch).to(
            gradients[items[0][0]].dtype
        )
        for row, (index, matrix, transposed) in enumerate(items):
            value = batch[row, :, : matrix.size(1)]
            updates[index] = (value.T if transposed else value).reshape(
                gradients[index].shape
            )
    return updates


class MuSGD(optim.Optimizer):
    """MuSGD with the project's 2D/4D Muon parameter split."""

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        nesterov: bool = True,
        muon: float = 0.2,
        sgd: float = 1.0,
    ):
        incoming = list(params)
        expanded = []
        for item in incoming:
            if isinstance(item, dict):
                group = dict(item)
                group_params = list(group.pop("params"))
                explicit = group.pop("use_muon", None)
            else:
                group = {}
                group_params = [item]
                explicit = None
            buckets = (
                [(bool(explicit), group_params)]
                if explicit is not None
                else [
                    (True, [p for p in group_params if p.ndim in {2, 4}]),
                    (False, [p for p in group_params if p.ndim not in {2, 4}]),
                ]
            )
            for use_muon, selected in buckets:
                if selected:
                    part = dict(group)
                    part["params"] = selected
                    part["use_muon"] = use_muon
                    expanded.append(part)

        super().__init__(
            expanded,
            defaults={
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "nesterov": nesterov,
                "use_muon": False,
            },
        )
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [param for param in group["params"] if param.grad is not None]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            for param in params:
                if not self.state[param]:
                    self.state[param]["momentum_buffer"] = torch.zeros_like(param)
                    if group["use_muon"]:
                        self.state[param]["momentum_buffer_SGD"] = torch.zeros_like(param)
            if group["use_muon"]:
                updates = muon_update(
                    [param.grad for param in params],
                    [self.state[param]["momentum_buffer"] for param in params],
                    beta=momentum,
                    nesterov=nesterov,
                )
                torch._foreach_add_(params, updates, alpha=-(lr * self.muon))
                buffers = [self.state[param]["momentum_buffer_SGD"] for param in params]
                lr *= self.sgd
            else:
                buffers = [self.state[param]["momentum_buffer"] for param in params]
            gradients = [param.grad for param in params]
            if group["weight_decay"]:
                gradients = torch._foreach_add(
                    gradients, params, alpha=group["weight_decay"]
                )
            torch._foreach_mul_(buffers, momentum)
            torch._foreach_add_(buffers, gradients)
            updates = (
                torch._foreach_add(gradients, buffers, alpha=momentum)
                if nesterov
                else buffers
            )
            torch._foreach_add_(params, updates, alpha=-lr)
        return loss
