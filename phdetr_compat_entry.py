#!/usr/bin/env python3
"""Run PH-DETR after adapting its tuple transforms to torchvision 0.28."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch
import torchvision

sys.path.insert(0, str(Path.cwd()))
import src.data._misc as _misc
from src.data.transforms._transforms import ConvertBoxes, ConvertPILImage


def _convert_pil_forward(self, *inputs):
    sample = inputs if len(inputs) > 1 else inputs[0]
    image, target, dataset = sample
    image = torchvision.transforms.v2.functional.pil_to_tensor(image)
    if self.dtype == "float32":
        image = image.float()
    if self.scale:
        image = image / 255.0
    return _misc.Image(image), target, dataset


def _convert_boxes_forward(self, *inputs):
    sample = inputs if len(inputs) > 1 else inputs[0]
    image, target, dataset = sample
    boxes = target.get("boxes")
    if boxes is not None and self.fmt:
        spatial_size = getattr(boxes, _misc._boxes_keys[1])
        boxes = torchvision.ops.box_convert(
            boxes, in_fmt=boxes.format.value.lower(), out_fmt=self.fmt.lower()
        )
        boxes = _misc.convert_to_tv_tensor(
            boxes,
            key="boxes",
            box_format=self.fmt.upper(),
            spatial_size=spatial_size,
        )
        if self.normalize:
            boxes = boxes / torch.tensor(spatial_size[::-1]).tile(2)[None]
        target["boxes"] = boxes
    return image, target, dataset


ConvertPILImage.forward = _convert_pil_forward
ConvertBoxes.forward = _convert_boxes_forward

runpy.run_path(str(Path.cwd() / "train.py"), run_name="__main__")
