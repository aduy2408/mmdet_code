"""HF-side equivalent of the project's YOLO/MMDetection bbox pipeline.

The implementation keeps the same public protocol values as
``train_all_levir_baseline.yolo_pipeline``: Mosaic, RandomAffine with
translate=.1 and scale=.5..1.5, HSV deltas, horizontal flip, letterbox-style
resize/pad, and final invalid-box filtering. It is intentionally framework
agnostic so DETR and RT-DETR Transformers runners can share it.
"""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np
from PIL import Image


class YoloHFTransform:
    def __init__(self, image_size: int, *, train: bool, mosaic: bool, mosaic_close_epoch: int = 0):
        self.image_size = int(image_size)
        self.train = bool(train)
        self.mosaic = bool(mosaic)
        self.mosaic_close_epoch = int(mosaic_close_epoch)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _mosaic_enabled(self) -> bool:
        return self.train and self.mosaic and (
            self.mosaic_close_epoch <= 0 or self.epoch < self.mosaic_close_epoch
        )

    def __call__(self, image: Image.Image, boxes: list[list[float]], labels: list[int],
                 extra: Iterable[tuple[Image.Image, list[list[float]], list[int]]] = ()):
        image = image.convert("RGB")
        boxes = np.asarray(boxes, dtype=np.float32).reshape((-1, 4))
        labels = np.asarray(labels, dtype=np.int64)
        if self._mosaic_enabled():
            image, boxes, labels = self._mosaic(image, boxes, labels, list(extra))
        if self.train:
            image, boxes = self._random_affine(image, boxes, border=self.image_size // 2 if self._mosaic_enabled() else 0)
            image = self._hsv(image)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if len(boxes):
                    boxes[:, 0] = image.width - boxes[:, 0] - boxes[:, 2]
        image, boxes = self._letterbox(image, boxes)
        keep = (boxes[:, 2] >= 1.0) & (boxes[:, 3] >= 1.0) if len(boxes) else np.zeros(0, dtype=bool)
        return image, boxes[keep].tolist(), labels[keep].tolist()

    def _resize_fit(self, image: Image.Image, boxes: np.ndarray, target: int):
        scale = min(target / image.width, target / image.height)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, :2] *= scale
            boxes[:, 2:] *= scale
        return image, boxes

    def _mosaic(self, image, boxes, labels, extra):
        samples = [(image, boxes, labels)] + extra[:3]
        while len(samples) < 4:
            samples.append((image, boxes, labels))
        size = self.image_size
        canvas = Image.new("RGB", (size * 2, size * 2), (114, 114, 114))
        all_boxes, all_labels = [], []
        for index, (sample_image, sample_boxes, sample_labels) in enumerate(samples):
            tile, tile_boxes = self._resize_fit(sample_image.convert("RGB"), np.asarray(sample_boxes, dtype=np.float32).reshape((-1, 4)), size)
            x_offset = (index % 2) * size
            y_offset = (index // 2) * size
            canvas.paste(tile, (x_offset, y_offset))
            if len(tile_boxes):
                tile_boxes = tile_boxes.copy()
                tile_boxes[:, 0] += x_offset
                tile_boxes[:, 1] += y_offset
                all_boxes.append(tile_boxes)
                all_labels.extend(np.asarray(sample_labels, dtype=np.int64).tolist())
        merged_boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4), dtype=np.float32)
        return canvas, merged_boxes, np.asarray(all_labels, dtype=np.int64)

    def _random_affine(self, image, boxes, border: int):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("HF YOLO augmentation requires opencv-python") from exc
        width, height = image.size
        scale = random.uniform(0.5, 1.5)
        tx = random.uniform(-0.1, 0.1) * width + border
        ty = random.uniform(-0.1, 0.1) * height + border
        matrix = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
        array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        output_size = (width + 2 * border, height + 2 * border)
        array = cv2.warpAffine(array, matrix, output_size, borderValue=(114, 114, 114))
        if len(boxes):
            corners = np.stack([
                boxes[:, [0, 1]],
                boxes[:, [0, 1]] + boxes[:, [2, 3]],
                boxes[:, [0, 1]] + np.stack([boxes[:, 2], np.zeros(len(boxes))], axis=1),
                boxes[:, [0, 1]] + np.stack([np.zeros(len(boxes)), boxes[:, 3]], axis=1),
            ], axis=1)
            corners = corners * scale + np.array([tx, ty], dtype=np.float32)
            x1 = corners[:, :, 0].min(axis=1).clip(0, output_size[0])
            y1 = corners[:, :, 1].min(axis=1).clip(0, output_size[1])
            x2 = corners[:, :, 0].max(axis=1).clip(0, output_size[0])
            y2 = corners[:, :, 1].max(axis=1).clip(0, output_size[1])
            boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)), boxes

    def _hsv(self, image: Image.Image) -> Image.Image:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("HF YOLO augmentation requires opencv-python") from exc
        array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV).astype(np.int16)
        array[..., 0] = (array[..., 0] + random.randint(-3, 3)) % 180
        array[..., 1] = np.clip(array[..., 1] + random.randint(-179, 179), 0, 255)
        array[..., 2] = np.clip(array[..., 2] + random.randint(-102, 102), 0, 255)
        return Image.fromarray(cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_HSV2RGB))

    def _letterbox(self, image: Image.Image, boxes: np.ndarray):
        resized, boxes = self._resize_fit(image, boxes, self.image_size)
        canvas = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
        canvas.paste(resized, (0, 0))
        return canvas, boxes
