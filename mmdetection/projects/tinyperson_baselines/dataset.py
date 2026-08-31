"""Corner-aware TinyPerson dataset and image loader.

TinyBenchmark stores tiled samples as COCO image records that retain the source
file name and add ``corner=[x1, y1, x2, y2]``. Standard COCO loading ignores
that field and would train on the full image with tile-relative boxes.
"""

from __future__ import annotations

from typing import Union

from mmcv.transforms import LoadImageFromFile

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class TinyPersonDataset(CocoDataset):
    """COCO dataset that retains TinyBenchmark tile coordinates."""

    METAINFO = dict(classes=("person",), palette=[(220, 20, 60)])

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, list[dict]]:
        data_info = super().parse_data_info(raw_data_info)
        corner = raw_data_info["raw_img_info"].get("corner")
        if corner is not None:
            data_info["corner"] = corner
        return data_info


@TRANSFORMS.register_module()
class LoadTinyPersonImageFromFile(LoadImageFromFile):
    """Load a source image and crop the tile described by ``corner``."""

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        corner = results.get("corner")
        if corner is None:
            return results

        x1, y1, x2, y2 = map(int, corner)
        image = results["img"]
        if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
            raise ValueError(
                f"Invalid TinyPerson corner {corner} for image shape {image.shape[:2]}"
            )
        image = image[y1:y2, x1:x2]
        results["img"] = image
        results["img_shape"] = image.shape[:2]
        results["ori_shape"] = image.shape[:2]
        return results
