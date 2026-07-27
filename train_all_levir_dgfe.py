#!/usr/bin/env python3
"""Prepare LEVIR-Ship as COCO and train MMDetection models with DGFE."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from typing import Any

import train_all_levir_baseline as baseline


_base_parse_args = baseline.parse_args
_base_patch_config = baseline.patch_config


def parse_args() -> argparse.Namespace:
    """Extend the baseline CLI with DGFE options."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dgfe-levels", type=int, nargs="+", default=[0])
    parser.add_argument("--dgfe-reduction", type=int, default=8)
    parser.add_argument("--dgfe-threshold", type=float, default=0.0156862)
    parser.add_argument("--dgfe-sharpness", type=float, default=10.0)
    parser.add_argument("--dgfe-alpha-init", type=float, default=1e-3)
    parser.add_argument("--dgfe-alpha-max", type=float, default=1.0)
    parser.add_argument("--dgfe-recon-ratio", type=float, default=0.5)
    parser.add_argument("--dgfe-upsample-steps", type=int, default=1)
    dgfe_args, baseline_argv = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *baseline_argv]
        args = _base_parse_args()
    finally:
        sys.argv = original_argv

    for name, value in vars(dgfe_args).items():
        setattr(args, name, value)
    if not any(
        arg == "--work-dir" or arg.startswith("--work-dir=")
        for arg in original_argv[1:]
    ):
        args.work_dir = "mmdetection/work_dirs/levir_dgfe"
    return args


def patch_config(
    cfg: Any,
    model_name: str,
    args: argparse.Namespace,
    dataset_out: Any,
    image_dir: Any,
) -> Any:
    """Apply the baseline LEVIR config and add DGFE to the requested FPN levels."""
    cfg = _base_patch_config(
        cfg, model_name, args, dataset_out, image_dir
    )
    base_neck = deepcopy(cfg.model.neck)
    if isinstance(base_neck, list):
        out_channels = base_neck[-1].get(
            "out_channels", base_neck[0].get("out_channels", 256)
        )
    else:
        out_channels = base_neck.get("out_channels", 256)
    cfg.model.neck = dict(
        type="FeatureAugmentNeck",
        base_neck=base_neck,
        out_channels=out_channels,
        levels=tuple(args.dgfe_levels),
        dgfe=dict(
            type="FeatureDGFE",
            reduction=args.dgfe_reduction,
            threshold_init=args.dgfe_threshold,
            sharpness=args.dgfe_sharpness,
            alpha_init=args.dgfe_alpha_init,
            alpha_max=args.dgfe_alpha_max,
            recon_ratio=args.dgfe_recon_ratio,
            upsample_steps=args.dgfe_upsample_steps,
        ),
        api=None,
    )
    return cfg


baseline.parse_args = parse_args
baseline.patch_config = patch_config


if __name__ == "__main__":
    baseline.main()
