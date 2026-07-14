# Copyright (c) OpenMMLab. All rights reserved.
import math
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseModel
from torch import Tensor

from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig
from ..utils import samplelist_boxtype2tensor

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


class BaseDetector(BaseModel, metaclass=ABCMeta):
    """Base class for detectors.

    Args:
       data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self) -> bool:
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self) -> bool:
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self) -> bool:
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    @abstractmethod
    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        pass

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples

    def neck_forward(self, feats, batch_inputs: Tensor):
        """Forward necks that optionally need the input image tensor."""
        if not self.with_neck:
            return feats
        if getattr(self.neck, 'needs_batch_inputs', False):
            return self.neck(feats, batch_inputs=batch_inputs)
        return self.neck(feats)

    def set_epoch(self, epoch: int) -> None:
        if self.with_neck and hasattr(self.neck, 'set_epoch'):
            self.neck.set_epoch(epoch)

    def api_modules(self) -> list:
        if not self.with_neck:
            return []
        return getattr(self.neck, 'api_modules', [])

    def has_api_loss(self) -> bool:
        return (bool(self.api_modules()) and self.training
                and torch.is_grad_enabled())

    def clear_api_state(self) -> None:
        if hasattr(self.neck, 'clear_api_state'):
            self.neck.clear_api_state()

    def capture_api(self) -> None:
        if hasattr(self.neck, 'capture_api'):
            self.neck.capture_api()

    def perturb_api(self) -> None:
        if hasattr(self.neck, 'perturb_api'):
            self.neck.perturb_api()

    def dgfe_aux_list(self) -> list:
        if not self.with_neck or not hasattr(self.neck, 'dgfe_aux_list'):
            return []
        return self.neck.dgfe_aux_list()

    def dgfe_adapters(self) -> list:
        adapters = []
        for name in ('bbox_head', 'roi_head', 'rpn_head'):
            module = getattr(self, name, None)
            if module is not None and hasattr(module, 'build_dgfe_spatial_target'):
                adapters.append(module)
        return adapters

    @staticmethod
    def _loss_terms(losses: dict, names: set[str] | None = None) -> Tensor:
        total = None
        for name, value in losses.items():
            if 'loss' not in name or (names is not None and name not in names):
                continue
            if isinstance(value, Tensor):
                loss = value.mean()
            elif isinstance(value, (list, tuple)):
                loss = sum(v.mean() for v in value)
            else:
                continue
            total = loss if total is None else total + loss
        if total is None:
            raise RuntimeError('No differentiable loss found.')
        return total

    @staticmethod
    def sum_loss_dict(losses: dict) -> Tensor:
        return BaseDetector._loss_terms(losses)

    @staticmethod
    def is_boxgrad_mode(target_mode: str) -> bool:
        return str(target_mode).lower() in {
            'boxgrad', 'locgrad', 'localization', 'bbox', 'box'
        }

    @staticmethod
    def is_localization_loss_name(name: str) -> bool:
        name_l = name.lower()
        if 'loss' not in name_l:
            return False
        if any(skip in name_l for skip in (
                'cls', 'class', 'centerness', 'object', 'obj', 'api',
                'dgfe')):
            return False
        return any(key in name_l for key in (
            'bbox', 'box', 'iou', 'giou', 'diou', 'ciou', 'dfl', 'reg',
            'l1'))

    @classmethod
    def localization_loss_names(cls, losses: dict) -> set[str]:
        return {
            name
            for name, value in losses.items()
            if cls.is_localization_loss_name(name)
            and isinstance(value, (Tensor, list, tuple))
        }

    def dgfe_localization_loss_names(self, losses: dict) -> set[str]:
        names = set()
        for adapter in self.dgfe_adapters():
            getter = getattr(adapter, 'dgfe_localization_loss_names', None)
            if getter is not None:
                names.update(getter(losses))
        return names or self.localization_loss_names(losses)

    @staticmethod
    def scale_loss_dict(losses: dict, weight: float, prefix: str) -> dict:
        scaled = {}
        for name, value in losses.items():
            key = f'{prefix}{name}'
            if isinstance(value, Tensor):
                scaled[key] = value * weight
            elif isinstance(value, list):
                scaled[key] = [v * weight for v in value]
            elif isinstance(value, tuple):
                scaled[key] = tuple(v * weight for v in value)
        return scaled

    @staticmethod
    def scale_selected_loss_dict(losses: dict, weight: float,
                                 prefix: str, names: set[str]) -> dict:
        return BaseDetector.scale_loss_dict(
            {name: value for name, value in losses.items() if name in names},
            weight, prefix)

    @staticmethod
    def _normalize_inputs(batch_inputs: Tensor) -> Tensor:
        img = batch_inputs
        img_min = img.amin(dim=(2, 3), keepdim=True)
        img_max = img.amax(dim=(2, 3), keepdim=True)
        return (img - img_min) / (img_max - img_min).clamp(min=1e-6)

    @staticmethod
    def build_api_target(batch_inputs: Tensor,
                         batch_data_samples: SampleList,
                         feature: Tensor,
                         target_mode: str = 'foreground') -> Tensor:
        target = feature.new_zeros((feature.shape[0], 1, feature.shape[2],
                                    feature.shape[3]))
        _, _, img_h, img_w = batch_inputs.shape
        ring_mode = target_mode in {'boundary', 'boundary_ring', 'ring'}
        for batch_idx, data_sample in enumerate(batch_data_samples):
            bboxes = data_sample.gt_instances.bboxes
            bboxes = bboxes.tensor if hasattr(bboxes, 'tensor') else bboxes
            if bboxes.numel() == 0:
                continue
            boxes = bboxes.to(device=feature.device, dtype=feature.dtype)
            xs1 = (boxes[:, 0] / img_w * feature.shape[3]).floor().long()
            ys1 = (boxes[:, 1] / img_h * feature.shape[2]).floor().long()
            xs2 = (boxes[:, 2] / img_w * feature.shape[3]).ceil().long()
            ys2 = (boxes[:, 3] / img_h * feature.shape[2]).ceil().long()
            for x1, y1, x2, y2 in zip(xs1, ys1, xs2, ys2):
                x1 = int(x1.clamp(0, feature.shape[3] - 1))
                y1 = int(y1.clamp(0, feature.shape[2] - 1))
                x2 = int(x2.clamp(x1 + 1, feature.shape[3]))
                y2 = int(y2.clamp(y1 + 1, feature.shape[2]))
                if ring_mode:
                    target[
                        batch_idx, :,
                        max(0, y1 - 1):min(feature.shape[2], y2 + 1),
                        max(0, x1 - 1):min(feature.shape[3], x2 + 1)] = 1
                    if x2 - x1 > 2 and y2 - y1 > 2:
                        target[batch_idx, :, y1 + 1:y2 - 1, x1 + 1:x2 - 1] = 0
                else:
                    target[batch_idx, :, y1:y2, x1:x2] = 1
        return target

    @staticmethod
    def _slice_from_box(start: float, end: float, limit: int) -> tuple[int, int]:
        start_i = max(0, min(limit - 1, int(math.floor(start))))
        end_i = max(start_i + 1, min(limit, int(math.ceil(end))))
        return start_i, end_i

    def build_dgfe_spatial_target(self, logits: Tensor,
                                  batch_inputs: Tensor,
                                  batch_data_samples: SampleList) -> Tensor:
        for adapter in self.dgfe_adapters():
            target = adapter.build_dgfe_spatial_target(
                self, logits, batch_inputs, batch_data_samples)
            if target is not None:
                return target
        target = torch.zeros_like(logits)
        _, _, img_h, img_w = batch_inputs.shape
        feat_h, feat_w = logits.shape[-2:]
        ring = max(float(getattr(self.neck, 'dgfe_boundary_ring', 1.0)), 0.0)
        inner_value = max(
            min(float(getattr(self.neck, 'dgfe_inner_value', 0.3)), 1.0),
            0.0)
        tiny_area = float(getattr(self.neck, 'dgfe_tiny_area', 4.0))

        def write_max(region: Tensor, value: float) -> None:
            if region.numel():
                region.copy_(torch.maximum(
                    region, torch.full_like(region, value)))

        for batch_idx, data_sample in enumerate(batch_data_samples):
            bboxes = data_sample.gt_instances.bboxes
            bboxes = bboxes.tensor if hasattr(bboxes, 'tensor') else bboxes
            if bboxes.numel() == 0:
                continue
            boxes = bboxes.to(device=logits.device, dtype=logits.dtype)
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box]
                fx1, fx2 = x1 * feat_w / img_w, x2 * feat_w / img_w
                fy1, fy2 = y1 * feat_h / img_h, y2 * feat_h / img_h
                if fx2 <= fx1 or fy2 <= fy1:
                    continue
                ix1, ix2 = self._slice_from_box(fx1, fx2, feat_w)
                iy1, iy2 = self._slice_from_box(fy1, fy2, feat_h)
                if (ix2 - ix1) * (iy2 - iy1) <= tiny_area:
                    write_max(target[batch_idx, :, iy1:iy2, ix1:ix2], 1.0)
                    continue
                write_max(target[batch_idx, :, iy1:iy2, ix1:ix2],
                          inner_value)
                edge = max(int(math.ceil(ring)), 1)
                write_max(target[batch_idx, :, iy1:min(iy1 + edge, iy2),
                                 ix1:ix2], 1.0)
                write_max(target[batch_idx, :, max(iy2 - edge, iy1):iy2,
                                 ix1:ix2], 1.0)
                write_max(target[batch_idx, :, iy1:iy2,
                                 ix1:min(ix1 + edge, ix2)], 1.0)
                write_max(target[batch_idx, :, iy1:iy2,
                                 max(ix2 - edge, ix1):ix2], 1.0)

                ox1, ox2 = self._slice_from_box(fx1 - ring, fx2 + ring,
                                                feat_w)
                oy1, oy2 = self._slice_from_box(fy1 - ring, fy2 + ring,
                                                feat_h)
                write_max(target[batch_idx, :, oy1:iy1, ox1:ox2], 1.0)
                write_max(target[batch_idx, :, iy2:oy2, ox1:ox2], 1.0)
                write_max(target[batch_idx, :, iy1:iy2, ox1:ix1], 1.0)
                write_max(target[batch_idx, :, iy1:iy2, ix2:ox2], 1.0)
        return target

    def dgfe_spatial_gain(self) -> float:
        gain = float(getattr(self.neck, 'dgfe_spatial_gain', 0.0))
        warmup_epochs = int(getattr(self.neck, 'dgfe_spatial_warmup_epochs',
                                    0))
        if warmup_epochs <= 0:
            return gain
        epoch = float(getattr(self.neck, 'dgfe_epoch', 0))
        t = min(max(epoch, 0.0) / max(float(warmup_epochs), 1.0), 1.0)
        start = float(getattr(self.neck, 'dgfe_spatial_warmup_start', 0.1))
        return gain * (start + (1.0 - start) * t)

    def add_dgfe_losses(self, losses: dict, batch_inputs: Tensor,
                        batch_data_samples: SampleList) -> dict:
        if not self.with_neck:
            return losses
        aux_list = self.dgfe_aux_list()
        if not aux_list:
            return losses
        rec_gain = float(getattr(self.neck, 'dgfe_rec_gain', 0.0))
        spatial_gain = self.dgfe_spatial_gain()
        if rec_gain > 0:
            rec_losses = []
            for aux in aux_list:
                recon = aux.get('recon')
                if recon is None:
                    continue
                rec_target = aux.get('image_target')
                if rec_target is None:
                    rec_target = self._normalize_inputs(batch_inputs)
                rec_target = rec_target.to(
                    device=recon.device, dtype=recon.dtype)
                if rec_target.shape[-2:] != recon.shape[-2:]:
                    rec_target = F.interpolate(
                        rec_target,
                        size=recon.shape[-2:],
                        mode='bilinear',
                        align_corners=False)
                rec_losses.append(F.smooth_l1_loss(recon, rec_target))
            if rec_losses:
                losses['loss_dgfe_rec'] = torch.stack(rec_losses).mean(
                ) * rec_gain
        if spatial_gain > 0:
            spatial_losses = []
            neg_ratio = max(int(getattr(self.neck, 'dgfe_neg_pos_ratio', 3)),
                            1)
            neg_gain = float(getattr(self.neck, 'dgfe_neg_gain', 0.25))
            for aux in aux_list:
                logits = aux.get('spatial_logits')
                if logits is None:
                    continue
                target = self.build_dgfe_spatial_target(
                    logits, batch_inputs, batch_data_samples)
                target = target.to(dtype=logits.dtype)
                bce = F.binary_cross_entropy_with_logits(
                    logits, target, reduction='none')
                pos_mask = target > 0
                if not pos_mask.any():
                    spatial_losses.append(logits.sum() * 0.0)
                    continue
                pos_loss = bce[pos_mask].mean()
                neg = bce[~pos_mask]
                neg_loss = logits.sum() * 0.0
                if neg.numel():
                    k = min(int(pos_mask.sum().item()) * neg_ratio,
                            neg.numel())
                    neg_loss = neg.topk(k).values.mean()
                spatial_losses.append(pos_loss + neg_gain * neg_loss)
            if spatial_losses:
                losses['loss_dgfe_spatial'] = torch.stack(
                    spatial_losses).mean() * spatial_gain
        return losses

    def api_augmented_losses(self, batch_inputs: Tensor,
                             batch_data_samples: SampleList,
                             clean_losses: dict,
                             compute_losses,
                             replay_losses=None):
        api_modules = self.api_modules()
        if not api_modules:
            return clean_losses
        api = api_modules[0]
        if api.current_rho == 0 or api.current_api_weight == 0:
            self.clear_api_state()
            return clean_losses

        try:
            if api.captured is None:
                return clean_losses
            boxgrad = self.is_boxgrad_mode(api.target_mode)
            loc_names = self.dgfe_localization_loss_names(clean_losses)
            clean_total = (self._loss_terms(clean_losses, loc_names)
                           if boxgrad and loc_names else
                           self.sum_loss_dict(clean_losses))
            target = self.build_api_target(batch_inputs, batch_data_samples,
                                           api.captured, api.target_mode)
            aux_loss = (api.captured.sum() * 0.0
                        if boxgrad else api.auxiliary_loss(target))
            grad = torch.autograd.grad(
                clean_total + aux_loss,
                api.captured,
                retain_graph=True,
                allow_unused=True)[0]
            if not api.set_perturbation_from_grad(grad, batch_inputs,
                                                  batch_data_samples):
                return clean_losses

            self.perturb_api()
            perturbed_features = None
            if replay_losses is not None and hasattr(self.neck,
                                                      'perturb_features'):
                perturbed_features = self.neck.perturb_features()
            adv_losses = (replay_losses(perturbed_features)
                          if perturbed_features is not None
                          else compute_losses())
            weight = api.current_api_weight
            if boxgrad and loc_names:
                clean_losses.update(
                    self.scale_selected_loss_dict(adv_losses, weight,
                                                  'api_adv_', loc_names))
            else:
                clean_losses.update(
                    self.scale_loss_dict(adv_losses, weight, 'api_adv_'))
                clean_losses['loss_api_aux'] = aux_loss * weight
            return clean_losses
        finally:
            self.clear_api_state()
