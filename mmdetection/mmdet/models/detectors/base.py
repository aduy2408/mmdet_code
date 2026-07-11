# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import torch
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

    @staticmethod
    def sum_loss_dict(losses: dict) -> Tensor:
        total = None
        for name, value in losses.items():
            if 'loss' not in name:
                continue
            if isinstance(value, Tensor):
                loss = value.mean()
            elif isinstance(value, (list, tuple)):
                loss = sum(v.mean() for v in value)
            else:
                continue
            total = loss if total is None else total + loss
        if total is None:
            raise RuntimeError('No differentiable loss found for API.')
        return total

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

    def api_augmented_losses(self, batch_inputs: Tensor,
                             batch_data_samples: SampleList,
                             clean_losses: dict,
                             compute_losses):
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
            clean_total = self.sum_loss_dict(clean_losses)
            target = self.build_api_target(batch_inputs, batch_data_samples,
                                           api.captured, api.target_mode)
            aux_loss = api.auxiliary_loss(target)
            grad = torch.autograd.grad(
                clean_total + aux_loss,
                api.captured,
                retain_graph=True,
                allow_unused=True)[0]
            if not api.set_perturbation_from_grad(grad):
                return clean_losses

            self.perturb_api()
            adv_losses = compute_losses()
            weight = api.current_api_weight
            clean_losses.update(self.scale_loss_dict(adv_losses, weight,
                                                     'api_adv_'))
            clean_losses['loss_api_aux'] = aux_loss * weight
            return clean_losses
        finally:
            self.clear_api_state()
