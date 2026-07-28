from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.detectors import FCOS
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList


def _box_tensor(boxes) -> Tensor:
    return boxes.tensor if hasattr(boxes, 'tensor') else boxes


@MODELS.register_module()
class DBSSFCOS(FCOS):
    """FCOS with DBSS-enhanced P3 and a separation objective."""

    def __init__(
            self,
            *args,
            position_stride: int = 8,
            improvement_margin: float = 0.03,
            loss_sep_weight: float = 0.5,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self.neck, 'forward_with_aux'):
            raise TypeError('DBSSFCOS requires a neck with forward_with_aux()')
        if position_stride < 1:
            raise ValueError('position_stride must be positive')
        if improvement_margin < 0 or loss_sep_weight < 0:
            raise ValueError(
                'improvement_margin and loss_sep_weight must be non-negative')
        self.position_stride = int(position_stride)
        self.improvement_margin = float(improvement_margin)
        self.loss_sep_weight = float(loss_sep_weight)

    def _valid_shapes(self, batch_data_samples: SampleList
                      ) -> list[tuple[int, int]]:
        return [
            (math.ceil(sample.metainfo['img_shape'][0] / self.position_stride),
             math.ceil(sample.metainfo['img_shape'][1] / self.position_stride))
            for sample in batch_data_samples
        ]

    def _extract_feat_with_dbss_aux(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> tuple[tuple[Tensor, ...], dict]:
        backbone_features = self.backbone(batch_inputs)
        return self.neck.forward_with_aux(
            backbone_features,
            valid_shapes=self._valid_shapes(batch_data_samples))

    @staticmethod
    def _sample_centers(feature: Tensor, boxes: Tensor,
                        image_shape: tuple[int, int]) -> Tensor:
        if boxes.numel() == 0:
            return feature.new_empty((0, feature.shape[1]))
        image_height, image_width = image_shape
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        x = centers[:, 0] / max(image_width - 1, 1) * 2 - 1
        y = centers[:, 1] / max(image_height - 1, 1) * 2 - 1
        grid = torch.stack((x, y), dim=-1).clamp(-1, 1)
        sampled = F.grid_sample(
            feature, grid.view(1, -1, 1, 2),
            mode='bilinear', padding_mode='border', align_corners=True)
        return sampled.squeeze(0).squeeze(-1).transpose(0, 1)

    def separation_objective(
            self, aux: dict,
            batch_data_samples: SampleList) -> dict[str, Tensor]:
        pre_p3 = aux['pre_p3']
        post_p3 = aux['post_p3']
        background_candidates = aux['selected_candidates_p3']
        hinge_values = []
        gap_pre_values = []
        gap_post_values = []

        for image_index, sample in enumerate(batch_data_samples):
            boxes = _box_tensor(sample.gt_instances.bboxes).to(
                device=pre_p3.device, dtype=pre_p3.dtype)
            if boxes.numel() == 0:
                continue
            image_shape = sample.metainfo['img_shape'][:2]
            pre_foreground = self._sample_centers(
                pre_p3[image_index:image_index + 1], boxes, image_shape)
            post_foreground = self._sample_centers(
                post_p3[image_index:image_index + 1], boxes, image_shape)
            foreground_centroid = pre_foreground.mean(dim=0).detach()
            background_centroid = (
                background_candidates[image_index].mean(dim=0).detach())
            expanded_foreground = foreground_centroid.expand_as(pre_foreground)
            expanded_background = background_centroid.expand_as(pre_foreground)
            gap_pre = (
                F.cosine_similarity(pre_foreground, expanded_foreground)
                - F.cosine_similarity(pre_foreground, expanded_background))
            gap_post = (
                F.cosine_similarity(post_foreground, expanded_foreground)
                - F.cosine_similarity(post_foreground, expanded_background))
            hinge_values.append(F.relu(
                self.improvement_margin - (gap_post - gap_pre)))
            gap_pre_values.append(gap_pre)
            gap_post_values.append(gap_post)

        if not hinge_values:
            zero = post_p3.sum() * 0
            return dict(
                loss_dbss_sep=zero,
                dbss_gap_pre=zero.detach(),
                dbss_gap_post=zero.detach(),
                dbss_gap_gain=zero.detach(),
                dbss_active_ratio=zero.detach())

        hinge = torch.cat(hinge_values)
        gap_pre = torch.cat(gap_pre_values)
        gap_post = torch.cat(gap_post_values)
        return dict(
            loss_dbss_sep=hinge.mean() * self.loss_sep_weight,
            dbss_gap_pre=gap_pre.mean().detach(),
            dbss_gap_post=gap_post.mean().detach(),
            dbss_gap_gain=(gap_post - gap_pre).mean().detach(),
            dbss_active_ratio=(hinge > 0).to(hinge.dtype).mean().detach())

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict[str, Tensor]:
        features, aux = self._extract_feat_with_dbss_aux(
            batch_inputs, batch_data_samples)
        losses = self.bbox_head.loss(features, batch_data_samples)
        losses.update(self.separation_objective(aux, batch_data_samples))
        losses.update(
            dbss_displacement_ratio=aux['displacement_ratio'].detach(),
            dbss_basis_count=aux['basis_count'].float().mean().detach(),
            dbss_basis_max_cosine=aux['basis_max_cosine'].mean().detach(),
            dbss_basis_effective_rank=aux[
                'basis_effective_rank'].mean().detach(),
            dbss_gamma_mean=aux['gamma_mean'].mean().detach(),
            dbss_gamma_std=aux['gamma_std'].mean().detach(),
            dbss_residual_rms=aux['residual_rms'].mean().detach(),
            dbss_ridge_retry=aux['ridge_retry'].detach(),
            dbss_ridge_lstsq_fallback=aux[
                'ridge_lstsq_fallback'].detach(),
            dbss_direction_weight_ratio=aux[
                'direction_weight_ratio'].detach())
        return losses

    def predict(
            self,
            batch_inputs: Tensor,
            batch_data_samples: SampleList,
            rescale: bool = True) -> SampleList:
        features, _ = self._extract_feat_with_dbss_aux(
            batch_inputs, batch_data_samples)
        results = self.bbox_head.predict(
            features, batch_data_samples, rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, results)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None):
        if batch_data_samples is None:
            return self.bbox_head.forward(self.extract_feat(batch_inputs))
        features, _ = self._extract_feat_with_dbss_aux(
            batch_inputs, batch_data_samples)
        return self.bbox_head.forward(features)
