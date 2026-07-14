# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.structures import InstanceData

import dgfe_project  # noqa: F401
from dgfe_project.dgfe_heads.atss_dgfe import DGFEDenseHeadMixin
from dgfe_project.dgfe_heads.fcos_dgfe import FCOSDGFEHead
from dgfe_project.dgfe_core.spatial_targets import aligned_iou
from dgfe_project.dgfe_heads.two_stage_dgfe import DGFETwoStageRoIMixin
from train_all_mmdet import patch_dgfe_model_specific_head
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.necks.feature_augment_neck import (
    AdversarialPerturbationInjection, FeatureAugmentNeck)
from mmdet.structures import DetDataSample


class DummyNeck(nn.Module):

    def __init__(self, aux=None):
        super().__init__()
        self.dgfe_rec_gain = 1.0
        self.dgfe_spatial_gain = 1.0
        self.dgfe_boundary_ring = 1.0
        self.dgfe_inner_value = 0.2
        self.dgfe_tiny_area = 2.0
        self.dgfe_neg_pos_ratio = 2
        self.dgfe_neg_gain = 0.5
        self.dgfe_spatial_target_mode = 'iou'
        self.dgfe_edge_error_norm = 1.0
        self.dgfe_epoch = 0
        self._aux = aux or []

    def dgfe_aux_list(self):
        return self._aux


class DummyDetector(BaseDetector):

    def __init__(self, aux=None):
        super().__init__()
        self.neck = DummyNeck(aux)

    def loss(self, batch_inputs, batch_data_samples):
        return {}

    def predict(self, batch_inputs, batch_data_samples):
        return []

    def _forward(self, batch_inputs, batch_data_samples=None):
        return ()

    def extract_feat(self, batch_inputs):
        return ()


class DummyAdapter(DGFEDenseHeadMixin, nn.Module):

    def __init__(self):
        super().__init__()
        self._dgfe_assignment_records = [
            dict(batch_idx=0, level=0, gt_idx=0, quality=0.25)
        ]


class DummyAdapterDetector(DummyDetector):

    def __init__(self):
        super().__init__()
        self.bbox_head = DummyAdapter()
        self.neck.dgfe_spatial_target_mode = 'edge_error'


def _sample(boxes):
    data_sample = DetDataSample()
    data_sample.gt_instances = InstanceData(
        bboxes=torch.tensor(boxes, dtype=torch.float32),
        labels=torch.zeros(len(boxes), dtype=torch.long))
    return data_sample


def test_dgfe_spatial_target_and_losses():
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    recon = torch.zeros(1, 3, 32, 32, requires_grad=True)
    detector = DummyDetector([dict(recon=recon, spatial_logits=logits)])
    batch_inputs = torch.rand(1, 3, 32, 32)
    data_samples = [_sample([[8, 8, 24, 24], [0, 0, 4, 4]])]

    target = detector.build_dgfe_spatial_target(logits, batch_inputs,
                                                data_samples)
    assert target.shape == logits.shape
    assert target.max() == 1
    assert target[0, 0, 3, 3] == detector.neck.dgfe_inner_value
    assert target[0, 0, 0, 0] == 1

    losses = detector.add_dgfe_losses({}, batch_inputs, data_samples)
    assert set(losses) == {'loss_dgfe_rec', 'loss_dgfe_spatial'}
    assert losses['loss_dgfe_rec'].requires_grad
    assert losses['loss_dgfe_spatial'].requires_grad


def test_boxgrad_loss_name_filtering():
    losses = {
        'loss_cls': torch.tensor(1.0),
        'loss_bbox': torch.tensor(2.0),
        'loss_dfl': torch.tensor(3.0),
        'loss_dgfe_spatial': torch.tensor(4.0),
        'loss_api_aux': torch.tensor(5.0),
    }
    assert BaseDetector.is_boxgrad_mode('boxgrad')
    assert BaseDetector.localization_loss_names(losses) == {
        'loss_bbox', 'loss_dfl'
    }


def test_dgfe_adapter_edge_error_target():
    detector = DummyAdapterDetector()
    logits = torch.zeros(1, 1, 8, 8)
    batch_inputs = torch.rand(1, 3, 32, 32)
    data_samples = [_sample([[8, 8, 24, 24]])]

    target = detector.build_dgfe_spatial_target(logits, batch_inputs,
                                                data_samples)

    assert target.shape == logits.shape
    assert target.max() == 0.75
    assert detector.dgfe_localization_loss_names({
        'loss_cls': torch.tensor(1.0),
        'loss_bbox': torch.tensor(1.0),
    }) == {'loss_bbox'}


def test_patch_dgfe_model_specific_head():
    model = dict(bbox_head=dict(type='ATSSHead'))
    patch_dgfe_model_specific_head(model, 'atss')
    assert model['bbox_head']['type'] == 'ATSSDGFEHead'

    model = dict(bbox_head=dict(type='TOODHead'))
    patch_dgfe_model_specific_head(model, 'tood')
    assert model['bbox_head']['type'] == 'TOODDGFEHead'

    model = dict(bbox_head=dict(type='FCOSHead'))
    patch_dgfe_model_specific_head(model, 'fcos')
    assert model['bbox_head']['type'] == 'FCOSDGFEHead'

    model = dict(type='FasterRCNN', roi_head=dict(type='StandardRoIHead'))
    patch_dgfe_model_specific_head(model, 'faster_rcnn')
    assert model['type'] == 'DGFEFasterRCNN'
    assert model['roi_head']['type'] == 'DGFEStandardRoIHead'

    model = dict(type='CascadeRCNN', roi_head=dict(type='CascadeRoIHead'))
    patch_dgfe_model_specific_head(model, 'cascade_rcnn', hybrid=False)
    assert model['type'] == 'CascadeRCNN'
    assert model['roi_head']['type'] == 'DGFECascadeRoIHead'


class _IdentityCoder:

    def decode(self, priors, deltas):
        return priors + deltas


class _BBoxHead:
    reg_class_agnostic = True
    num_classes = 1
    bbox_coder = _IdentityCoder()


class _SamplingResult:

    def __init__(self, gt_idx=0):
        self.pos_inds = torch.tensor([0])
        self.pos_assigned_gt_inds = torch.tensor([gt_idx])
        self.pos_gt_labels = torch.tensor([0])
        self.pos_priors = torch.tensor([[4., 4., 12., 12.]])
        self.neg_priors = torch.tensor([[20., 20., 24., 24.]])
        self.pos_gt_bboxes = torch.tensor([[4., 4., 12., 12.]])

    @property
    def priors(self):
        return torch.cat((self.pos_priors, self.neg_priors))


def test_two_stage_exact_assignment_records():
    bbox_results = dict(
        bbox_pred=torch.zeros(2, 4))
    records = DGFETwoStageRoIMixin._records_from_bbox(
        2, _BBoxHead(), bbox_results, [_SamplingResult(gt_idx=3)])

    assert len(records) == 1
    assert records[0]['stage'] == 2
    assert records[0]['gt_idx'] == 3
    assert records[0]['quality'] == 1.0
    assert torch.equal(records[0]['roi'], torch.tensor([4., 4., 12., 12.]))


def test_two_stage_roi_only_boxgrad_names():
    losses = {
        'rpn_loss_bbox': torch.tensor(1.),
        'rpn_loss_cls': torch.tensor(1.),
        'loss_cls': torch.tensor(1.),
        'loss_bbox': torch.tensor(1.),
        's0.loss_bbox': torch.tensor(1.),
        's1.loss_bbox': torch.tensor(1.),
        's2.loss_bbox': torch.tensor(1.),
    }
    names = DGFETwoStageRoIMixin.dgfe_localization_loss_names(None, losses)
    assert names == {
        'loss_bbox', 's0.loss_bbox', 's1.loss_bbox', 's2.loss_bbox'
    }

    feature = torch.tensor(2., requires_grad=True)
    losses = {
        'loss_cls': feature * 7,
        'loss_bbox': feature.square(),
        'rpn_loss_bbox': feature * 11,
    }
    selected = DGFETwoStageRoIMixin.dgfe_localization_loss_names(None, losses)
    grad, = torch.autograd.grad(BaseDetector._loss_terms(losses, selected),
                                feature)
    assert grad == 4  # ROI bbox only: d(feature**2)/d(feature).


def test_api_feature_replay_is_non_mutating():
    api = AdversarialPerturbationInjection(channels=4, rho=1.0)
    api.train()
    api.perturbation = torch.arange(4, dtype=torch.float32).view(
        1, 4, 1, 1).expand(1, 4, 2, 2)
    clean = (torch.ones(1, 4, 2, 2, requires_grad=True),
             torch.full((1, 4, 1, 1), 2.0, requires_grad=True))

    neck = FeatureAugmentNeck.__new__(FeatureAugmentNeck)
    nn.Module.__init__(neck)
    neck.levels = (0, )
    neck.api_modules_by_level = nn.ModuleDict({'0': api})
    neck._api_clean_features = clean
    replay = neck.perturb_features()

    assert replay is not clean
    assert replay[0] is not clean[0]
    assert replay[1] is clean[1]
    assert torch.equal(clean[0], torch.ones_like(clean[0]))
    assert torch.equal(replay[0], clean[0] + api.perturbation)
    assert replay[0].grad_fn is not None  # Cached feature was not detached.


def test_api_replay_matches_forward_with_fgsm_dropout():
    api = AdversarialPerturbationInjection(
        channels=4, use_fgsm_dropout=True, fgsm_drop_rate=0.5)
    api.train()
    feature = torch.ones(1, 4, 2, 2)
    api.perturbation = torch.tensor([1., 2., 3., 4.]).view(
        1, 4, 1, 1).expand_as(feature)
    api.perturb()

    assert torch.equal(api(feature), api.apply_perturbation(feature))


def test_fcos_dgfe_exact_assignment_and_norm_scaling():
    head = FCOSDGFEHead(
        num_classes=1,
        in_channels=8,
        feat_channels=8,
        stacked_convs=1,
        strides=(8, ),
        regress_ranges=((-1, 1e8), ),
        norm_cfg=None,
        norm_on_bbox=True)
    labels = torch.tensor([0, 1, 1, 1])
    targets = torch.zeros(4, 4)
    targets[0] = 0.5  # stride 8 -> [4, 4, 4, 4] around point [4, 4].
    head._dgfe_last_targets = ([labels], [targets])
    bbox_pred = torch.zeros(1, 4, 2, 2)
    bbox_pred[0, :, 0, 0] = 0.5
    gt = InstanceData(
        bboxes=torch.tensor([[0., 0., 8., 8.]]),
        labels=torch.tensor([0]))

    head._collect_dgfe_records([bbox_pred], [gt])
    records = head.dgfe_assignment_records()
    assert len(records) == 1
    assert records[0] == dict(
        batch_idx=0, level=0, gt_idx=0, quality=1.0)

    head._dgfe_last_targets = None
    head._collect_dgfe_records([bbox_pred], [gt])
    assert head.dgfe_assignment_records() == []


def test_aligned_iou_does_not_rematch_another_gt():
    predictions = torch.tensor([[10., 10., 20., 20.]])
    assigned_gt = torch.tensor([[0., 0., 10., 10.]])
    assert aligned_iou(predictions, assigned_gt).item() == 0.0
