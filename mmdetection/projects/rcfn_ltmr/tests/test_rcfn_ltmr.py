import torch
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import FCOSHead
from projects.rcfn_ltmr import LTMRFCOSHead, RCFNFPN


def _head():
    return LTMRFCOSHead(
        num_classes=2,
        in_channels=1,
        feat_channels=1,
        stacked_convs=1,
        norm_cfg=None,
        strides=(8, 16, 32, 64, 128),
        regress_ranges=((-1, 64), (64, 128), (128, 256),
                        (256, 512), (512, 1e8)))


def _instances(boxes, labels=None):
    result = InstanceData()
    result.bboxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if labels is not None:
        result.labels = torch.tensor(labels, dtype=torch.long)
    return result


def test_ring_stats_and_identity_initialization():
    neck = RCFNFPN(
        in_channels=[1, 2, 4, 8],
        out_channels=4,
        num_outs=5,
        start_level=1,
        add_extra_convs='on_output')
    flat = torch.ones(1, 4, 5, 5, requires_grad=True)
    mean, var = neck.ring_stats(flat)
    assert torch.equal(mean, flat)
    assert torch.all(var == neck.eps)

    inputs = tuple(
        torch.randn(1, channels, size, size)
        for channels, size in zip((1, 2, 4, 8), (32, 16, 8, 4)))
    baseline = super(RCFNFPN, neck).forward(inputs)
    output = neck(inputs)
    assert torch.equal(output[0], baseline[0])
    assert all(torch.equal(a, b) for a, b in zip(output[1:], baseline[1:]))
    output[0].sum().backward()
    assert torch.isfinite(neck.gamma.grad).all()


def test_assignment_background_overlap_and_empty_gt():
    head = _head()
    sizes = [(4, 4), (2, 2), (1, 1), (1, 1), (1, 1)]
    points = head.prior_generator.grid_priors(sizes, device='cpu')
    counts = [len(item) for item in points]
    all_points = torch.cat(points)
    ranges = torch.cat([
        point.new_tensor(head.regress_ranges[level])[None].expand_as(point)
        for level, point in enumerate(points)
    ])

    empty = _instances([], [])
    assert (head.assigned_gt_inds(
        all_points, empty, ranges, counts) == -1).all()

    overlapping = _instances([[0, 0, 32, 32], [8, 8, 24, 24]], [0, 1])
    assigned = head.assigned_gt_inds(
        all_points, overlapping, ranges, counts)
    center = ((all_points[:, 0] == 12) & (all_points[:, 1] == 12)).nonzero()
    assert assigned[center].item() == 1
    assert (assigned >= 0).any() and (assigned < 0).any()


def test_ltmr_filters_other_gt_ignore_and_padding():
    head = _head()
    points = head.prior_generator.grid_priors(
        [(5, 5), (1, 1), (1, 1), (1, 1), (1, 1)],
        device='cpu')[0]
    logits = torch.zeros(1, 2, 5, 5, requires_grad=True)
    gt = _instances([[12, 12, 20, 20], [28, 12, 36, 20]], [0, 0])
    assignments = [torch.full((25,), -1, dtype=torch.long)]
    assignments[0][12] = 0
    assignments[0][14] = 1
    ignored = [_instances([[0, 0, 8, 40]])]
    loss = head.local_tiny_margin_loss(
        logits, points, assignments, [gt],
        [{'img_shape': (32, 32, 3)}], ignored)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_ltmr_zero_when_no_valid_tiny_pair():
    head = _head()
    points = head.prior_generator.grid_priors(
        [(2, 2), (1, 1), (1, 1), (1, 1), (1, 1)],
        device='cpu')[0]
    logits = torch.randn(1, 2, 2, 2, requires_grad=True)
    large_gt = _instances([[0, 0, 64, 64]], [0])
    loss = head.local_tiny_margin_loss(
        logits, points, [torch.zeros(4, dtype=torch.long)], [large_gt],
        [{'img_shape': (16, 16, 3)}])
    assert loss.item() == 0
    loss.backward()
    assert logits.grad is not None


def test_ltmr_inference_matches_fcos():
    kwargs = dict(
        num_classes=2,
        in_channels=1,
        feat_channels=1,
        stacked_convs=1,
        norm_cfg=None)
    baseline = FCOSHead(**kwargs).eval()
    ltmr = LTMRFCOSHead(**kwargs).eval()
    ltmr.load_state_dict(baseline.state_dict(), strict=False)
    features = tuple(
        torch.randn(1, 1, 64 // stride[0], 64 // stride[1])
        for stride in baseline.prior_generator.strides)
    with torch.inference_mode():
        expected = baseline(features)
        actual = ltmr(features)
    for expected_group, actual_group in zip(expected, actual):
        for expected_tensor, actual_tensor in zip(
                expected_group, actual_group):
            assert torch.equal(expected_tensor, actual_tensor)
