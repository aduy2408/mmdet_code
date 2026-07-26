import unittest

import numpy as np
import torch

from ablate_fcos_p3_centroids import (
    centroid_from_grid,
    decode_candidates,
    maximum_cardinality_match,
    normalized_patch_distribution,
)


class CentroidAblationTest(unittest.TestCase):
    def test_decoder_never_applies_stride_twice(self):
        class Recorder:
            def __init__(self):
                self.seen = None

            def decode(self, priors, bbox_pred, max_shape):
                self.seen = bbox_pred.clone()
                return bbox_pred

        for norm_on_bbox in (False, True):
            coder = Recorder()
            head = type(
                "Head",
                (),
                {"bbox_coder": coder, "norm_on_bbox": norm_on_bbox},
            )()
            distances = torch.tensor([[8.0, 16.0, 24.0, 32.0]])
            decoded = decode_candidates(
                head, torch.tensor([[4.0, 4.0]]), distances, (64, 64)
            )
            self.assertTrue(torch.equal(decoded, distances))
            self.assertTrue(torch.equal(coder.seen, distances))

    def test_zero_displacement_preserves_half_offset_prior(self):
        source = torch.ones(9, 9)
        grid = np.array([36.0, 44.0])  # stride-8 prior at index (4, 5)
        centroid, entropy, boundary, fraction = centroid_from_grid(
            source, (4, 5), grid, 8, 5, 0.25
        )
        np.testing.assert_allclose(centroid, grid)
        self.assertAlmostEqual(entropy, 1, places=6)
        self.assertFalse(boundary)
        self.assertEqual(fraction, 1)

    def test_boundary_patch_and_robust_normalization(self):
        source = torch.arange(25, dtype=torch.float32).reshape(5, 5)
        offsets, probabilities, entropy, boundary, fraction = (
            normalized_patch_distribution(source, (0, 0), 5, 0.25)
        )
        self.assertEqual(offsets.shape, (9, 2))
        self.assertAlmostEqual(float(probabilities.sum()), 1)
        self.assertTrue(boundary)
        self.assertEqual(fraction, 9 / 25)
        self.assertGreaterEqual(entropy, 0)
        self.assertLessEqual(entropy, 1)

    def test_class_aware_maximum_cardinality_matching(self):
        candidates = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [20, 0, 30, 10]])
        candidate_classes = np.array([0, 1, 0])
        gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]])
        gt_classes = np.array([0, 0])
        pairs = maximum_cardinality_match(
            candidates, candidate_classes, gt, gt_classes, 0.5
        )
        self.assertEqual({(candidate, target) for candidate, target, _ in pairs}, {(0, 0), (2, 1)})


if __name__ == "__main__":
    unittest.main()
