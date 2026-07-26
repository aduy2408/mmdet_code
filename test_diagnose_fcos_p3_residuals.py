import unittest

import numpy as np
import torch

from diagnose_fcos_p3_residuals import (
    grouped_leave_one_out,
    pca_reconstruct,
    residual_maps,
    ring_predict,
    softmax_centroid,
)


class FixedResidualTest(unittest.TestCase):
    def test_fixed_residual_operators_and_centroid(self):
        feature = torch.arange(2 * 4 * 7 * 7, dtype=torch.float32).reshape(
            2, 4, 7, 7
        )
        constant = torch.ones(1, 4, 7, 7)
        self.assertTrue(
            torch.allclose(
                ring_predict(constant, 5)[:, :, 2:-2, 2:-2],
                constant[:, :, 2:-2, 2:-2],
            )
        )

        grouped = grouped_leave_one_out(feature, 2)
        self.assertTrue(torch.equal(grouped[:, 0], feature[:, 1]))
        self.assertTrue(torch.equal(grouped[:, 2], feature[:, 3]))

        mean = torch.zeros(4)
        basis = torch.eye(4)[:, :2]
        reconstructed = pca_reconstruct(feature, mean, basis)
        self.assertEqual(reconstructed.shape, feature.shape)
        self.assertTrue(torch.equal(reconstructed[:, :2], feature[:, :2]))
        self.assertFalse(bool(reconstructed[:, 2:].any()))

        maps = residual_maps(
            feature, ring_predict(feature, 3), grouped, (0, 1), (0, 1)
        )
        self.assertGreaterEqual(
            set(maps), {"spatial", "channel", "agreement", "hardness"}
        )
        self.assertTrue(bool(torch.isfinite(maps["hardness"]).all()))
        self.assertGreaterEqual(float(maps["agreement"].min()), 0)
        self.assertLessEqual(float(maps["agreement"].max()), 1)

        heat = torch.zeros(7, 7)
        heat[3, 4] = 10
        peak, centroid = softmax_centroid(heat, (3, 3), 5, 0.1)
        self.assertTrue(np.array_equal(peak, [4, 3]))
        self.assertLess(np.linalg.norm(centroid - [4, 3]), 1e-3)


if __name__ == "__main__":
    unittest.main()
