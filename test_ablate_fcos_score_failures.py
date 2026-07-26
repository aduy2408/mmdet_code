import unittest

import numpy as np

from ablate_fcos_score_failures import (
    cluster_fraction_ci,
    direct_suppressor,
    select_with_nms,
)


class ScoreFailureAblationTest(unittest.TestCase):
    def test_direct_suppressor(self):
        candidates = {
            "boxes": np.array(
                [[0, 0, 10, 10], [1, 0, 11, 10], [20, 20, 30, 30]],
                dtype=float,
            ),
            "class_id": np.array([0, 0, 0]),
            "final_score": np.array([0.5, 0.9, 0.8]),
        }
        self.assertEqual(
            direct_suppressor(0, np.array([1, 2]), candidates, 0.5), 1
        )
        self.assertEqual(
            direct_suppressor(2, np.array([1]), candidates, 0.5), -1
        )

    def test_oracle_selection_universe(self):
        candidates = {
            "boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float),
            "class_id": np.array([0, 0]),
        }
        selected = select_with_nms(
            candidates,
            np.array([1]),
            np.array([0.9]),
            {"type": "nms", "iou_threshold": 0.5},
            100,
        )
        self.assertEqual(selected.tolist(), [1])

    def test_cluster_bootstrap(self):
        rows = [
            {"image_id": 1, "flag": True},
            {"image_id": 1, "flag": True},
            {"image_id": 2, "flag": False},
        ]
        ci = cluster_fraction_ci(rows, lambda row: row["flag"], 100, 42)
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], ci[1])


if __name__ == "__main__":
    unittest.main()
