import unittest

import numpy as np
import torch

from decompose_fcos_tiny_errors import (
    ErrorDecomposition,
    ORACLE_GAIN,
    geometry_category,
    instrumented_assignment,
    one_iou,
    oracle_boxes,
)


class ErrorDecompositionTest(unittest.TestCase):
    def test_empty_candidate_stages(self):
        diagnostic = object.__new__(ErrorDecomposition)
        diagnostic.device = torch.device("cpu")
        diagnostic.args = type("Args", (), {"score_threshold": 0.05})()
        diagnostic.head = type(
            "Head",
            (),
            {
                "test_cfg": type(
                    "Cfg",
                    (),
                    {
                        "nms": {"type": "nms", "iou_threshold": 0.5},
                        "max_per_img": 100,
                    },
                )()
            },
        )()
        candidates = {
            "candidate_id": np.array([1]),
            "cls_score": np.array([0.01]),
            "boxes": np.array([[0.0, 0.0, 1.0, 1.0]]),
            "final_score": np.array([0.005]),
            "class_id": np.array([0]),
        }
        filtered, official, stages = diagnostic.stage_populations(candidates, set())
        self.assertEqual(len(filtered), 0)
        self.assertEqual(len(official), 0)
        self.assertEqual(stages[1], "classification_threshold")

    def test_oracle_geometry(self):
        target = np.array([0.0, 0.0, 10.0, 10.0])
        predicted = np.array([-1.0, 2.0, 5.0, 10.0])
        boxes = oracle_boxes(predicted, target)
        self.assertGreater(one_iou(boxes["center"], target), one_iou(predicted, target))
        self.assertGreater(one_iou(boxes["extent"], target), one_iou(predicted, target))
        self.assertAlmostEqual(one_iou(boxes["full"], target), 1)

    def test_category_order(self):
        self.assertEqual(geometry_category(0.5, 0, 0), "success")
        self.assertEqual(geometry_category(0.3, ORACLE_GAIN, 0), "center_only")
        self.assertEqual(geometry_category(0.3, 0, ORACLE_GAIN), "extent_only")
        self.assertEqual(
            geometry_category(0.3, ORACLE_GAIN, ORACLE_GAIN),
            "center_and_extent",
        )
        self.assertEqual(geometry_category(0.3, 0, 0), "geometry_other")

    def test_instrumented_assignment_simple(self):
        head = type(
            "Head",
            (),
            {
                "regress_ranges": ((-1, 64),),
                "num_classes": 1,
                "norm_on_bbox": False,
                "strides": (8,),
            },
        )()
        points = [torch.tensor([[4.0, 4.0], [12.0, 4.0], [20.0, 4.0]])]
        boxes = torch.tensor([[0.0, 0.0, 16.0, 8.0]])
        labels = torch.tensor([0])
        assigned, targets, indices = instrumented_assignment(
            head, points, boxes, labels
        )
        self.assertEqual(indices[0].tolist(), [0, 0, -1])
        self.assertEqual(assigned[0].tolist(), [0, 0, 1])
        self.assertEqual(targets[0].shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
