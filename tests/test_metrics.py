from __future__ import annotations

import unittest

import torch

from metrics import macro_f1, mean_iou, pixel_accuracy


class SegmentationMetricsTest(unittest.TestCase):
    def test_accuracy_and_macro_f1_from_confusion_matrix(self) -> None:
        confusion = torch.tensor(
            [
                [3, 1],
                [2, 4],
            ],
            dtype=torch.long,
        )

        accuracy = pixel_accuracy(confusion)
        f1, per_class_f1 = macro_f1(confusion)

        self.assertAlmostEqual(accuracy, 0.7)
        self.assertTrue(
            torch.allclose(per_class_f1, torch.tensor([2 * 3 / 9, 2 * 4 / 11]))
        )
        self.assertAlmostEqual(f1, ((2 * 3 / 9) + (2 * 4 / 11)) / 2)

    def test_metrics_ignore_classes_absent_from_target_and_prediction(self) -> None:
        confusion = torch.tensor(
            [
                [5, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=torch.long,
        )

        miou, per_class_iou = mean_iou(confusion)
        f1, per_class_f1 = macro_f1(confusion)

        self.assertAlmostEqual(miou, 1.0, places=6)
        self.assertAlmostEqual(f1, 1.0, places=6)
        self.assertTrue(
            torch.allclose(
                per_class_iou,
                torch.tensor([1.0, 0.0, 0.0]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                per_class_f1,
                torch.tensor([1.0, 0.0, 0.0]),
                atol=1e-6,
            )
        )

    def test_empty_confusion_matrix_returns_zero_metrics(self) -> None:
        confusion = torch.zeros((2, 2), dtype=torch.long)

        f1, _ = macro_f1(confusion)

        self.assertEqual(pixel_accuracy(confusion), 0.0)
        self.assertEqual(f1, 0.0)


if __name__ == "__main__":
    unittest.main()
