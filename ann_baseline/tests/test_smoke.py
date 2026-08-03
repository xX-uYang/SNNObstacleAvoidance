from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from ann_baseline.interfaces import SNNPrior
from ann_baseline.model import ConditionalANN


class SNNPriorTests(unittest.TestCase):
    def test_probability_normalization(self) -> None:
        prior = SNNPrior.from_probabilities(8.0, 1.0, 1.0)
        self.assertAlmostEqual(prior.left_probability, 0.8)
        self.assertAlmostEqual(prior.straight_probability, 0.1)
        self.assertAlmostEqual(prior.right_probability, 0.1)
        self.assertTrue(prior.available)

    def test_label_to_one_hot(self) -> None:
        self.assertEqual(
            SNNPrior.from_label("left").as_model_input(),
            (1.0, 0.0, 0.0, 1.0),
        )


class ModelTests(unittest.TestCase):
    def test_image_only_shape(self) -> None:
        model = ConditionalANN(pretrained=False, use_snn_prior=False)
        output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 3))

    def test_fusion_shape(self) -> None:
        model = ConditionalANN(pretrained=False, use_snn_prior=True)
        output = model(
            torch.zeros(2, 3, 64, 64),
            torch.tensor(
                [[1 / 3, 1 / 3, 1 / 3, 0.0], [0.8, 0.1, 0.1, 1.0]]
            ),
        )
        self.assertEqual(tuple(output.shape), (2, 3))

    def test_steering_is_bounded(self) -> None:
        model = ConditionalANN(
            task="steering",
            pretrained=False,
            use_snn_prior=True,
        )
        output = model(
            torch.zeros(2, 3, 64, 64),
            torch.tensor(
                [[1 / 3, 1 / 3, 1 / 3, 0.0], [0.8, 0.1, 0.1, 1.0]]
            ),
        )
        self.assertTrue(bool(torch.all(output <= 1.0)))
        self.assertTrue(bool(torch.all(output >= -1.0)))


if __name__ == "__main__":
    unittest.main()
