from __future__ import annotations

import unittest

import torch

from models import build_cached_feature_model
from models.decoders import UPerNetDecoder


class UPerNetDecoderTest(unittest.TestCase):
    def test_cached_model_forward_and_backward(self) -> None:
        config = {
            "data": {"num_classes": 5},
            "encoder": {"hidden_layers": [3, 6, 9, 12]},
            "model": {
                "decoder": "upernet",
                "decoder_channels": 16,
                "ppm_channels": 4,
                "ppm_scales": [1, 2, 3, 6],
                "dropout": 0.0,
            },
        }
        model = build_cached_feature_model(config, in_channels=8, num_layers=4)
        batch = {
            "features": torch.randn(1, 8, 8, 8),
            "features_by_layer": torch.randn(1, 4, 8, 8, 8),
            "target": torch.zeros(1, 32, 32, dtype=torch.long),
        }

        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (1, 5, 32, 32))
        self.assertTrue(torch.isfinite(logits).all())
        logits.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_requires_multiple_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            UPerNetDecoder(
                in_channels=8,
                num_classes=5,
                num_layers=1,
                decoder_channels=16,
                ppm_channels=4,
            )


if __name__ == "__main__":
    unittest.main()
