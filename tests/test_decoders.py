from __future__ import annotations

import unittest

import torch

from models import build_cached_feature_model
from models.decoders import GalileoDPTDecoder, GalileoLinearProbeDecoder, UPerNetDecoder


class GalileoDPTDecoderTest(unittest.TestCase):
    def test_reassembles_four_galileo_layers_and_backpropagates(self) -> None:
        config = {
            "data": {"num_classes": 5},
            "encoder": {"hidden_layers": [3, 6, 9, 12]},
            "model": {
                "decoder": "galileo_dpt",
                "decoder_channels": 16,
                "fusion_blocks": 1,
                "head_channels": 8,
                "dropout": 0.0,
            },
        }
        model = build_cached_feature_model(config, in_channels=8, num_layers=4)
        batch = {
            "features": torch.randn(2, 8, 8, 8),
            "features_by_layer": torch.randn(2, 4, 8, 8, 8),
            "target": torch.zeros(2, 32, 32, dtype=torch.long),
        }
        pyramid_shapes = []
        hooks = [
            adapter.register_forward_hook(
                lambda _module, _inputs, output: pyramid_shapes.append(tuple(output.shape))
            )
            for adapter in model.decoder.reassemble
        ]

        logits = model(batch)
        for hook in hooks:
            hook.remove()

        self.assertIsInstance(model.decoder, GalileoDPTDecoder)
        self.assertEqual(
            pyramid_shapes,
            [
                (2, 16, 32, 32),
                (2, 16, 16, 16),
                (2, 16, 8, 8),
                (2, 16, 4, 4),
            ],
        )
        self.assertEqual(tuple(logits.shape), (2, 5, 32, 32))
        self.assertTrue(torch.isfinite(logits).all())
        logits.mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_requires_exactly_four_ordered_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires exactly four"):
            GalileoDPTDecoder(
                in_channels=8,
                num_classes=5,
                num_layers=3,
                decoder_channels=16,
                fusion_blocks=1,
                head_channels=8,
            )

    def test_requires_at_least_one_fusion_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "fusion_blocks must be at least one"):
            GalileoDPTDecoder(
                in_channels=8,
                num_classes=5,
                num_layers=4,
                decoder_channels=16,
                fusion_blocks=0,
                head_channels=8,
            )


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


class GalileoLinearProbeTest(unittest.TestCase):
    def test_patch_tokens_map_directly_to_pixel_logits(self) -> None:
        config = {
            "data": {"num_classes": 5},
            "encoder": {"hidden_layers": [3, 6, 9, 12]},
            "model": {"decoder": "linear_probe", "output_patch_size": 4},
        }
        model = build_cached_feature_model(config, in_channels=8)
        batch = {
            "features": torch.randn(2, 8, 8, 8),
            "target": torch.zeros(2, 32, 32, dtype=torch.long),
        }

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (2, 5, 32, 32))
        self.assertIsInstance(model.decoder, GalileoLinearProbeDecoder)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            8 * (5 * 4 * 4) + (5 * 4 * 4),
        )
        logits.mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
