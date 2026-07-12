from __future__ import annotations

import unittest

import torch

from models.decoders import ThreeDAwareDPTDecoder
from models.encoders import GalileoHFEncoder


class GalileoTemporalFeatureTest(unittest.TestCase):
    def test_preserves_time_and_averages_only_band_groups(self) -> None:
        batch, height, width, timesteps, groups, channels = 2, 2, 3, 4, 2, 5
        grouped = torch.arange(
            batch * height * width * timesteps * groups * channels,
            dtype=torch.float32,
        ).reshape(batch, height, width, timesteps, groups, channels)
        hidden = grouped.reshape(batch, -1, channels)

        temporal = GalileoHFEncoder._hidden_to_temporal_feature_grid(
            hidden,
            grid_h=height,
            grid_w=width,
            timesteps=timesteps,
        )

        expected = grouped.mean(dim=4).permute(0, 3, 4, 1, 2)
        self.assertEqual(tuple(temporal.shape), (batch, timesteps, channels, height, width))
        self.assertTrue(torch.equal(temporal, expected))


class ThreeDAwareDPTDecoderTest(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        model = ThreeDAwareDPTDecoder(
            in_channels=8,
            num_classes=5,
            decoder_channels=16,
            num_heads=4,
            spatial_window=4,
            global_3d_blocks=1,
            fusion_blocks_per_stage=1,
            mlp_expansion=2,
            dropout=0.0,
            drop_path=0.0,
            temporal_pool_heads=4,
        )
        features = tuple(torch.randn(1, 3, 8, 4, 4) for _ in range(4))
        months = torch.tensor([[9, 10, 11]])

        logits = model(features, months=months, target_size=(16, 16))

        self.assertEqual(tuple(logits.shape), (1, 5, 16, 16))
        self.assertTrue(torch.isfinite(logits).all())
        logits.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_requires_four_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "four Galileo hidden layers"):
            ThreeDAwareDPTDecoder(
                in_channels=8,
                num_classes=5,
                num_layers=3,
                decoder_channels=16,
                num_heads=4,
                temporal_pool_heads=4,
            )


if __name__ == "__main__":
    unittest.main()
