from __future__ import annotations

import unittest
from pathlib import Path

import torch

from models import (
    build_cached_feature_model,
    cached_decoder_uses_feature_pyramid,
    cached_decoder_uses_temporal_features,
)
from models.decoders import MonthAwareTemporalReadout, TemporalReadoutDecoder
from utils import feature_cache_dir, load_config


class MonthAwareTemporalReadoutTest(unittest.TestCase):
    def test_zero_initialized_scores_start_as_an_exact_time_mean(self) -> None:
        readout = MonthAwareTemporalReadout(
            channels=8,
            num_layers=4,
            num_months=12,
            hidden_channels=4,
            dropout=0.0,
        )
        features = torch.randn(2, 3, 8, 4, 4)
        months = torch.tensor([[9, 10, 11], [9, 10, 11]])

        fused = readout(features, months, layer_index=3)

        torch.testing.assert_close(fused, features.mean(dim=1), rtol=1e-6, atol=1e-7)

    def test_temporal_scorer_receives_gradients(self) -> None:
        readout = MonthAwareTemporalReadout(
            channels=8,
            num_layers=4,
            hidden_channels=4,
        )
        features = torch.randn(2, 3, 8, 4, 4)
        months = torch.tensor([[9, 10, 11], [9, 10, 11]])

        readout(features, months, layer_index=2).square().mean().backward()

        final_weight = readout.scorer[-1].weight
        self.assertIsNotNone(final_weight.grad)
        self.assertGreater(float(final_weight.grad.abs().sum()), 0.0)


class TemporalReadoutDecoderTest(unittest.TestCase):
    def test_all_four_configs_use_the_same_temporal_cache_protocol(self) -> None:
        paths = sorted(Path("configs").glob("*_temporal_readout.yaml"))
        self.assertEqual(len(paths), 4)

        configs = [load_config(path) for path in paths]
        self.assertTrue(all(cached_decoder_uses_temporal_features(config) for config in configs))
        self.assertTrue(
            all(config["cache"]["format"] == "temporal_v2" for config in configs)
        )
        self.assertEqual(
            len({feature_cache_dir(config, "train") for config in configs}),
            1,
        )

    def test_all_four_spatial_decoders_accept_temporal_cache_features(self) -> None:
        decoder_configs = {
            "temporal_readout_single_layer_dpt": {
                "decoder_channels": 16,
                "decoder_blocks": 1,
            },
            "temporal_readout_multi_layer_dpt": {
                "decoder_channels": 16,
                "decoder_blocks": 1,
                "fusion_blocks": 1,
            },
            "temporal_readout_upernet": {
                "decoder_channels": 16,
                "ppm_channels": 4,
                "ppm_scales": [1, 2, 3, 6],
            },
            "temporal_readout_galileo_dpt": {
                "decoder_channels": 16,
                "fusion_blocks": 1,
                "head_channels": 8,
                "preserve_native_deep_skip": True,
            },
        }

        for decoder_name, decoder_options in decoder_configs.items():
            with self.subTest(decoder=decoder_name):
                model_cfg = {
                    "decoder": decoder_name,
                    "dropout": 0.0,
                    "temporal_readout": {
                        "num_months": 12,
                        "hidden_channels": 4,
                        "dropout": 0.0,
                    },
                    **decoder_options,
                }
                config = {
                    "data": {"num_classes": 5},
                    "encoder": {"hidden_layers": [3, 6, 9, 12]},
                    "model": model_cfg,
                }
                model = build_cached_feature_model(
                    config,
                    in_channels=8,
                    num_layers=4,
                )
                batch = {
                    "temporal_features_by_layer": torch.randn(
                        2,
                        4,
                        3,
                        8,
                        4,
                        4,
                    ),
                    "months": torch.tensor([[9, 10, 11], [9, 10, 11]]),
                    "target": torch.zeros(2, 16, 16, dtype=torch.long),
                }

                logits = model(batch)

                self.assertIsInstance(model.decoder, TemporalReadoutDecoder)
                self.assertEqual(tuple(logits.shape), (2, 5, 16, 16))
                self.assertTrue(torch.isfinite(logits).all())
                logits.mean().backward()
                self.assertTrue(
                    any(parameter.grad is not None for parameter in model.parameters())
                )
                self.assertTrue(cached_decoder_uses_temporal_features(config))
                self.assertFalse(cached_decoder_uses_feature_pyramid(config))


if __name__ == "__main__":
    unittest.main()
