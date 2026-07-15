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
from utils import apply_phenology_overlay, feature_cache_dir, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_PRIOR_PATH = PROJECT_ROOT / "data" / "priors" / "pastis_ext_prior_draft.csv"


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
    def test_phenology_overlay_keeps_decoder_config_and_namespaces_outputs(self) -> None:
        base = load_config(PROJECT_ROOT / "configs" / "galileo_upernet_temporal_readout.yaml")
        combined = apply_phenology_overlay(
            base,
            PROJECT_ROOT / "configs" / "phenology" / "external.yaml",
        )

        self.assertEqual(combined["model"]["decoder"], "temporal_readout_upernet")
        self.assertTrue(combined["phenology"]["enabled"])
        self.assertEqual(
            combined["phenology"]["path"],
            "data/priors/pastis_ext_prior_draft.csv",
        )
        self.assertTrue(combined["train"]["log_dir"].endswith("_phenology_external"))
        self.assertTrue(
            combined["train"]["checkpoint_dir"].endswith("_phenology_external")
        )

    def test_one_overlay_composes_with_every_temporal_decoder_route(self) -> None:
        paths = (
            "galileo_single_layer_dpt_temporal_readout.yaml",
            "galileo_multi_layer_dpt_temporal_readout.yaml",
            "galileo_upernet_temporal_readout.yaml",
            "galileo_adapted_dpt_temporal_readout.yaml",
            "galileo_3d_aware_dpt_late_fusion.yaml",
        )
        overlay = PROJECT_ROOT / "configs" / "phenology" / "external.yaml"

        for name in paths:
            with self.subTest(config=name):
                config = apply_phenology_overlay(
                    load_config(PROJECT_ROOT / "configs" / name),
                    overlay,
                )
                self.assertTrue(cached_decoder_uses_temporal_features(config))
                self.assertTrue(config["phenology"]["enabled"])
                self.assertTrue(
                    config["train"]["log_dir"].endswith("_phenology_external")
                )

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
                    "data": {"num_classes": 19},
                    "encoder": {"hidden_layers": [3, 6, 9, 12]},
                    "model": model_cfg,
                    "phenology": {
                        "enabled": True,
                        "path": str(EXTERNAL_PRIOR_PATH),
                        "hidden_dim": 8,
                        "strength": 0.1,
                    },
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
                self.assertEqual(tuple(logits.shape), (2, 19, 16, 16))
                self.assertTrue(torch.isfinite(logits).all())
                logits.mean().backward()
                self.assertTrue(
                    any(parameter.grad is not None for parameter in model.parameters())
                )
                self.assertIsNotNone(model.temporal_phenology_prior)
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        for parameter in model.temporal_phenology_prior.parameters()
                    )
                )
                self.assertTrue(cached_decoder_uses_temporal_features(config))
                self.assertFalse(cached_decoder_uses_feature_pyramid(config))


if __name__ == "__main__":
    unittest.main()
