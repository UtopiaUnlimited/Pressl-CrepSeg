from __future__ import annotations

import unittest

import torch

from models import build_cached_feature_model
from models.decoders import ThreeDAwareDPTDecoder
from models.encoders import GalileoHFEncoder
from models.phenology import PhenologyPriorAdapter, load_phenology_prior


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
    def test_external_prior_table_and_zero_strength_fallback(self) -> None:
        table = load_phenology_prior(
            "data/priors/pastis_ext_prior_draft.csv",
            num_classes=19,
        )
        self.assertEqual(tuple(table.shape), (19, 12))
        self.assertTrue(torch.isfinite(table).all())

        adapter = PhenologyPriorAdapter(
            prior_table=table,
            decoder_channels=16,
            hidden_dim=8,
            strength=0.0,
        )
        context = adapter(torch.tensor([[9, 10, 11]]))
        self.assertEqual(tuple(context.shape), (1, 3, 16))
        self.assertTrue(torch.equal(context, torch.zeros_like(context)))

    def test_temporal_prior_is_injected_before_decoder_fusion(self) -> None:
        prior = PhenologyPriorAdapter(
            prior_table=torch.rand(5, 12),
            decoder_channels=16,
            hidden_dim=8,
            strength=0.1,
        )
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
            phenology_prior=prior,
        )
        features = tuple(torch.randn(1, 3, 8, 4, 4) for _ in range(4))
        logits = model(features, months=torch.tensor([[9, 10, 11]]), target_size=(16, 16))
        self.assertEqual(tuple(logits.shape), (1, 5, 16, 16))
        self.assertTrue(torch.isfinite(logits).all())
        logits.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in prior.parameters()))

    def test_cached_wrapper_forward_and_backward(self) -> None:
        config = {
            "data": {"num_classes": 5},
            "encoder": {"hidden_layers": [3, 6, 9, 12]},
            "model": {
                "decoder": "3d_aware_dpt",
                "decoder_channels": 16,
                "num_heads": 4,
                "spatial_window": 4,
                "global_3d_blocks": 1,
                "fusion_blocks_per_stage": 1,
                "mlp_expansion": 2,
                "dropout": 0.0,
                "drop_path": 0.0,
                "temporal_pool_heads": 4,
                "num_months": 12,
            },
        }
        model = build_cached_feature_model(config, in_channels=8, num_layers=4)
        batch = {
            "temporal_features_by_layer": torch.randn(1, 4, 3, 8, 4, 4),
            "months": torch.tensor([[9, 10, 11]]),
            "target": torch.zeros(1, 16, 16, dtype=torch.long),
        }

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (1, 5, 16, 16))
        logits.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

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
        deepest_refine_shapes = []
        hook = model.reassemble[-1].refine.register_forward_hook(
            lambda _module, _inputs, output: deepest_refine_shapes.append(
                tuple(output.shape)
            )
        )

        logits = model(features, months=months, target_size=(16, 16))
        hook.remove()

        self.assertEqual(tuple(logits.shape), (1, 5, 16, 16))
        self.assertEqual(
            deepest_refine_shapes,
            [(1, 16, 3, 2, 2), (1, 16, 3, 4, 4)],
        )
        self.assertTrue(torch.isfinite(logits).all())
        logits.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_native_skip_keeps_legacy_state_dict_keys(self) -> None:
        kwargs = {
            "in_channels": 8,
            "num_classes": 5,
            "decoder_channels": 16,
            "num_heads": 4,
            "global_3d_blocks": 1,
            "fusion_blocks_per_stage": 1,
            "temporal_pool_heads": 4,
        }
        legacy = ThreeDAwareDPTDecoder(
            **kwargs,
            preserve_native_deep_skip=False,
        )
        current = ThreeDAwareDPTDecoder(
            **kwargs,
            preserve_native_deep_skip=True,
        )

        self.assertEqual(set(legacy.state_dict()), set(current.state_dict()))
        current.load_state_dict(legacy.state_dict(), strict=True)

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
