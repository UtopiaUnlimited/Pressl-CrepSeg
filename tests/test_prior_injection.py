from __future__ import annotations

import unittest
from pathlib import Path

import torch

from models import (
    ContentAwarePriorFusion,
    PhenologyPriorTokenEncoder,
    PriorBatch,
    SourceAwareSpatialFiLMFusion,
    TemporalFeaturePyramidPriorInjection,
    build_cached_feature_model,
    cached_decoder_uses_temporal_features,
)
from models.phenology import build_phenology_token_encoder
from utils import apply_prior_injection_overlay, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_PRIOR_PATH = PROJECT_ROOT / "data" / "priors" / "pastis_ext_prior_v1.csv"
CA_HPI_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "prior_injection" / "ca_hpi_structured.yaml"
)
SA_SPATIAL_FILM_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "prior_injection" / "sa_spatial_film_m1_m2_m3_m4.yaml"
)


def small_prior_config(decoder_name: str) -> dict:
    model_options: dict = {
        "decoder": decoder_name,
        "decoder_channels": 16,
        "dropout": 0.0,
    }
    if decoder_name == "3d_aware_dpt":
        model_options.update(
            {
                "num_heads": 4,
                "spatial_window": 4,
                "global_3d_blocks": 1,
                "fusion_blocks_per_stage": 1,
                "mlp_expansion": 2,
                "drop_path": 0.0,
                "temporal_pool_heads": 4,
                "num_months": 12,
            }
        )
    elif decoder_name == "temporal_readout_single_layer_dpt":
        model_options.update(
            {
                "decoder_blocks": 1,
                "temporal_readout": {
                    "num_months": 12,
                    "hidden_channels": 8,
                    "dropout": 0.0,
                },
            }
        )
    else:
        raise ValueError(decoder_name)

    return {
        "data": {"num_classes": 19},
        "encoder": {"hidden_layers": [3, 6, 9, 12]},
        "model": model_options,
        "prior_injection": {
            "enabled": True,
            "method": "ca_hpi",
            "token_dim": 16,
            "source": {
                "kind": "phenology_table",
                "path": str(EXTERNAL_PRIOR_PATH),
                "default_confidence": 1.0,
            },
            "encoder": {
                "hidden_dim": 16,
                "time_frequencies": 2,
                "dropout": 0.0,
            },
            "fusion": {
                "attention_dim": 16,
                "num_heads": 4,
                "gate_hidden_dim": 8,
                "dropout": 0.0,
                "confidence_bias_scale": 1.0,
                "initial_strength": 0.2,
                "learnable_strength": True,
            },
            "diagnostics": {"enabled": True},
        },
    }


class PriorConfigurationTest(unittest.TestCase):
    def test_overlay_keeps_decoder_and_namespaces_outputs(self) -> None:
        base = load_config(PROJECT_ROOT / "configs" / "galileo_3d_aware_dpt.yaml")
        combined = apply_prior_injection_overlay(base, CA_HPI_CONFIG_PATH)

        self.assertEqual(combined["model"]["decoder"], "3d_aware_dpt")
        self.assertTrue(combined["prior_injection"]["enabled"])
        self.assertEqual(combined["prior_injection"]["method"], "ca_hpi")
        self.assertTrue(
            combined["train"]["log_dir"].endswith("_prior_ca_hpi_structured_v1")
        )
        self.assertTrue(
            combined["train"]["checkpoint_dir"].endswith("_prior_ca_hpi_structured_v1")
        )

    def test_one_overlay_composes_with_every_temporal_decoder_route(self) -> None:
        paths = (
            "galileo_single_layer_dpt_temporal_readout.yaml",
            "galileo_multi_layer_dpt_temporal_readout.yaml",
            "galileo_upernet_temporal_readout.yaml",
            "galileo_adapted_dpt_temporal_readout.yaml",
            "galileo_3d_aware_dpt.yaml",
        )
        for name in paths:
            with self.subTest(config=name):
                combined = apply_prior_injection_overlay(
                    load_config(PROJECT_ROOT / "configs" / name),
                    CA_HPI_CONFIG_PATH,
                )
                self.assertTrue(cached_decoder_uses_temporal_features(combined))
                self.assertTrue(combined["prior_injection"]["enabled"])

    def test_source_aware_spatial_film_overlay_keeps_decoder_front_route(self) -> None:
        base = load_config(PROJECT_ROOT / "configs" / "galileo_3d_aware_dpt.yaml")
        combined = apply_prior_injection_overlay(base, SA_SPATIAL_FILM_CONFIG_PATH)

        self.assertEqual(combined["model"]["decoder"], "3d_aware_dpt")
        self.assertEqual(
            combined["prior_injection"]["method"],
            "source_aware_spatial_film",
        )
        self.assertEqual(
            combined["prior_injection"]["fusion"]["initial_strength"], 0.01
        )
        self.assertTrue(
            combined["train"]["log_dir"].endswith(
                "_prior_sa_spatial_film_m1_m2_m3_m4_v1"
            )
        )


class StructuredPriorTest(unittest.TestCase):
    def test_existing_confidence_labels_reach_prior_batch(self) -> None:
        encoder = build_phenology_token_encoder(small_prior_config("3d_aware_dpt"))

        prior = encoder(batch_size=1)
        confidence_by_class = prior.confidence.reshape(1, 19, 12)[0, :, 0]

        self.assertEqual(float(confidence_by_class[0]), 1.0)
        self.assertAlmostEqual(float(confidence_by_class[1]), 0.4, places=6)
        self.assertAlmostEqual(float(confidence_by_class[3]), 0.7, places=6)

    def test_phenology_adapter_builds_complete_entity_month_library(self) -> None:
        table = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 0.8, 0.7],
            ],
            dtype=torch.float32,
        )
        encoder = PhenologyPriorTokenEncoder(
            table,
            token_dim=8,
            hidden_dim=8,
            time_frequencies=2,
            dropout=0.0,
        )

        prior = encoder(batch_size=2)

        self.assertEqual(tuple(prior.tokens.shape), (2, 12, 8))
        self.assertEqual(tuple(prior.mask.shape), (2, 12))
        self.assertTrue(prior.mask.all())
        self.assertEqual(tuple(prior.entity_ids.shape), (2, 12))
        self.assertEqual(tuple(prior.time_values.shape), (2, 12, 1))
        self.assertTrue(torch.isfinite(prior.tokens).all())

    def test_mask_and_confidence_control_attention(self) -> None:
        fusion = ContentAwarePriorFusion(
            vision_dim=4,
            prior_dim=4,
            attention_dim=4,
            num_heads=2,
            gate_hidden_dim=4,
            dropout=0.0,
            confidence_bias_scale=1.0,
        )
        prior = PriorBatch(
            tokens=torch.ones(1, 3, 4),
            mask=torch.tensor([[True, True, False]]),
            confidence=torch.tensor([[1.0, 0.25, 1.0]]),
            type_ids=torch.zeros(1, 3, dtype=torch.long),
        )

        diagnostics = fusion(torch.zeros(1, 2, 4), prior)
        attention = diagnostics.attention.mean(dim=(1, 2))

        self.assertTrue(torch.equal(attention[:, 2], torch.zeros(1)))
        self.assertGreater(float(attention[0, 0]), float(attention[0, 1]))

    def test_all_unknown_prior_is_a_zero_residual(self) -> None:
        fusion = ContentAwarePriorFusion(
            vision_dim=4,
            prior_dim=4,
            attention_dim=4,
            num_heads=2,
            gate_hidden_dim=4,
        )
        prior = PriorBatch(
            tokens=torch.randn(1, 3, 4),
            mask=torch.zeros(1, 3, dtype=torch.bool),
            confidence=torch.zeros(1, 3),
            type_ids=torch.zeros(1, 3, dtype=torch.long),
        )

        diagnostics = fusion(torch.randn(1, 2, 4), prior)

        self.assertTrue(
            torch.equal(diagnostics.residual, torch.zeros_like(diagnostics.residual))
        )
        self.assertTrue(torch.isfinite(diagnostics.attention).all())

    def test_source_balance_removes_token_count_prior(self) -> None:
        fusion = ContentAwarePriorFusion(
            vision_dim=4,
            prior_dim=4,
            attention_dim=4,
            num_heads=2,
            gate_hidden_dim=4,
            dropout=0.0,
            confidence_bias_scale=0.0,
            source_balance_bias_scale=1.0,
        )
        for projection in (fusion.query_projection, fusion.key_projection):
            torch.nn.init.zeros_(projection.weight)
            torch.nn.init.zeros_(projection.bias)
        prior = PriorBatch(
            tokens=torch.zeros(1, 4, 4),
            mask=torch.ones(1, 4, dtype=torch.bool),
            confidence=torch.ones(1, 4),
            type_ids=torch.tensor([[0, 0, 0, 1]], dtype=torch.long),
        )

        attention = fusion(torch.zeros(1, 1, 4), prior).attention.mean(dim=(1, 2))

        self.assertAlmostEqual(float(attention[0, :3].sum()), 0.5, places=6)
        self.assertAlmostEqual(float(attention[0, 3]), 0.5, places=6)

    def test_source_aware_spatial_film_uses_explicit_source_weights(self) -> None:
        fusion = SourceAwareSpatialFiLMFusion(
            vision_dim=4,
            prior_dim=4,
            attention_dim=4,
            num_heads=2,
            gate_hidden_dim=4,
            source_gate_hidden_dim=4,
            film_hidden_dim=4,
            dropout=0.0,
            confidence_bias_scale=0.0,
            source_balance_bias_scale=1.0,
        )
        for projection in (fusion.query_projection, fusion.key_projection):
            torch.nn.init.zeros_(projection.weight)
            torch.nn.init.zeros_(projection.bias)
        for parameter in fusion.source_gate.parameters():
            torch.nn.init.zeros_(parameter)
        prior = PriorBatch(
            tokens=torch.randn(1, 4, 4),
            mask=torch.ones(1, 4, dtype=torch.bool),
            confidence=torch.ones(1, 4),
            type_ids=torch.tensor([[0, 0, 0, 1]], dtype=torch.long),
            source_names=("large_source", "small_source"),
        )

        diagnostics = fusion(torch.randn(1, 3, 4), prior)

        self.assertEqual(tuple(diagnostics.source_weights.shape), (1, 3, 2))
        self.assertTrue(
            torch.allclose(
                diagnostics.source_weights.sum(dim=-1),
                torch.ones(1, 3),
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.source_weights[..., 0],
                diagnostics.source_weights[..., 1],
                atol=1e-6,
            )
        )
        self.assertEqual(tuple(diagnostics.channel_scale.shape), (1, 3, 4))
        self.assertEqual(tuple(diagnostics.channel_shift.shape), (1, 3, 4))
        self.assertTrue(torch.isfinite(diagnostics.residual).all())

    def test_source_aware_spatial_film_all_unknown_is_zero(self) -> None:
        fusion = SourceAwareSpatialFiLMFusion(
            vision_dim=4,
            prior_dim=4,
            attention_dim=4,
            num_heads=2,
            gate_hidden_dim=4,
            source_gate_hidden_dim=4,
            film_hidden_dim=4,
            dropout=0.0,
        )
        prior = PriorBatch(
            tokens=torch.randn(2, 3, 4),
            mask=torch.zeros(2, 3, dtype=torch.bool),
            confidence=torch.zeros(2, 3),
            type_ids=torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.long),
            source_names=("source_0", "source_1"),
        )

        diagnostics = fusion(torch.randn(2, 5, 4), prior)

        self.assertTrue(
            torch.equal(diagnostics.residual, torch.zeros_like(diagnostics.residual))
        )
        self.assertTrue(torch.isfinite(diagnostics.source_weights).all())


class PreDecoderPriorInjectionTest(unittest.TestCase):
    def test_recorded_diagnostics_are_scalar_finite_and_consumed_once(self) -> None:
        prior_encoder = PhenologyPriorTokenEncoder(
            torch.rand(3, 4),
            token_dim=8,
            hidden_dim=8,
            time_frequencies=2,
        )
        injector = TemporalFeaturePyramidPriorInjection(
            vision_dim=8,
            prior_dim=8,
            num_layers=2,
            attention_dim=8,
            num_heads=2,
            gate_hidden_dim=8,
            dropout=0.0,
            initial_strength=0.0,
            learnable_strength=True,
            record_diagnostics=True,
        )
        features = tuple(torch.randn(2, 3, 8, 4, 4) for _ in range(2))

        enhanced = injector(features, prior_encoder(batch_size=2))
        recorded = injector.pop_prior_diagnostics()

        self.assertEqual(len(recorded), 2 * 13)
        self.assertEqual(injector.pop_prior_diagnostics(), {})
        for value in recorded.values():
            self.assertEqual(value.numel(), 1)
            self.assertTrue(torch.isfinite(value))
        for layer_index in range(2):
            prefix = f"layer_{layer_index}/"
            self.assertEqual(float(recorded[prefix + "strength"]), 0.0)
            self.assertEqual(float(recorded[prefix + "applied_residual_ratio"]), 0.0)
            self.assertGreaterEqual(float(recorded[prefix + "gate_mean"]), 0.0)
            self.assertLessEqual(float(recorded[prefix + "gate_mean"]), 1.0)
            self.assertGreaterEqual(float(recorded[prefix + "attention_entropy"]), 0.0)
            self.assertLessEqual(float(recorded[prefix + "attention_entropy"]), 1.0)
        for original, output in zip(features, enhanced):
            self.assertTrue(torch.equal(original, output))

    def test_zero_strength_is_an_exact_feature_pyramid_fallback(self) -> None:
        prior_encoder = PhenologyPriorTokenEncoder(
            torch.rand(3, 4),
            token_dim=8,
            hidden_dim=8,
            time_frequencies=2,
        )
        injector = TemporalFeaturePyramidPriorInjection(
            vision_dim=8,
            prior_dim=8,
            num_layers=2,
            attention_dim=8,
            num_heads=2,
            gate_hidden_dim=8,
            dropout=0.0,
            initial_strength=0.0,
            learnable_strength=True,
        )
        features = tuple(torch.randn(2, 3, 8, 4, 4) for _ in range(2))

        enhanced = injector(features, prior_encoder(batch_size=2))

        for original, output in zip(features, enhanced):
            self.assertTrue(torch.equal(original, output))

        sum(output.square().mean() for output in enhanced).backward()
        self.assertIsNotNone(injector.raw_strength.grad)
        self.assertGreater(float(injector.raw_strength.grad.abs().sum()), 0.0)

    def test_multisource_diagnostics_report_attention_mass_by_source(self) -> None:
        injector = TemporalFeaturePyramidPriorInjection(
            vision_dim=4,
            prior_dim=4,
            num_layers=1,
            attention_dim=4,
            num_heads=2,
            gate_hidden_dim=4,
            dropout=0.0,
            source_balance_bias_scale=1.0,
            record_diagnostics=True,
        )
        prior = PriorBatch(
            tokens=torch.randn(1, 3, 4),
            mask=torch.ones(1, 3, dtype=torch.bool),
            confidence=torch.ones(1, 3),
            type_ids=torch.tensor([[0, 0, 1]], dtype=torch.long),
            source_names=("source_0", "source_1"),
        )

        injector((torch.randn(1, 2, 4, 2, 2),), prior)
        diagnostics = injector.pop_prior_diagnostics()

        masses = [
            diagnostics["layer_0/source_0/attention_mass"],
            diagnostics["layer_0/source_1/attention_mass"],
        ]
        self.assertAlmostEqual(float(sum(masses)), 1.0, places=6)
        self.assertIn("layer_0/source_0/valid_token_fraction", diagnostics)
        self.assertIn("layer_0/source_1/valid_token_fraction", diagnostics)

    def test_spatial_film_injector_records_channel_and_source_diagnostics(self) -> None:
        injector = TemporalFeaturePyramidPriorInjection(
            vision_dim=4,
            prior_dim=4,
            num_layers=1,
            attention_dim=4,
            num_heads=2,
            gate_hidden_dim=4,
            fusion_mode="source_aware_spatial_film",
            source_gate_hidden_dim=4,
            film_hidden_dim=4,
            dropout=0.0,
            source_balance_bias_scale=1.0,
            initial_gate_bias=-1.0,
            initial_strength=0.01,
            record_diagnostics=True,
        )
        prior = PriorBatch(
            tokens=torch.randn(1, 3, 4),
            mask=torch.ones(1, 3, dtype=torch.bool),
            confidence=torch.ones(1, 3),
            type_ids=torch.tensor([[0, 0, 1]], dtype=torch.long),
            source_names=("source_0", "source_1"),
        )

        original = torch.randn(1, 2, 4, 2, 2, requires_grad=True)
        enhanced = injector((original,), prior)[0]
        diagnostics = injector.pop_prior_diagnostics()

        self.assertFalse(torch.equal(original, enhanced))
        self.assertIn("layer_0/film_scale_abs_mean", diagnostics)
        self.assertIn("layer_0/film_shift_abs_mean", diagnostics)
        masses = (
            diagnostics["layer_0/source_0/attention_mass"]
            + diagnostics["layer_0/source_1/attention_mass"]
        )
        self.assertAlmostEqual(float(masses), 1.0, places=6)
        enhanced.square().mean().backward()
        self.assertIsNotNone(injector.fusion.film[-1].weight.grad)

    def test_same_pre_decoder_module_runs_before_two_decoder_families(self) -> None:
        for decoder_name in (
            "3d_aware_dpt",
            "temporal_readout_single_layer_dpt",
        ):
            with self.subTest(decoder=decoder_name):
                model = build_cached_feature_model(
                    small_prior_config(decoder_name),
                    in_channels=8,
                    num_layers=4,
                )
                batch = {
                    "temporal_features_by_layer": torch.randn(1, 4, 3, 8, 4, 4),
                    "months": torch.tensor([[9, 10, 11]]),
                    "target": torch.zeros(1, 16, 16, dtype=torch.long),
                }

                logits = model(batch)

                self.assertEqual(tuple(logits.shape), (1, 19, 16, 16))
                self.assertIsNotNone(model.prior_token_encoder)
                self.assertIsNotNone(model.pre_decoder_prior_injection)
                self.assertEqual(model.pre_decoder_prior_injection.num_layers, 4)
                self.assertTrue(model.pre_decoder_prior_injection.record_diagnostics)
                self.assertTrue(
                    model.pre_decoder_prior_injection.pop_prior_diagnostics()
                )
                self.assertIsNone(model.temporal_phenology_prior)
                logits.mean().backward()
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        for parameter in model.prior_token_encoder.parameters()
                    )
                )

    def test_legacy_and_ca_hpi_are_mutually_exclusive(self) -> None:
        config = small_prior_config("3d_aware_dpt")
        config["phenology"] = {
            "enabled": True,
            "path": str(EXTERNAL_PRIOR_PATH),
        }

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_cached_feature_model(config, in_channels=8, num_layers=4)


if __name__ == "__main__":
    unittest.main()
