from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from models import build_cached_feature_model
from models.prior_sources import build_prior_token_encoder
from scripts.prepare_environment_prior_tables import prepare_climate, prepare_soil


CLIMATE_FIELDS = ("t2m_c", "tp_mm", "ssrd_mj_m2", "swvl1")
SOIL_FIELDS = (
    "ph",
    "soc_gkg",
    "clay_pct",
    "sand_pct",
    "cec_cmolkg",
    "nitrogen_gkg",
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_stats(path: Path, features: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mean": {feature: 0.0 for feature in features},
                "std": {feature: 1.0 for feature in features},
            },
            handle,
        )


def _small_environment_config(climate_path: Path, climate_stats: Path, soil_path: Path, soil_stats: Path) -> dict:
    return {
        "data": {"num_classes": 19},
        "encoder": {"hidden_layers": [3, 6, 9, 12]},
        "model": {
            "decoder": "temporal_readout_single_layer_dpt",
            "decoder_channels": 16,
            "decoder_blocks": 1,
            "dropout": 0.0,
            "temporal_readout": {"num_months": 12, "hidden_channels": 8, "dropout": 0.0},
        },
        "prior_injection": {
            "enabled": True,
            "method": "ca_hpi",
            "token_dim": 16,
            "sources": [
                {
                    "kind": "climate_table",
                    "path": str(climate_path),
                    "stats_path": str(climate_stats),
                    "features": list(CLIMATE_FIELDS),
                },
                {
                    "kind": "soil_table",
                    "path": str(soil_path),
                    "stats_path": str(soil_stats),
                    "features": list(SOIL_FIELDS),
                    "depths": ["0-5", "5-15", "15-30"],
                },
            ],
            "encoder": {"hidden_dim": 16, "time_frequencies": 2, "dropout": 0.0},
            "fusion": {
                "attention_dim": 16,
                "num_heads": 4,
                "gate_hidden_dim": 8,
                "dropout": 0.0,
                "confidence_bias_scale": 1.0,
                "initial_strength": 0.2,
                "learnable_strength": True,
            },
        },
    }


class EnvironmentPriorEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.climate_path = root / "climate.csv"
        self.climate_stats = root / "climate_stats.json"
        self.soil_path = root / "soil.csv"
        self.soil_stats = root / "soil_stats.json"
        climate_rows: list[dict[str, object]] = []
        for patch_id in (10000, 10001):
            for month in range(1, 13):
                climate_rows.append(
                    {
                        "patch_id": patch_id,
                        "month": month,
                        "t2m_c": patch_id / 1000 + month,
                        "tp_mm": month * 2,
                        "ssrd_mj_m2": month * 3,
                        "swvl1": month / 100,
                        "valid": "true",
                        "confidence": 1.0 if month != 12 else 0.5,
                    }
                )
        _write_csv(
            self.climate_path,
            ["patch_id", "month", *CLIMATE_FIELDS, "valid", "confidence"],
            climate_rows,
        )
        _write_stats(self.climate_stats, CLIMATE_FIELDS)

        soil_rows: list[dict[str, object]] = []
        for patch_id in (10000, 10001):
            for depth_index, depth in enumerate(("0-5", "5-15", "15-30")):
                soil_rows.append(
                    {
                        "patch_id": patch_id,
                        "depth_cm": depth,
                        "ph": 6.0 + depth_index,
                        "soc_gkg": 10.0 + depth_index,
                        "clay_pct": 20.0 + depth_index,
                        "sand_pct": 30.0 + depth_index,
                        "cec_cmolkg": 4.0 + depth_index,
                        "nitrogen_gkg": 1.0 + depth_index,
                        "valid": "true",
                        "confidence": 0.8,
                    }
                )
        _write_csv(
            self.soil_path,
            ["patch_id", "depth_cm", *SOIL_FIELDS, "valid", "confidence"],
            soil_rows,
        )
        _write_stats(self.soil_stats, SOIL_FIELDS)
        self.config = _small_environment_config(
            self.climate_path,
            self.climate_stats,
            self.soil_path,
            self.soil_stats,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_combined_m2_m3_batch_has_12_climate_and_3_soil_tokens(self) -> None:
        encoder = build_prior_token_encoder(self.config)
        prior = encoder(batch_size=2, batch={"patch_id": [10001, 10000]})

        self.assertEqual(tuple(prior.tokens.shape), (2, 15, 16))
        self.assertTrue(prior.mask.all())
        self.assertTrue(torch.allclose(prior.confidence[:, 11], torch.full((2,), 0.5)))
        self.assertTrue(torch.allclose(prior.confidence[:, 12:], torch.full((2, 3), 0.8)))
        self.assertTrue(torch.equal(prior.type_ids[:, :12], torch.zeros(2, 12, dtype=torch.long)))
        self.assertTrue(torch.equal(prior.type_ids[:, 12:], torch.ones(2, 3, dtype=torch.long)))

    def test_missing_patch_is_rejected_by_default(self) -> None:
        encoder = build_prior_token_encoder(self.config)
        with self.assertRaisesRegex(KeyError, "no rows for patch_id"):
            encoder(batch_size=1, batch={"patch_id": [99999]})

    def test_cached_temporal_decoder_consumes_m2_m3_priors(self) -> None:
        model = build_cached_feature_model(self.config, in_channels=8, num_layers=4)
        batch = {
            "temporal_features_by_layer": torch.randn(2, 4, 3, 8, 4, 4),
            "months": torch.tensor([[9, 10, 11], [9, 10, 11]]),
            "patch_id": [10000, 10001],
            "target": torch.zeros(2, 16, 16, dtype=torch.long),
        }
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2, 19, 16, 16))
        logits.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.prior_token_encoder.parameters()))

    def test_preparation_freezes_tables_and_uses_train_folds_for_statistics(self) -> None:
        root = Path(self.temp_dir.name)
        climate_output = root / "climate_frozen.csv"
        climate_stats = root / "climate_frozen_stats.json"
        soil_output = root / "soil_frozen.csv"
        soil_stats = root / "soil_frozen_stats.json"
        patch_folds = {10000: 1, 10001: 4}

        prepare_climate(
            str(self.climate_path),
            str(climate_output),
            str(climate_stats),
            patch_folds,
            {1},
            allow_incomplete=False,
            source_version="test",
        )
        prepare_soil(
            str(self.soil_path),
            str(soil_output),
            str(soil_stats),
            patch_folds,
            {1},
            allow_incomplete=False,
            source_version="test",
        )
        with climate_stats.open("r", encoding="utf-8") as handle:
            climate_payload = json.load(handle)
        with soil_stats.open("r", encoding="utf-8") as handle:
            soil_payload = json.load(handle)
        self.assertEqual(climate_payload["train_folds"], [1])
        self.assertEqual(soil_payload["train_folds"], [1])
        with climate_output.open("r", encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 24)
        with soil_output.open("r", encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 6)


if __name__ == "__main__":
    unittest.main()
