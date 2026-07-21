from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models import ClassTemporalPrototypeMemory, build_cached_feature_model
from scripts.build_class_temporal_prototypes import build_prototype_archives


def _write_temporal_cache(
    path: Path,
    patch_id: int,
    label: int,
    value: float,
) -> None:
    temporal = np.full((4, 2, 4, 2, 2), value, dtype=np.float16)
    np.savez_compressed(
        path,
        temporal_features_by_layer=temporal,
        months=np.asarray([0, 1], dtype=np.int64),
        target=np.full((4, 4), label, dtype=np.int64),
        patch_id=np.asarray(patch_id),
        fold=np.asarray(1),
        cache_format=np.asarray("temporal_v2"),
    )


def _write_memory_archive(path: Path, prototypes_per_group: int = 1) -> None:
    rng = np.random.default_rng(7)
    prototypes = rng.normal(
        size=(3, 3, 2, prototypes_per_group, 4)
    ).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=-1, keepdims=True)
    metadata = {
        "partition_modulus": 2,
        "feature_layer_index": 3,
        "prototypes_per_group": prototypes_per_group,
    }
    np.savez_compressed(
        path,
        prototypes=prototypes,
        mask=np.ones(prototypes.shape[:-1], dtype=np.bool_),
        confidence=np.ones(prototypes.shape[:-1], dtype=np.float32),
        counts=np.full(prototypes.shape[:-1], 8, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


class ClassTemporalPrototypeBuilderTest(unittest.TestCase):
    def test_builder_writes_k1_and_k4_crossfit_archives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            _write_temporal_cache(cache_dir / "10000_y0_x0.npz", 10000, 1, 1.0)
            _write_temporal_cache(cache_dir / "10001_y0_x0.npz", 10001, 2, 2.0)

            outputs = build_prototype_archives(
                cache_dir=cache_dir,
                output_dir=root / "priors",
                num_classes=3,
                train_folds=(1, 2, 3),
                ignore_index=-1,
                prototypes_per_group=(1, 4),
                feature_layer_index=-1,
                min_tokens_per_prototype=1,
                seed=42,
            )

            self.assertEqual(len(outputs), 2)
            by_name = {path.name: path for path in outputs}
            self.assertIn(
                "pastis_fold123_final_layer_class_temporal_prototypes_k1_online_v1.npz",
                by_name,
            )
            self.assertIn(
                "pastis_fold123_final_layer_class_temporal_prototypes_k4_online_v1.npz",
                by_name,
            )
            with np.load(by_name[next(name for name in by_name if "_k4_" in name)]) as archive:
                self.assertEqual(tuple(archive["prototypes"].shape), (3, 3, 2, 4, 4))
                self.assertEqual(int(archive["counts"][2, 1, 0].sum()), 4)
                metadata = json.loads(str(archive["metadata_json"].item()))
                self.assertEqual(metadata["feature_layer_index"], 3)
                self.assertEqual(metadata["partition_file_counts"], [1, 1])


class ClassTemporalPrototypeMemoryTest(unittest.TestCase):
    def test_crossfit_banks_and_zero_strength_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "memory.npz"
            _write_memory_archive(archive)
            memory = ClassTemporalPrototypeMemory(
                archive_path=archive,
                vision_dim=4,
                num_classes=3,
                num_months=2,
                train_folds=(1, 2, 3),
                feature_layer_index=3,
                query_dim=4,
                gate_hidden_dim=4,
                initial_strength=0.0,
                record_diagnostics=True,
            )
            selected = memory._bank_indices(
                {
                    "patch_id": [10000, 10001, 20000],
                    "fold": torch.tensor([1, 2, 4]),
                },
                batch_size=3,
                device=torch.device("cpu"),
            )
            self.assertEqual(selected.tolist(), [1, 0, 2])

            features = tuple(torch.randn(3, 2, 4, 2, 2) for _ in range(4))
            enhanced = memory(
                features,
                months=torch.tensor([[0, 1], [0, 1], [0, 1]]),
                batch={
                    "patch_id": [10000, 10001, 20000],
                    "fold": torch.tensor([1, 2, 4]),
                },
                layer_indices=(0, 1, 2, 3),
            )
            for original, output in zip(features, enhanced):
                self.assertTrue(torch.equal(original, output))
            diagnostics = memory.pop_prior_diagnostics()
            self.assertEqual(float(diagnostics["strength"]), 0.0)
            self.assertTrue(all(torch.isfinite(value) for value in diagnostics.values()))

    def test_cached_model_accepts_prototype_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "memory.npz"
            _write_memory_archive(archive)
            config = {
                "data": {
                    "num_classes": 3,
                    "selected_timesteps": 2,
                    "train_folds": [1, 2, 3],
                },
                "encoder": {"hidden_layers": [3, 6, 9, 12]},
                "model": {
                    "decoder": "3d_aware_dpt",
                    "decoder_channels": 16,
                    "num_heads": 4,
                    "spatial_window": 2,
                    "global_3d_blocks": 1,
                    "fusion_blocks_per_stage": 1,
                    "mlp_expansion": 2,
                    "drop_path": 0.0,
                    "temporal_pool_heads": 4,
                    "num_months": 2,
                    "dropout": 0.0,
                },
                "prior_injection": {
                    "enabled": True,
                    "method": "class_temporal_prototype",
                    "source": {"path": str(archive), "prototypes_per_group": 1},
                    "fusion": {
                        "feature_layer_index": -1,
                        "query_dim": 4,
                        "gate_hidden_dim": 4,
                        "dropout": 0.0,
                        "initial_strength": 0.01,
                    },
                    "diagnostics": {"enabled": True},
                },
            }
            model = build_cached_feature_model(config, in_channels=4, num_layers=4)
            batch = {
                "temporal_features_by_layer": torch.randn(2, 4, 2, 4, 2, 2),
                "months": torch.tensor([[0, 1], [0, 1]]),
                "target": torch.zeros(2, 8, 8, dtype=torch.long),
                "patch_id": [10000, 20000],
                "fold": torch.tensor([1, 4]),
            }
            logits = model(batch)
            self.assertEqual(tuple(logits.shape), (2, 3, 8, 8))
            self.assertIsNotNone(model.prototype_memory)
            self.assertTrue(model.prototype_memory.pop_prior_diagnostics())


if __name__ == "__main__":
    unittest.main()
