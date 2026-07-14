from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data import CachedFeatureDataset, cached_feature_collate_fn


def _metadata() -> dict:
    return {
        "patch_id": np.asarray(10000),
        "sample_id": np.asarray("10000_y0_x0"),
        "tile_id": np.asarray(0),
        "tile_y": np.asarray(0),
        "tile_x": np.asarray(0),
        "fold": np.asarray(1),
        "months": np.asarray([9, 10, 11], dtype=np.int64),
        "target": np.zeros((16, 16), dtype=np.int64),
    }


class CachedFeatureDatasetTest(unittest.TestCase):
    def test_temporal_only_cache_derives_legacy_spatial_interfaces(self) -> None:
        temporal = np.arange(4 * 3 * 8 * 4 * 4, dtype=np.float32).reshape(4, 3, 8, 4, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "10000_y0_x0.npz"
            np.savez_compressed(
                path,
                **_metadata(),
                cache_format=np.asarray("temporal_v2"),
                temporal_features_by_layer=temporal.astype(np.float16),
            )

            dataset = CachedFeatureDataset(directory, load_features_by_layer=True)
            item = dataset[0]

            self.assertEqual(tuple(item["features"].shape), (8, 4, 4))
            self.assertEqual(tuple(item["features_by_layer"].shape), (4, 8, 4, 4))
            self.assertNotIn("temporal_features_by_layer", item)
            np.testing.assert_allclose(
                item["features"].numpy(),
                temporal.astype(np.float16).astype(np.float32)[-1].mean(axis=0),
            )
            np.testing.assert_allclose(
                item["features_by_layer"].numpy(),
                temporal.astype(np.float16).astype(np.float32).mean(axis=1),
            )

    def test_temporal_decoder_receives_layers_and_months(self) -> None:
        temporal = np.ones((4, 3, 8, 4, 4), dtype=np.float16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "10000_y0_x0.npz"
            np.savez_compressed(
                path,
                **_metadata(),
                cache_format=np.asarray("temporal_v2"),
                temporal_features_by_layer=temporal,
            )

            dataset = CachedFeatureDataset(
                directory,
                load_features_by_layer=False,
                load_temporal_features_by_layer=True,
            )
            collated = cached_feature_collate_fn([dataset[0], dataset[0]])

            self.assertEqual(
                tuple(collated["temporal_features_by_layer"].shape),
                (2, 4, 3, 8, 4, 4),
            )
            self.assertEqual(tuple(collated["months"].shape), (2, 3))
            self.assertEqual(collated["cache_format"], ["temporal_v2", "temporal_v2"])

    def test_legacy_spatial_cache_stays_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "10000_y0_x0.npz"
            np.savez_compressed(
                path,
                **_metadata(),
                features=np.ones((8, 4, 4), dtype=np.float32),
                features_by_layer=np.ones((4, 8, 4, 4), dtype=np.float32),
            )

            item = CachedFeatureDataset(directory, load_features_by_layer=True)[0]
            self.assertEqual(tuple(item["features"].shape), (8, 4, 4))
            self.assertEqual(tuple(item["features_by_layer"].shape), (4, 8, 4, 4))
            self.assertEqual(item["cache_format"], "spatial_v1")

            temporal_dataset = CachedFeatureDataset(
                directory,
                load_features_by_layer=False,
                load_temporal_features_by_layer=True,
            )
            with self.assertRaisesRegex(ValueError, "no temporal_features_by_layer"):
                temporal_dataset[0]


if __name__ == "__main__":
    unittest.main()
