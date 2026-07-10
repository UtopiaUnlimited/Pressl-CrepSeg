from __future__ import annotations

import unittest

import numpy as np
import torch

from data.pastis import (
    GALILEO_S2_MEAN,
    GALILEO_S2_STD,
    aggregate_monthly_s2,
    normalize_s2_for_galileo,
    remap_void_label,
)
from losses.ce import cross_entropy_loss
from utils.cache_paths import feature_cache_dir


class PastisPaperProtocolTest(unittest.TestCase):
    def test_monthly_aggregation_uses_crop_year_and_interpolates_missing_month(self) -> None:
        dates = [
            20180920,
            20181005,
            20181025,
            20181110,
            20190110,
            20190210,
            20190310,
            20190410,
            20190510,
            20190610,
            20190710,
            20190810,
            20190910,
            20191010,
        ]
        values = [9, 10, 14, 20, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
        s2 = np.stack(
            [np.full((10, 2, 2), value, dtype=np.float32) for value in values],
            axis=0,
        )

        monthly, months, representative_dates, counts = aggregate_monthly_s2(s2, dates)

        self.assertEqual(monthly.shape, (12, 10, 2, 2))
        self.assertEqual(representative_dates.tolist()[0], 20181001)
        self.assertEqual(representative_dates.tolist()[-1], 20190901)
        self.assertEqual(months.tolist(), [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(counts.tolist(), [2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(monthly[0], 12.0)
        np.testing.assert_allclose(monthly[2], 30.0)

    def test_galileo_unclipped_scaling(self) -> None:
        means = GALILEO_S2_MEAN[None, :, None, None]
        stds = GALILEO_S2_STD[None, :, None, None]
        s2 = np.concatenate([means - 2 * stds, means, means + 2 * stds], axis=0)

        normalized = normalize_s2_for_galileo(s2, std_multiplier=2.0)

        np.testing.assert_allclose(normalized[:, :, 0, 0], [
            np.zeros(10),
            np.full(10, 0.5),
            np.ones(10),
        ], rtol=1e-6, atol=1e-6)

    def test_void_label_is_ignored(self) -> None:
        target = np.asarray([[0, 18, 19]], dtype=np.int64)
        remapped = remap_void_label(target, void_label=19, ignore_index=-1)
        self.assertEqual(remapped.tolist(), [[0, 18, -1]])

    def test_all_void_batch_has_finite_zero_loss(self) -> None:
        logits = torch.randn(1, 19, 4, 4, requires_grad=True)
        target = torch.full((1, 4, 4), -1, dtype=torch.long)
        loss = cross_entropy_loss(logits, target, ignore_index=-1)
        self.assertEqual(float(loss.item()), 0.0)
        loss.backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_cache_path_records_paper_protocol(self) -> None:
        config = {
            "data": {
                "temporal_aggregation": "monthly",
                "selected_timesteps": 12,
                "tile_size": 64,
            },
            "encoder": {
                "name": "galileo-base-patch8",
                "patch_size": 4,
                "hidden_layers": [3, 6, 9, 12],
            },
        }
        self.assertEqual(
            feature_cache_dir(config, "train"),
            "data/cache/galileo-base-patch8/"
            "monthly12_tile64_patch4_hl3-6-9-12_train",
        )


if __name__ == "__main__":
    unittest.main()
