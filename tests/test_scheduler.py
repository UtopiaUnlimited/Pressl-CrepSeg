from __future__ import annotations

import unittest

import torch

from train import build_scheduler


class WarmupCosineSchedulerTest(unittest.TestCase):
    def test_starts_at_zero_and_reaches_warmup_peak(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        config = {
            "train": {"epochs": 4},
            "scheduler": {
                "name": "warmup_cosine",
                "warmup_epochs": 1,
                "min_lr": 0.01,
            },
        }

        scheduler = build_scheduler(config, optimizer, steps_per_epoch=2)

        self.assertIsNotNone(scheduler)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)


if __name__ == "__main__":
    unittest.main()
