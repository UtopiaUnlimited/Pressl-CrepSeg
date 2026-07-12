from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from train import Trainer


class MetricSequenceTrainer(Trainer):
    def __init__(self, *args, metrics: list[tuple[float, float]], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = iter(metrics)

    def train_epoch(self, epoch, global_step, max_batches):
        return 1.0, global_step + 1

    def validate(self, epoch, max_batches=None):
        return next(self.metrics)


class EarlyStoppingTest(unittest.TestCase):
    def test_stops_after_patience_without_miou_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(1, 1)
            trainer = MetricSequenceTrainer(
                model=model,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                scheduler=None,
                train_loader=[],
                val_loader=[],
                device=torch.device("cpu"),
                num_classes=2,
                checkpoint_dir=root / "checkpoints",
                log_dir=root / "logs",
                save_best=True,
                early_stopping={
                    "enabled": True,
                    "monitor": "val_miou",
                    "mode": "max",
                    "patience": 2,
                    "min_delta": 0.01,
                    "start_epoch": 1,
                },
                metrics=[
                    (0.9, 0.10),
                    (0.8, 0.20),
                    (0.7, 0.205),
                    (0.6, 0.19),
                    (0.5, 0.30),
                ],
            )

            summary = trainer.fit(epochs=5)

            best_miou = torch.load(
                root / "checkpoints" / "best_val_miou.pt",
                weights_only=False,
            )

        self.assertTrue(summary["stopped_early"])
        self.assertEqual(summary["stopped_epoch"], 4)
        self.assertEqual(summary["epochs_trained"], 4)
        self.assertAlmostEqual(best_miou["val_miou"], 0.205)

    def test_can_monitor_loss_in_min_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(1, 1)
            trainer = Trainer(
                model=model,
                criterion=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                scheduler=None,
                train_loader=[],
                val_loader=[],
                device=torch.device("cpu"),
                num_classes=2,
                checkpoint_dir=root / "checkpoints",
                log_dir=root / "logs",
                early_stopping={
                    "enabled": True,
                    "monitor": "val_loss",
                    "mode": "min",
                    "patience": 1,
                    "min_delta": 0.05,
                },
            )

            self.assertTrue(trainer._is_early_stopping_improvement(1.0))
            trainer.early_stopping_best = 1.0
            self.assertTrue(trainer._is_early_stopping_improvement(0.90))
            self.assertFalse(trainer._is_early_stopping_improvement(0.97))
            if trainer.writer is not None:
                trainer.writer.close()


if __name__ == "__main__":
    unittest.main()
