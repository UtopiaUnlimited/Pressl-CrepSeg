from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from train import Trainer


class BatchLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(1, 1)
        self.decoder = nn.Linear(1, 1)
        self.diagnostics = ScalarDiagnosticProvider()
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False

    def forward(self, batch: dict) -> torch.Tensor:
        return self.decoder(self.diagnostics(batch["features"]))


class ScalarDiagnosticProvider(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_count = 0
        self.latest: dict[str, torch.Tensor] = {}

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        self.latest = {
            "layer_0/strength": features.new_tensor(float(self.forward_count))
        }
        return features

    def pop_prior_diagnostics(self) -> dict[str, torch.Tensor]:
        diagnostics = self.latest
        self.latest = {}
        return diagnostics


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class TrainerAccumulationTest(unittest.TestCase):
    def _build_trainer(self, root: Path, accumulation: int = 1) -> tuple[Trainer, CountingSGD]:
        model = BatchLinear()
        optimizer = CountingSGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        batches = [
            {
                "features": torch.ones(1, 1),
                "target": torch.zeros(1, 1),
            }
            for _ in range(3)
        ]
        trainer = Trainer(
            model=model,
            criterion=nn.MSELoss(),
            optimizer=optimizer,
            scheduler=None,
            train_loader=batches,
            val_loader=[],
            device=torch.device("cpu"),
            num_classes=1,
            log_dir=root / "logs",
            checkpoint_dir=root / "checkpoints",
            gradient_accumulation_steps=accumulation,
            save_trainable_only=True,
        )
        return trainer, optimizer

    def test_steps_optimizer_once_per_accumulation_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer, optimizer = self._build_trainer(Path(directory), accumulation=2)
            _, global_step = trainer.train_epoch(epoch=1, global_step=0, max_batches=None)
            if trainer.writer is not None:
                trainer.writer.close()

        self.assertEqual(global_step, 3)
        self.assertEqual(optimizer.step_count, 2)
        self.assertEqual(
            trainer.last_train_prior_diagnostics,
            {"layer_0/strength": 2.0},
        )

    def test_trainable_only_checkpoint_excludes_frozen_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer, _ = self._build_trainer(root)
            trainer.save_checkpoint("model.pt", epoch=1, val_loss=0.5, val_miou=0.25)
            checkpoint = torch.load(root / "checkpoints" / "model.pt", weights_only=False)
            if trainer.writer is not None:
                trainer.writer.close()

        self.assertTrue(checkpoint["trainable_only"])
        self.assertEqual(set(checkpoint["model"]), {"decoder.weight", "decoder.bias"})


if __name__ == "__main__":
    unittest.main()
