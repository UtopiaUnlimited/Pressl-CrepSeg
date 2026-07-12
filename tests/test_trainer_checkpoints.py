from pathlib import Path

import torch

from train.trainer import Trainer


class MetricSequenceTrainer(Trainer):
    def __init__(self, *args, metrics: list[tuple[float, float]], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = iter(metrics)

    def train_epoch(self, epoch, global_step, max_batches):
        return 1.0, global_step + 1

    def validate(self, epoch, max_batches=None):
        return next(self.metrics)


def test_saves_best_loss_and_best_miou_independently(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = MetricSequenceTrainer(
        model=model,
        criterion=torch.nn.MSELoss(),
        optimizer=optimizer,
        scheduler=None,
        train_loader=[],
        val_loader=[],
        device=torch.device("cpu"),
        num_classes=2,
        checkpoint_dir=tmp_path,
        log_dir=tmp_path / "logs",
        metrics=[(0.8, 0.30), (0.7, 0.25), (0.9, 0.40)],
    )

    trainer.fit(epochs=3)

    best_loss = torch.load(tmp_path / "best_val_loss.pt", weights_only=False)
    legacy_best = torch.load(tmp_path / "best.pt", weights_only=False)
    best_miou = torch.load(tmp_path / "best_val_miou.pt", weights_only=False)

    assert best_loss["epoch"] == 2
    assert best_loss["val_loss"] == 0.7
    assert best_loss["val_miou"] == 0.25
    assert legacy_best["epoch"] == best_loss["epoch"]
    assert best_miou["epoch"] == 3
    assert best_miou["val_loss"] == 0.9
    assert best_miou["val_miou"] == 0.40
