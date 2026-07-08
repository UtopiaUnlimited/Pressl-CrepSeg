from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import ConfusionMatrix, mean_iou


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_classes: int,
        amp: bool = False,
        log_dir: str | Path = "logs",
        checkpoint_dir: str | Path = "checkpoints",
        ignore_index: int | None = None,
        save_best: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = int(num_classes)
        self.amp = bool(amp and device.type == "cuda")
        self.ignore_index = ignore_index
        self.save_best = save_best
        self.best_val_loss = float("inf")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            SummaryWriter = None
        self.writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter is not None else None

    def fit(
        self,
        epochs: int,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
    ) -> None:
        global_step = 0
        for epoch in range(1, epochs + 1):
            train_loss, global_step = self.train_epoch(epoch, global_step, max_train_batches)
            val_loss, val_miou = self.validate(epoch, max_val_batches)
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"epoch={epoch} train_loss={train_loss:.5f} "
                f"val_loss={val_loss:.5f} val_miou={val_miou:.5f} lr={lr:.8f}"
            )
            if self.writer is not None:
                self.writer.add_scalar("loss/train", train_loss, epoch)
                self.writer.add_scalar("loss/val", val_loss, epoch)
                self.writer.add_scalar("metrics/val_miou", val_miou, epoch)
                self.writer.add_scalar("lr", lr, epoch)
                if self.device.type == "cuda":
                    self.writer.add_scalar("gpu/max_memory_allocated", torch.cuda.max_memory_allocated(), epoch)

            if self.save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt", epoch, val_loss)

        if self.writer is not None:
            self.writer.close()

    def train_epoch(
        self,
        epoch: int,
        global_step: int,
        max_batches: int | None,
    ) -> tuple[float, int]:
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        iterator = tqdm(self.train_loader, desc=f"train {epoch}", leave=False)
        for batch_index, batch in enumerate(iterator, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            target = batch["target"].to(self.device, non_blocking=True)
            batch["target"] = target

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(batch)
                loss = self.criterion(logits, target)

            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            loss_value = float(loss.detach().item())
            total_loss += loss_value
            batch_count += 1
            global_step += 1
            iterator.set_postfix(loss=f"{loss_value:.4f}")
            if self.writer is not None:
                self.writer.add_scalar("loss/train_step", loss_value, global_step)

        return total_loss / max(1, batch_count), global_step

    @torch.no_grad()
    def validate(self, epoch: int, max_batches: int | None = None) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        confusion = ConfusionMatrix(num_classes=self.num_classes, ignore_index=self.ignore_index)

        iterator = tqdm(self.val_loader, desc=f"val {epoch}", leave=False)
        for batch_index, batch in enumerate(iterator, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            target = batch["target"].to(self.device, non_blocking=True)
            batch["target"] = target

            logits = self.model(batch)
            loss = self.criterion(logits, target)
            total_loss += float(loss.item())
            batch_count += 1
            confusion.update(logits, target)

        miou, _ = mean_iou(confusion.matrix)
        return total_loss / max(1, batch_count), miou

    def save_checkpoint(self, name: str, epoch: int, val_loss: float) -> None:
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
