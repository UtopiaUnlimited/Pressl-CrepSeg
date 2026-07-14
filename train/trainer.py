from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import ConfusionMatrix, macro_f1, mean_iou, pixel_accuracy
from .plotting import save_training_history


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
        amp_dtype: str = "float16",
        log_dir: str | Path = "logs",
        checkpoint_dir: str | Path = "checkpoints",
        ignore_index: int | None = None,
        save_best: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float | None = None,
        save_trainable_only: bool = False,
        save_last: bool = False,
        early_stopping: dict | None = None,
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
        amp_dtype = str(amp_dtype).lower()
        if amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'.")
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self.ignore_index = ignore_index
        self.save_best = save_best
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1.")
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.save_trainable_only = bool(save_trainable_only)
        self.save_last = bool(save_last)
        early_stopping = early_stopping or {}
        self.early_stopping_enabled = bool(early_stopping.get("enabled", False))
        self.early_stopping_monitor = str(early_stopping.get("monitor", "val_miou"))
        self.early_stopping_mode = str(
            early_stopping.get(
                "mode",
                "min" if self.early_stopping_monitor == "val_loss" else "max",
            )
        ).lower()
        self.early_stopping_patience = int(early_stopping.get("patience", 12))
        self.early_stopping_min_delta = float(early_stopping.get("min_delta", 0.0))
        self.early_stopping_start_epoch = int(early_stopping.get("start_epoch", 1))
        if self.early_stopping_monitor not in {"val_loss", "val_miou"}:
            raise ValueError("early_stopping.monitor must be 'val_loss' or 'val_miou'.")
        if self.early_stopping_mode not in {"min", "max"}:
            raise ValueError("early_stopping.mode must be 'min' or 'max'.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping.patience must be at least 1.")
        self.early_stopping_best = (
            float("inf") if self.early_stopping_mode == "min" else float("-inf")
        )
        self.early_stopping_bad_epochs = 0
        self.stopped_epoch: int | None = None
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp and self.amp_dtype == torch.float16,
        )
        self.best_val_loss = float("inf")
        self.best_val_miou = float("-inf")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, float]] = []

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            SummaryWriter = None
        self.writer = (
            SummaryWriter(log_dir=str(self.log_dir)) if SummaryWriter is not None else None
        )

    def fit(
        self,
        epochs: int,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
    ) -> dict[str, float | int | bool | None]:
        global_step = 0
        last_epoch = 0
        last_val_loss = float("nan")
        last_val_miou = float("nan")
        last_val_acc = float("nan")
        last_val_f1 = float("nan")
        for epoch in range(1, epochs + 1):
            train_loss, global_step = self.train_epoch(epoch, global_step, max_train_batches)
            validation = self.validate(epoch, max_val_batches)
            val_loss, val_miou, val_acc, val_f1 = self._unpack_validation_metrics(
                validation
            )
            lr = self.optimizer.param_groups[0]["lr"]

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_miou": val_miou,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                }
            )

            print(
                f"epoch={epoch} train_loss={train_loss:.5f} "
                f"val_loss={val_loss:.5f} val_miou={val_miou:.5f} "
                f"val_acc={val_acc:.5f} val_f1={val_f1:.5f} lr={lr:.8f}"
            )
            if self.writer is not None:
                self.writer.add_scalar("loss/train", train_loss, epoch)
                self.writer.add_scalar("loss/val", val_loss, epoch)
                self.writer.add_scalar("metrics/val_miou", val_miou, epoch)
                self.writer.add_scalar("metrics/val_acc", val_acc, epoch)
                self.writer.add_scalar("metrics/val_f1", val_f1, epoch)
                self.writer.add_scalar("lr", lr, epoch)
                if self.device.type == "cuda":
                    self.writer.add_scalar("gpu/max_memory_allocated", torch.cuda.max_memory_allocated(), epoch)

            if self.save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    "best_val_loss.pt", epoch, val_loss, val_miou, val_acc, val_f1
                )
                # Keep the historical filename compatible with existing commands.
                self.save_checkpoint("best.pt", epoch, val_loss, val_miou, val_acc, val_f1)

            if self.save_best and val_miou > self.best_val_miou:
                self.best_val_miou = val_miou
                self.save_checkpoint(
                    "best_val_miou.pt", epoch, val_loss, val_miou, val_acc, val_f1
                )

            last_epoch = epoch
            last_val_loss = val_loss
            last_val_miou = val_miou
            last_val_acc = val_acc
            last_val_f1 = val_f1
            if self.early_stopping_enabled:
                metric = val_loss if self.early_stopping_monitor == "val_loss" else val_miou
                if self._is_early_stopping_improvement(metric):
                    self.early_stopping_best = metric
                    self.early_stopping_bad_epochs = 0
                elif epoch >= self.early_stopping_start_epoch:
                    self.early_stopping_bad_epochs += 1

                if self.writer is not None:
                    self.writer.add_scalar(
                        "early_stopping/bad_epochs",
                        self.early_stopping_bad_epochs,
                        epoch,
                    )
                if self.early_stopping_bad_epochs >= self.early_stopping_patience:
                    self.stopped_epoch = epoch
                    print(
                        f"early_stopping epoch={epoch} monitor={self.early_stopping_monitor} "
                        f"best={self.early_stopping_best:.5f} "
                        f"patience={self.early_stopping_patience}"
                    )
                    break

        if self.save_last and last_epoch > 0:
            self.save_checkpoint(
                "last.pt",
                last_epoch,
                last_val_loss,
                last_val_miou,
                last_val_acc,
                last_val_f1,
            )

        if self.writer is not None:
            self.writer.close()
        save_training_history(self.history, self.log_dir)
        return {
            "epochs_trained": last_epoch,
            "last_val_loss": last_val_loss,
            "last_val_miou": last_val_miou,
            "last_val_acc": last_val_acc,
            "last_val_f1": last_val_f1,
            "best_val_loss": self.best_val_loss,
            "best_val_miou": self.best_val_miou,
            "stopped_early": self.stopped_epoch is not None,
            "stopped_epoch": self.stopped_epoch,
        }

    @staticmethod
    def _unpack_validation_metrics(
        validation: tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        """Accept the historical 2-tuple from custom Trainer subclasses."""

        if len(validation) == 4:
            return tuple(float(value) for value in validation)
        if len(validation) == 2:
            val_loss, val_miou = validation
            return float(val_loss), float(val_miou), 0.0, 0.0
        raise ValueError(
            "validate() must return (loss, mIoU) or (loss, mIoU, accuracy, F1)."
        )

    def _is_early_stopping_improvement(self, metric: float) -> bool:
        if self.early_stopping_mode == "min":
            return metric < self.early_stopping_best - self.early_stopping_min_delta
        return metric > self.early_stopping_best + self.early_stopping_min_delta

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
        effective_batches = len(self.train_loader)
        if max_batches is not None:
            effective_batches = min(effective_batches, max_batches)
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(iterator, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            target = batch["target"].to(self.device, non_blocking=True)
            batch["target"] = target

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp,
            ):
                logits = self.model(batch)
                loss = self.criterion(logits, target)

            group_start = (batch_count // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
            group_size = min(
                self.gradient_accumulation_steps,
                effective_batches - group_start,
            )
            self.scaler.scale(loss / max(1, group_size)).backward()

            loss_value = float(loss.detach().item())
            total_loss += loss_value
            batch_count += 1
            global_step += 1
            should_update = (
                batch_count % self.gradient_accumulation_steps == 0
                or batch_count == effective_batches
            )
            if should_update:
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
            iterator.set_postfix(loss=f"{loss_value:.4f}")
            if self.writer is not None:
                self.writer.add_scalar("loss/train_step", loss_value, global_step)

        return total_loss / max(1, batch_count), global_step

    @torch.no_grad()
    def validate(
        self,
        epoch: int,
        max_batches: int | None = None,
    ) -> tuple[float, float, float, float]:
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
        accuracy = pixel_accuracy(confusion.matrix)
        f1, _ = macro_f1(confusion.matrix)
        return total_loss / max(1, batch_count), miou, accuracy, f1

    def save_checkpoint(
        self,
        name: str,
        epoch: int,
        val_loss: float,
        val_miou: float,
        val_acc: float = float("nan"),
        val_f1: float = float("nan"),
    ) -> None:
        path = self.checkpoint_dir / name
        state_dict = self.model.state_dict()
        if self.save_trainable_only:
            trainable_names = {
                parameter_name
                for parameter_name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
            state_dict = {
                parameter_name: value
                for parameter_name, value in state_dict.items()
                if parameter_name in trainable_names
            }
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_miou": val_miou,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "model": state_dict,
                "optimizer": self.optimizer.state_dict(),
                "trainable_only": self.save_trainable_only,
            },
            path,
        )
