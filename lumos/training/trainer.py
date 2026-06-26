import os
import time
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from lumos.model.lumos import LUMOS
from lumos.training.losses import MultiTaskLoss, DirectLearnedWeightedLoss
from lumos.training.schedulers import create_lr_scheduler
from lumos.training.monitor import TrainingMonitor


def _calculate_roc_auc(predictions: torch.Tensor, targets: torch.Tensor) -> list[float]:
    preds_np = predictions.cpu().detach().numpy()
    targets_np = targets.cpu().detach().numpy()
    if len(targets_np.shape) == 1:
        return [roc_auc_score(targets_np, preds_np)]
    return [roc_auc_score(targets_np[:, i], preds_np[:, i]) for i in range(targets_np.shape[1])]


class LUMOSTrainer:
    """Single-GPU trainer for LUMOS models.

    Supports multi-task training with configurable loss aggregation strategies,
    learning rate scheduling, TensorBoard logging, and checkpoint saving.
    """

    def __init__(
        self,
        model: LUMOS,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        target_types: list[str] | None = None,
        target_names: list[str] | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        loss_aggregation: str = "sum",
        lr_scheduler_type: str = "constant",
        lr_scheduler_kwargs: dict | None = None,
        checkpoint_dir: str = "checkpoints",
        tensorboard_dir: str | None = None,
        log_interval: int = 100,
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_types = target_types or []
        self.target_names = target_names or [f"target_{i}" for i in range(model.n_targets)]

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.criterion_binary = nn.BCEWithLogitsLoss()
        self.criterion_continuous = nn.MSELoss(reduction="sum")

        self.loss_aggregator = self._build_loss_aggregator(loss_aggregation)

        params = list(model.parameters())
        if self.loss_aggregator is not None:
            params.extend(list(self.loss_aggregator.parameters()))
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, betas=adam_betas)

        self.scheduler = create_lr_scheduler(self.optimizer, scheduler_type=lr_scheduler_type, **(lr_scheduler_kwargs or {}))

        self.monitor = TrainingMonitor(tensorboard_dir=tensorboard_dir, log_interval=log_interval)
        self.best_primary_metric = 0.0

    def _build_loss_aggregator(self, strategy: str):
        if strategy == "uncertainty_weighted":
            is_reg = torch.tensor([t == "continuous" for t in self.target_types], dtype=torch.bool)
            return MultiTaskLoss(is_regression=is_reg, reduction="sum").to(self.device)
        elif strategy == "learned_weights":
            return DirectLearnedWeightedLoss(n_tasks=len(self.target_types)).to(self.device)
        return None

    def _compute_loss_and_metrics(self, predictions: torch.Tensor, targets: torch.Tensor):
        n_targets = predictions.shape[-1]
        individual_losses = []
        metrics = [-1.0] * n_targets

        for i in range(n_targets):
            pred_i = predictions[..., i]
            target_i = targets[..., i]
            target_type = self.target_types[i] if i < len(self.target_types) else "binary"

            if target_type == "binary":
                loss_i = self.criterion_binary(pred_i, target_i)
                try:
                    auc = _calculate_roc_auc(pred_i.detach().sigmoid().squeeze(), target_i.squeeze())
                    metrics[i] = auc[0] if isinstance(auc, list) else auc
                except ValueError:
                    metrics[i] = -1.0
            else:
                non_zero = target_i != 0
                if non_zero.sum() > 0:
                    mse_sum = self.criterion_continuous(pred_i[non_zero], target_i[non_zero])
                    loss_i = mse_sum / non_zero.sum()
                    mape = (torch.abs(pred_i[non_zero].detach() - target_i[non_zero]) / torch.abs(target_i[non_zero])).mean().item()
                    metrics[i] = mape
                else:
                    loss_i = torch.tensor(0.0, device=self.device)

            individual_losses.append(loss_i)

        return torch.stack(individual_losses), metrics

    def _aggregate_loss(self, individual_losses: torch.Tensor) -> torch.Tensor:
        if self.loss_aggregator is not None:
            return self.loss_aggregator(individual_losses)
        return individual_losses.sum()

    def train(self, epochs: int = 1):
        """Run the training loop.

        Args:
            epochs: Number of passes over the training data.

        Returns:
            Dictionary with training history.
        """
        history = {"train_loss": [], "val_loss": []}
        global_step = 0

        for epoch in range(epochs):
            self.model.train()
            if self.loss_aggregator:
                self.loss_aggregator.train()

            epoch_loss = 0.0
            n_batches = 0

            for batch in self.train_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                predictions = self.model(
                    batch["activity_history"],
                    batch["event_context_history"],
                    batch["static_features"],
                    batch["future_event_context"],
                )
                individual_losses, train_metrics = self._compute_loss_and_metrics(predictions, batch["targets"])
                loss = self._aggregate_loss(individual_losses)

                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                if global_step % self.monitor.log_interval == 0:
                    log_payload = {
                        "training/loss": loss.item(),
                        "training/learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                    for idx, name in enumerate(self.target_names):
                        log_payload[f"training_metrics/{name}"] = train_metrics[idx]
                    self.monitor.log_metrics(log_payload, global_step)

                if global_step % 100 == 0:
                    gc.collect()

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            val_loss = self._validate() if self.val_loader else None
            history["val_loss"].append(val_loss)

            self._save_checkpoint(epoch, global_step, avg_train_loss)

        self.monitor.close()
        return history

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            predictions = self.model(
                batch["activity_history"],
                batch["event_context_history"],
                batch["static_features"],
                batch["future_event_context"],
            )
            individual_losses, _ = self._compute_loss_and_metrics(predictions, batch["targets"])
            total_loss += individual_losses.sum().item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int, step: int, loss: float):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        save_obj = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        if self.loss_aggregator is not None:
            save_obj["loss_aggregator_state_dict"] = self.loss_aggregator.state_dict()
        if self.scheduler is not None:
            save_obj["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(save_obj, path)
