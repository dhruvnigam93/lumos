import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al., arXiv:1705.07115).

    Learns task-specific log-variance parameters to automatically balance
    multiple loss terms during training.
    """

    def __init__(self, is_regression: torch.Tensor, reduction: str = "sum"):
        super().__init__()
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, individual_losses: torch.Tensor) -> torch.Tensor:
        dtype = individual_losses.dtype
        device = individual_losses.device

        stds = (torch.exp(self.log_vars) ** 0.5).to(device).to(dtype)
        current_is_regression = self.is_regression.to(device=device, dtype=torch.float32)

        coeffs = 1.0 / ((current_is_regression + 1.0) * (stds**2))
        weighted_losses = coeffs * individual_losses + torch.log(stds)

        if self.reduction == "sum":
            return weighted_losses.sum()
        elif self.reduction == "mean":
            return weighted_losses.mean()
        return weighted_losses


class DirectLearnedWeightedLoss(nn.Module):
    """Learned direct-weight multi-task loss.

    Learns log-space weights per task: total = sum(exp(log_w_i) * loss_i).
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        self.n_tasks = n_tasks
        self.log_weights = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, individual_losses: torch.Tensor) -> torch.Tensor:
        weights = torch.exp(self.log_weights)
        return torch.sum(weights * individual_losses)
