import math

import torch


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "constant",
    total_steps: int = 0,
    warmup_steps: int = 0,
    min_lr_factor: float = 0.0,
    step_size: int = 1000,
    gamma: float = 0.1,
):
    """Factory for building a PyTorch LR scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: One of "constant", "cosine", "cosine_warmup", "step".
        total_steps: Total training steps (for cosine schedulers).
        warmup_steps: Linear warmup steps (for cosine_warmup).
        min_lr_factor: Minimum LR as a fraction of the initial LR.
        step_size: Step interval for StepLR.
        gamma: Multiplicative factor for StepLR.

    Returns:
        A scheduler instance, or None for "constant".
    """
    if scheduler_type == "constant":
        return None

    if scheduler_type == "cosine":
        base_lr = optimizer.param_groups[0]["lr"]
        eta_min = base_lr * min_lr_factor
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

    if scheduler_type == "cosine_warmup":

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
