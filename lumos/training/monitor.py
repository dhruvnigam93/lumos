import psutil


class TrainingMonitor:
    """Lightweight training metrics monitor with TensorBoard logging."""

    def __init__(self, tensorboard_dir: str | None = None, log_interval: int = 100):
        self.writer = None
        if tensorboard_dir:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(tensorboard_dir)
        self.log_interval = log_interval
        self.step = 0

    def log_metrics(self, metrics_dict: dict, step: int):
        if self.writer is not None:
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)

    def get_system_metrics(self) -> dict:
        metrics = {
            "cpu_utilization": psutil.cpu_percent(interval=None),
            "cpu_memory": psutil.virtual_memory().percent,
        }

        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem = torch.cuda.memory_allocated(i) / 1024**2
                    metrics[f"gpu_{i}_memory_mb"] = mem
        except ImportError:
            pass

        return metrics

    def close(self):
        if self.writer is not None:
            self.writer.close()
